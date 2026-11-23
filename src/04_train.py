import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

from src import config as C
from src.common import IMG_SIZE, MEAN, STD, build_model, eval_transform


def train_transform():
    """Augmentations for real shelf photos (reasons in comments; summary in PROGRESS.md)."""
    return T.Compose([
        # framing/scale differ a lot on a shelf; scale>=0.5 keeps the vegetable in the crop
        T.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0), ratio=(0.8, 1.25)),
        T.RandomHorizontalFlip(0.5),                       # vegetables have no handedness
        T.RandomApply([T.RandomRotation(20)], p=0.5),      # tilted phone / tilted produce
        # shop lighting differs; also fights per-day colour/background shortcut found in Step 3.
        # hue kept tiny because colour is a real class cue (orange carrot, green cucumber)
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.03),
        T.RandomGrayscale(0.05),                           # rare: reduces colour reliance, not removes it
        # standalone photos are ~12x smoother than video frames (Laplacian var, Step 3) and 95% of
        # train is frames -> blur bridges that gap and also mimics out-of-focus shop photos
        T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.5),
        T.ToTensor(), T.Normalize(MEAN, STD),
        # occlusion: price tags, hands, neighbouring produce
        T.RandomErasing(p=0.25, scale=(0.02, 0.15), value=0),
    ])


class DS(Dataset):
    def __init__(self, df, tf, images_root=None):
        self.tf, self.imgs = tf, []
        for r in df.itertuples():
            p = Path(r.path)
            if not p.exists() and images_root:
                p = Path(images_root) / r.cls / p.name
            with Image.open(p) as im:
                self.imgs.append(im.convert("RGB"))
        self.y = np.array([C.CLASSES.index(c) for c in df["cls"]])
        self.w = df["weight"].values.astype(np.float32)
        self.is_frame = df["is_frame"].astype(bool).values

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.tf(self.imgs[i]), int(self.y[i]), float(self.w[i])


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    lg, ys, ws = [], [], []
    for x, y, w in loader:
        lg.append(model(x.to(device)).float().cpu()); ys.append(y); ws.append(w)
    lg, y, w = torch.cat(lg), torch.cat(ys), torch.cat(ws)
    ce = F.cross_entropy(lg, y, reduction="none")
    pred = lg.argmax(1).numpy()
    yn, wn = y.numpy(), w.numpy()
    lab = list(range(len(C.CLASSES)))
    m = dict(val_loss=ce.mean().item(), val_loss_w=(ce * w).sum().item() / w.sum().item(),
             val_acc=float((pred == yn).mean()), val_acc_w=float(((pred == yn) * wn).sum() / wn.sum()),
             val_f1=f1_score(yn, pred, labels=lab, average="macro", zero_division=0),
             val_f1_w=f1_score(yn, pred, labels=lab, average="macro", sample_weight=wn, zero_division=0))
    return m, pred, yn, wn


def set_trainable(model, modules):
    for p in model.parameters():
        p.requires_grad = False
    for m in modules:
        for p in m.parameters():
            p.requires_grad = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "mobilenet_v3_small", "efficientnet_b0"])
    ap.add_argument("--split-file", default=str(C.SPLITS / "split.csv"))
    ap.add_argument("--images-root", default=None)
    ap.add_argument("--epochs1", type=int, default=3)
    ap.add_argument("--epochs2", type=int, default=12)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-block", type=float, default=1e-4)
    ap.add_argument("--lr-head2", type=float, default=3e-4)
    ap.add_argument("--photo-boost", type=float, default=3.0)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--max-minutes", type=float, default=30)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default=str(C.MODELS / "best.pt"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for log/summary/figure files, e.g. _official")
    ap.add_argument("--final", action="store_true", help="train on ALL rows (train+val+test), fixed 3+2 epochs, keep last epoch (shipped model)")
    a = ap.parse_args()

    random.seed(C.SEED); np.random.seed(C.SEED); torch.manual_seed(C.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    sp = pd.read_csv(a.split_file)
    tr, va = sp[sp.split == "train"], sp[sp.split == "val"]  # test rows are never used here (except --final)
    if a.final:
        tr, a.epochs1, a.epochs2 = sp, 3, 2   # best.pt was selected at epoch 5 = 3 head + 2 fine-tune epochs
        print("FINAL MODE: training on ALL rows incl. test; val numbers below are NOT held-out")
    if a.smoke:
        tr, va = tr.sample(96, random_state=C.SEED), va.sample(48, random_state=C.SEED)
        a.epochs1, a.epochs2 = 1, 1
    dtr = DS(tr, train_transform(), a.images_root)
    dva = DS(va, eval_transform(), a.images_root)
    print(f"train {len(dtr)} | val {len(dva)} | train photos {int((~dtr.is_frame).sum())}")

    # photos are the closest thing to real shop photos but only ~5% of train -> sample them more often
    sw = np.where(dtr.is_frame, 1.0, a.photo_boost)
    sampler = WeightedRandomSampler(sw, num_samples=len(dtr), replacement=True,
                                    generator=torch.Generator().manual_seed(C.SEED))
    ltr = DataLoader(dtr, a.bs, sampler=sampler, num_workers=a.workers)
    lva = DataLoader(dva, 64, shuffle=False, num_workers=a.workers)

    # class-weighted loss (imbalance up to ~5:1 in train): inverse frequency, mean-normalised
    cnt = np.bincount(dtr.y, minlength=len(C.CLASSES)).astype(float)
    cw = cnt.sum() / (len(cnt) * np.maximum(cnt, 1))
    print("train class counts:", dict(zip(C.CLASSES, cnt.astype(int))), "| loss weights:", dict(zip(C.CLASSES, cw.round(2))))
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=device),
                               label_smoothing=a.label_smoothing)

    model, head, block = build_model(a.arch, pretrained=True)
    model.to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")
    stages = [("head", a.epochs1, [head], [(head, a.lr_head)]),
              ("finetune", a.epochs2, [block, head], [(block, a.lr_block), (head, a.lr_head2)])]

    log, best, best_state, t0, stop, ep = [], -1.0, None, time.time(), False, 0
    for sname, n_ep, mods, groups in stages:
        set_trainable(model, [m for g in mods for m in g])
        opt = torch.optim.AdamW([{"params": [p for m in g for p in m.parameters()], "lr": lr}
                                 for g, lr in groups], weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(n_ep, 1))
        bad = 0
        for _ in range(n_ep):
            ep += 1
            model.eval()                                    # keep BatchNorm stats of frozen parts fixed
            for g in mods:
                for m in g:
                    m.train()
            tl, tc, tn, te = 0.0, 0, 0, time.time()
            for x, y, _w in ltr:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                with torch.autocast(device_type=device, enabled=device == "cuda"):
                    out = model(x)
                    loss = crit(out.float(), y)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                tl += loss.item() * len(y); tc += (out.argmax(1) == y).sum().item(); tn += len(y)
            sched.step()
            m, *_ = evaluate(model, lva, device)
            row = dict(epoch=ep, stage=sname, lr=opt.param_groups[0]["lr"], train_loss=tl / tn, train_acc=tc / tn, **m)
            log.append(row)
            # early stopping / model selection: group-balanced val macro-F1 (val is tiny and has few groups
            # for 4 classes; plain F1 would be dominated by Carrot/Squash frames). Tie-break: lower weighted loss.
            score = m["val_f1_w"] - 1e-3 * m["val_loss_w"]
            improved = True if a.final else score > best
            if improved:
                best, bad = score, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                row["best"] = True
            else:
                bad += 1
            print(f"ep{ep:02d} [{sname}] {time.time()-te:5.1f}s train_loss {row['train_loss']:.3f} acc {row['train_acc']:.3f} | "
                  f"val loss {m['val_loss']:.3f} acc {m['val_acc']:.3f} F1 {m['val_f1']:.3f} F1_w {m['val_f1_w']:.3f}"
                  f"{'  *best*' if improved else ''}")
            if sname == "finetune" and bad >= a.patience:
                print("early stopping"); stop = True
            if (time.time() - t0) / 60 > a.max_minutes:
                print("time budget reached"); stop = True
            if stop:
                break
        if stop:
            break

    # ---- save best + final val report
    model.load_state_dict(best_state)
    m, pred, yn, wn = evaluate(model, lva, device)
    lab = list(range(len(C.CLASSES)))
    cm = confusion_matrix(yn, pred, labels=lab)
    rec = {c: (float(cm[i, i] / cm[i].sum()) if cm[i].sum() else None) for i, c in enumerate(C.CLASSES)}
    print("\nBest model on VAL:", {k: round(v, 4) for k, v in m.items()})
    print("val recall per class:", {k: (round(v, 3) if v is not None else None) for k, v in rec.items()})
    print("val confusion matrix (rows=true):\n", cm)
    photo = ~dva.is_frame
    if photo.any():
        print(f"val acc frames {float((pred == yn)[~photo].mean()):.3f} (n={int((~photo).sum())}) | photos {float((pred == yn)[photo].mean()):.3f} (n={int(photo.sum())})")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(arch=a.arch, state_dict=best_state, classes=C.CLASSES, img_size=IMG_SIZE,
                    val_metrics=m, config=vars(a)), a.out)
    pd.DataFrame(log).to_csv(C.OUTPUTS / f"train_log{a.tag}.csv", index=False)
    json.dump(dict(config=vars(a), device=device, minutes=round((time.time() - t0) / 60, 1),
                   epochs_run=ep, best_val=m, val_recall=rec, class_weights=dict(zip(C.CLASSES, cw.round(3).tolist())),
                   train_counts=dict(zip(C.CLASSES, cnt.astype(int).tolist()))),
              open(C.OUTPUTS / f"train_summary{a.tag}.json", "w"), indent=2)

    L = pd.DataFrame(log)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].plot(L.epoch, L.train_loss, label="train (class-weighted)"); ax[0].plot(L.epoch, L.val_loss, label="val"); ax[0].set_title("loss"); ax[0].legend()
    ax[1].plot(L.epoch, L.val_f1, label="macro-F1"); ax[1].plot(L.epoch, L.val_f1_w, label="macro-F1 group-balanced"); ax[1].set_title("val macro-F1"); ax[1].legend()
    ax[2].plot(L.epoch, L.train_acc, label="train"); ax[2].plot(L.epoch, L.val_acc, label="val"); ax[2].set_title("accuracy"); ax[2].legend()
    for x in ax:
        x.axvline(a.epochs1 + 0.5, color="gray", ls="--", lw=0.8); x.set_xlabel("epoch")
    plt.tight_layout(); plt.savefig(C.OUTPUTS / f"fig_train_curves{a.tag}.png", dpi=130); plt.close()
    print(f"Saved {a.out} | total {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
