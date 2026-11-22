"""Step 6: Grad-CAM + background-shortcut tests (uses models/best.pt and outputs/test_predictions.csv).
  python src/06_gradcam.py
Outputs: outputs/gradcam/*.png (per-example figures, overview, Capsicum comparison), gradcam_stats.csv,
         shortcut_tests.csv, gradcam_summary.json  (+ printed tables)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src import config as C
from src.common import IMG_SIZE, eval_transform, load_model

GC = C.OUTPUTS / "gradcam"
GC.mkdir(parents=True, exist_ok=True)
IDX = {c: i for i, c in enumerate(C.CLASSES)}
B0, B1 = int(0.2 * IMG_SIZE), int(0.8 * IMG_SIZE)  # "rough centre" = central 60% x 60% box (36% of the area)


def center_mask():
    m = np.zeros((IMG_SIZE, IMG_SIZE), bool)
    m[B0:B1, B0:B1] = True
    return m


def target_layer(model, arch):
    return {"resnet18": lambda: model.layer4[-1], "mobilenet_v3_small": lambda: model.features[-1],
            "efficientnet_b0": lambda: model.features[-1]}[arch]()


def open_img(path, cls, root):
    p = Path(path)
    if not p.exists() and root:
        p = Path(root) / cls / p.name
    with Image.open(p) as im:
        return im.convert("RGB")


def to_rgb(im):
    return np.asarray(T.CenterCrop(IMG_SIZE)(T.Resize(IMG_SIZE)(im)), dtype=np.float32) / 255.0


def cam_maps(cam, x, class_idx):
    return cam(input_tensor=x, targets=[ClassifierOutputTarget(int(c)) for c in class_idx])


def example(cam, row, kind, k, tf, device, root):
    im = open_img(row["path"], row["cls"], root)
    rgb, x = to_rgb(im), tf(im).unsqueeze(0).to(device)
    pi, ti = IDX[row["pred"]], IDX[row["cls"]]
    over_p = show_cam_on_image(rgb, cam_maps(cam, x, [pi])[0], use_rgb=True)
    panels = [(rgb, "image"), (over_p, f"CAM: predicted {row['pred']}")]
    if kind == "error":
        panels.append((show_cam_on_image(rgb, cam_maps(cam, x, [ti])[0], use_rgb=True), f"CAM: true {row['cls']}"))
    fig, ax = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.9))
    for a_, (img, t) in zip(np.atleast_1d(ax), panels):
        a_.imshow(img); a_.axis("off"); a_.set_title(t, fontsize=9)
    fig.suptitle(f"[{kind}] true={row['cls']}  pred={row['pred']} ({row['conf']:.2f})  {Path(row['path']).name[:30]}", fontsize=9)
    plt.tight_layout()
    plt.savefig(GC / f"ex{k:02d}_{kind}_{row['cls']}_as_{row['pred']}.png", dpi=110); plt.close()
    return over_p, f"{kind}: {row['cls']} -> {row['pred']} ({row['conf']:.2f})"


def cam_stats(cam, df, tf, device, root, M, kind):
    rows, area_out = [], 1.0 - M.mean()
    for i in range(0, len(df), 16):
        b = df.iloc[i:i + 16]
        x = torch.stack([tf(open_img(r.path, r.cls, root)) for r in b.itertuples()]).to(device)
        maps = cam_maps(cam, x, [IDX[p] for p in b["pred"]])
        for r, m in zip(b.itertuples(), maps):
            m = m / (m.sum() + 1e-8)
            out = float(1.0 - m[M].sum())
            rows.append(dict(path=r.path, cls=r.cls, pred=r.pred, correct=bool(r.cls == r.pred), group=r.group,
                             outside_share=out, outside_ratio=out / area_out))
    return pd.DataFrame(rows)


@torch.no_grad()
def shortcut_tests(model, te, tf, device, root, M):
    """Accuracy when (a) background is replaced by mean-grey / blurred (keep centre box),
    (b) the centre box is replaced by mean-grey (keep background only)."""
    Mt = torch.tensor(M)
    conds = ["orig", "bg_gray", "bg_blur", "center_gray"]
    P = {c: [] for c in conds}
    for i in range(0, len(te), 64):
        b = te.iloc[i:i + 64]
        x = torch.stack([tf(open_img(r.path, r.cls, root)) for r in b.itertuples()]).to(device)
        g = torch.zeros_like(x)  # zero in normalised space = dataset-mean colour
        xs = {"orig": x, "bg_gray": torch.where(Mt.to(device), x, g),
              "bg_blur": torch.where(Mt.to(device), x, TF.gaussian_blur(x, kernel_size=31, sigma=8.0)),
              "center_gray": torch.where(Mt.to(device), g, x)}
        for c in conds:
            P[c].append(model(xs[c]).argmax(1).cpu().numpy())
    return {c: np.concatenate(v) for c, v in P.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(C.MODELS / "best.pt"))
    ap.add_argument("--images-root", default=None)
    ap.add_argument("--n-per-class", type=int, default=20)
    ap.add_argument("--n-errors", type=int, default=30)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load_model(a.model, device)
    tf, M, rng = eval_transform(), center_mask(), np.random.RandomState(C.SEED)
    tp = pd.read_csv(C.OUTPUTS / "test_predictions.csv")
    tp["correct"] = tp["cls"] == tp["pred"]
    tp["group"] = tp["group"].astype(str)
    fail = tp[tp.is_frame].groupby("group").correct.mean().idxmin()
    print(f"weakest test video: {fail} (acc {tp[tp.group == fail].correct.mean():.3f}, n={int((tp.group == fail).sum())})")

    # ---------- which examples
    ex = []
    for c in ["Capsicum", "Carrot", "Cucumber", "Raddish", "Squash"]:
        pool = tp[(tp.cls == c) & tp.correct]
        if c == "Capsicum" and (pool.group == fail).any():
            pool = pool[pool.group == fail]  # correct frames of the SAME failing video, for contrast
        ex.append(("correct", pool.iloc[rng.randint(len(pool))]))
    fe = tp[(~tp.correct) & (tp.group == fail)].sort_values("conf", ascending=False)
    for q in (0, len(fe) // 2, len(fe) - 1):
        if len(fe):
            ex.append(("error", fe.iloc[q]))
    for c in ["Potato", "Squash"]:
        e = tp[(~tp.correct) & (tp.cls == c)].sort_values("conf", ascending=False)
        if len(e):
            ex.append(("error", e.iloc[0]))

    with GradCAM(model=model, target_layers=[target_layer(model, ck["arch"])]) as cam:
        overs = [example(cam, r, k, i, tf, device, a.images_root) for i, (k, r) in enumerate(ex)]
        # overview grid
        n = len(overs)
        cols = 5
        rows_ = int(np.ceil(n / cols))
        fig, ax = plt.subplots(rows_, cols, figsize=(3.2 * cols, 3.5 * rows_))
        for a_ in np.atleast_2d(ax).ravel():
            a_.axis("off")
        for a_, (img, t) in zip(np.atleast_2d(ax).ravel(), overs):
            a_.imshow(img); a_.set_title(t, fontsize=8)
        plt.tight_layout(); plt.savefig(GC / "overview.png", dpi=110); plt.close()

        # failing Capsicum frames vs reference Capsicum training frames
        sp = pd.read_csv(C.SPLITS / "split.csv")
        refs = []
        trc = sp[(sp.split == "train") & (sp.cls == "Capsicum")]
        for g_, gg in list(trc[trc.is_frame.astype(bool)].groupby("group")) + [("photo", trc[~trc.is_frame.astype(bool)])]:
            if len(gg):
                refs.append(gg.iloc[rng.randint(len(gg))])
        fr = [r for k, r in ex if k == "error" and r["group"] == fail][:3]
        fig, ax = plt.subplots(4, max(len(fr), len(refs)), figsize=(3.2 * max(len(fr), len(refs)), 12.5), squeeze=False)
        for a_ in ax.ravel():
            a_.axis("off")
        for j, r in enumerate(fr):
            im = open_img(r["path"], r["cls"], a.images_root); rgb = to_rgb(im)
            ax[0, j].imshow(rgb); ax[0, j].set_title(f"FAILING video {fail}\nreal Capsicum -> {r['pred']} ({r['conf']:.2f})", fontsize=8)
            ax[1, j].imshow(show_cam_on_image(rgb, cam_maps(cam, tf(im).unsqueeze(0).to(device), [IDX[r['pred']]])[0], use_rgb=True))
        for j, r in enumerate(refs):
            im = open_img(r["path"], r["cls"], a.images_root); rgb = to_rgb(im); x = tf(im).unsqueeze(0).to(device)
            with torch.no_grad():
                pr = torch.softmax(model(x), 1)[0]
            p = int(pr.argmax())
            ax[2, j].imshow(rgb); ax[2, j].set_title(f"TRAIN Capsicum ({str(r['group'])[:15]})\npred {C.CLASSES[p]} ({pr[p]:.2f})", fontsize=8)
            ax[3, j].imshow(show_cam_on_image(rgb, cam_maps(cam, x, [p])[0], use_rgb=True))
        plt.tight_layout(); plt.savefig(GC / "compare_capsicum_failing_vs_train.png", dpi=110); plt.close()

        # ---------- quantitative: CAM mass outside the central box
        samp = pd.concat([tp[tp.cls == c].sample(min(a.n_per_class, int((tp.cls == c).sum())), random_state=C.SEED)
                          for c in C.CLASSES])
        errs = tp[~tp.correct]
        samp = pd.concat([samp, errs.sample(min(a.n_errors, len(errs)), random_state=C.SEED)]).drop_duplicates("path")
        st = cam_stats(cam, samp, tf, device, a.images_root, M, "")
    st.to_csv(C.OUTPUTS / "gradcam_stats.csv", index=False)
    area_out = 1.0 - M.mean()
    pd.set_option("display.width", 200)
    print(f"\n== CAM mass outside the central 60%x60% box (uniform CAM would give {area_out:.2f}; ratio 1.0 = no centre preference) ==")
    print(f"all {len(st)} images: share {st.outside_share.mean():.3f}, ratio {st.outside_ratio.mean():.2f}")
    print(st.groupby("correct")[["outside_share", "outside_ratio"]].mean().round(3).to_string())
    print(st.groupby("cls")[["outside_share", "outside_ratio"]].mean().round(3).to_string())

    # ---------- quantitative: prediction change when background / centre is removed (all test images)
    P = shortcut_tests(model, tp, tf, device, a.images_root, M)
    y = tp["cls"].map(IDX).values
    rows = []
    for c, p in P.items():
        for name, mask in [("ALL", np.ones(len(y), bool))] + [(k, (tp.cls == k).values) for k in C.CLASSES] + \
                          [("frames", tp.is_frame.astype(bool).values), ("photos", ~tp.is_frame.astype(bool).values)]:
            rows.append(dict(condition=c, subset=name, n=int(mask.sum()), acc=float((p[mask] == y[mask]).mean()),
                             unchanged_vs_orig=float((p[mask] == P["orig"][mask]).mean())))
    sh = pd.DataFrame(rows)
    sh.to_csv(C.OUTPUTS / "shortcut_tests.csv", index=False)
    print("\n== Accuracy on all test images (n=%d) when parts are removed ==" % len(y))
    print("  bg_gray/bg_blur: only the central box is kept | center_gray: the central box is hidden, background kept")
    print(sh[sh.subset == "ALL"][["condition", "acc", "unchanged_vs_orig"]].round(3).to_string(index=False))
    print("\nAccuracy per class:\n", sh[sh.subset.isin(C.CLASSES)].pivot(index="subset", columns="condition", values="acc")[["orig", "bg_gray", "bg_blur", "center_gray"]].round(3).to_string())
    print("\nframes vs photos:\n", sh[sh.subset.isin(["frames", "photos"])].pivot(index="subset", columns="condition", values="acc")[["orig", "bg_gray", "bg_blur", "center_gray"]].round(3).to_string())
    print("\nWhat the centre-hidden model predicts (counts), all test images:")
    print(pd.crosstab(tp["cls"], np.array(C.CLASSES)[P["center_gray"]]).to_string())

    # ---------- why does the weak video fail? border colour / brightness of videos (from Step 3 features)
    F = pd.read_csv(C.OUTPUTS / "domain_features.csv")
    Fv = F[F.is_frame.astype(bool) & F.cls.isin(["Capsicum", "Cucumber"])]
    t = Fv.groupby(["cls", "unit", "date", "split"]).agg(n=("path", "size"), bright=("bright_mean", "mean"),
                                                        bR=("border_r", "mean"), bG=("border_g", "mean"), bB=("border_b", "mean")).round(1)
    print("\nVideos of Capsicum and Cucumber (mean brightness and border RGB):\n", t.to_string())
    json.dump(dict(weakest_video=fail, outside_share_mean=float(st.outside_share.mean()), outside_ratio_mean=float(st.outside_ratio.mean()),
                   shortcut_acc={c: float((p == y).mean()) for c, p in P.items()}),
              open(C.OUTPUTS / "gradcam_summary.json", "w"), indent=2)
    print("\nFigures: outputs/gradcam/overview.png, ex*.png, compare_capsicum_failing_vs_train.png")


if __name__ == "__main__":
    main()
