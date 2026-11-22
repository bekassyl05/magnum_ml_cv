"""Step 8b: 'unknown' threshold (bonus 1).  Needs data/ood (python src/08_get_ood.py).
    python src/08_unknown.py [--weights models/best.pt] [--retention 0.95]
Selection data: val images of the 6 classes (in-distribution) + OOD set (other vegetables / own photos).
Independent checks (not used for selection): test images, frames without any vegetable.
Outputs: outputs/ood_threshold.png, outputs/ood_thresholds.csv, models/unknown_config.json (read by predict.py)
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp, softmax
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

import predict
from src import config as C

OOD = ROOT / "data" / "ood"
IDX = {c: i for i, c in enumerate(C.CLASSES)}
EXT = {".jpg", ".jpeg", ".png", ".webp"}


@torch.inference_mode()
def logits_of(model, paths, skip_bad=False, bs=64):
    out = []
    for i in range(0, len(paths), bs):
        xs = []
        for p in paths[i:i + bs]:
            try:
                xs.append(predict.load_tensor(p))
            except Exception:
                if not skip_bad:
                    raise
        if xs:
            out.append(model(torch.stack(xs)).numpy())
    return np.concatenate(out)


def scores(lg, T):
    """Higher = looks more like one of the 6 classes."""
    return {"msp": softmax(lg, 1).max(1), "msp_T": softmax(lg / T, 1).max(1), "energy": logsumexp(lg, 1)}


def fit_T(lg, y):
    grid = np.geomspace(0.1, 5, 120)
    nll = [-np.log(softmax(lg / t, 1)[np.arange(len(y)), y] + 1e-12).mean() for t in grid]
    return float(grid[int(np.argmin(nll))])


def ece(conf, ok, bins=10):
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "models" / "best.pt"))
    ap.add_argument("--retention", type=float, default=0.95, help="share of val in-distribution images that must be kept")
    a = ap.parse_args()
    sp = pd.read_csv(C.SPLITS / "split.csv")
    val, te = sp[sp.split == "val"], sp[sp.split == "test"]
    model = predict.load_model(a.weights)
    lv, lt = logits_of(model, list(val.path)), logits_of(model, list(te.path))
    yv, yt = val.cls.map(IDX).values, te.cls.map(IDX).values
    sets = {}
    for name in ("other_veg", "no_veg_frames", "own"):
        fs = sorted(p for p in (OOD / name).glob("*") if p.suffix.lower() in EXT) if (OOD / name).exists() else []
        if fs:
            sets[name] = logits_of(model, [str(f) for f in fs], skip_bad=True)
            print(f"{name}: {len(sets[name])} images")
    if "other_veg" not in sets:
        sys.exit("no OOD images: run python src/08_get_ood.py first")

    T = fit_T(lv, yv)
    print(f"temperature T={T:.2f} fitted on val; ECE val {ece(softmax(lv, 1).max(1), lv.argmax(1) == yv):.3f} -> "
          f"{ece(softmax(lv / T, 1).max(1), lv.argmax(1) == yv):.3f}; test {ece(softmax(lt, 1).max(1), lt.argmax(1) == yt):.3f} -> "
          f"{ece(softmax(lt / T, 1).max(1), lt.argmax(1) == yt):.3f}")
    Sv, St = scores(lv, T), scores(lt, T)
    So = {k: scores(v, T) for k, v in sets.items()}
    dev_lg = np.concatenate([sets["other_veg"]] + ([sets["own"]] if "own" in sets else []))
    Sd = scores(dev_lg, T)
    y = np.r_[np.ones(len(lv)), np.zeros(len(dev_lg))]
    res = {s: dict(auroc=roc_auc_score(y, np.r_[Sv[s], Sd[s]]), aupr=average_precision_score(y, np.r_[Sv[s], Sd[s]])) for s in Sv}
    print("AUROC/AUPR (val ID vs dev OOD):", {k: (round(v["auroc"], 3), round(v["aupr"], 3)) for k, v in res.items()})
    best = max(res, key=lambda k: (round(res[k]["auroc"], 3), k == "msp"))  # ties -> plain max softmax
    print("chosen score:", best)

    rows = []
    for r in (0.99, 0.975, a.retention, 0.90):
        th = float(np.quantile(Sv[best], 1 - r))
        keep = St[best] >= th
        row = dict(val_ID_target=r, threshold=round(th, 4), val_ID_kept=(Sv[best] >= th).mean(),
                   other_veg_rejected=(So["other_veg"][best] < th).mean())
        for k in ("own", "no_veg_frames"):
            if k in So:
                row[f"{k}_rejected"] = (So[k][best] < th).mean()
        row["test_ID_kept"] = keep.mean()
        row["test_acc_of_kept"] = (lt.argmax(1)[keep] == yt[keep]).mean()
        rows.append(row)
    tab = pd.DataFrame(rows).round(3)
    tab.to_csv(C.OUTPUTS / "ood_thresholds.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"\nThreshold table (chosen row: val_ID_target = {a.retention}):\n", tab.to_string(index=False))
    thr = float(np.quantile(Sv[best], 1 - a.retention))
    kept = pd.Series(St[best] >= thr).groupby(te.group.astype(str).values).mean()
    print("\nTest: share kept per video/photo group at the chosen threshold (lowest 6):\n", kept.sort_values().head(6).round(3).to_string())

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    allv = np.concatenate([Sv[best], St[best]] + [So[k][best] for k in So])
    bins = np.linspace(allv.min(), allv.max(), 40)
    for lab, v in [("val ID", Sv[best]), ("test ID", St[best])] + [(k, So[k][best]) for k in So]:
        ax[0].hist(v, bins=bins, histtype="step", density=True, label=f"{lab} (n={len(v)})")
    ax[0].axvline(thr, color="k", ls="--")
    ax[0].set_title(f"score = {best}; dashed = threshold {thr:.3f}")
    ax[0].legend(fontsize=7)
    for s in Sv:
        fpr, tpr, _ = roc_curve(y, np.r_[Sv[s], Sd[s]])
        ax[1].plot(fpr, tpr, label=f"{s} AUROC {res[s]['auroc']:.3f}")
    ax[1].plot([0, 1], [0, 1], "gray", lw=0.7)
    ax[1].set_xlabel("OOD accepted (FPR)")
    ax[1].set_ylabel("val ID kept (TPR)")
    ax[1].set_title("ROC: val ID vs OOD (other veg / own)")
    ax[1].legend()
    g = np.linspace(allv.min(), allv.max(), 200)
    ax[2].plot(g, [(Sv[best] >= t).mean() for t in g], label="val ID kept")
    ax[2].plot(g, [(St[best] >= t).mean() for t in g], label="test ID kept")
    for k in So:
        ax[2].plot(g, [(So[k][best] < t).mean() for t in g], label=f"{k} rejected")
    ax[2].axvline(thr, color="k", ls="--")
    ax[2].set_xlabel("threshold")
    ax[2].set_title("retention / rejection vs threshold")
    ax[2].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(C.OUTPUTS / "ood_threshold.png", dpi=120)
    plt.close()

    cfg = dict(score=best, threshold=thr, temperature=T, criterion=f"val ID retention >= {a.retention}",
               calibrated_on=f"{Path(a.weights).name}, val split + OOD set (other_veg{' + own' if 'own' in sets else ''})",
               auroc=round(res[best]["auroc"], 4))
    (ROOT / "models" / "unknown_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print("\nSaved models/unknown_config.json:", cfg)


if __name__ == "__main__":
    main()
