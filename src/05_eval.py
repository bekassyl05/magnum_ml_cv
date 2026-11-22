"""Step 5: evaluation on the held-out test split (run ONCE per model, no tuning on test).
  python src/05_eval.py                                   # best.pt on my group-aware test split
  python src/05_eval.py --make-official-split             # writes splits/official_split.csv (Kaggle's split)
  python src/05_eval.py --model models/official.pt --split-file splits/official_split.csv --tag _official
Outputs (outputs/): test_metrics{tag}.json, test_per_class{tag}.csv, test_per_group{tag}.csv,
  test_predictions{tag}.csv, errors{tag}.csv, fig_confusion{tag}.png
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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from src import config as C
from src.common import load_model, predict_df

LAB = list(range(len(C.CLASSES)))


def make_official_split():
    """Kaggle's final_dataset/{train,val,test} in the same format as splits/split.csv."""
    cl = pd.read_csv(C.OUTPUTS / "clusters.csv")
    o = cl[cl.source.isin(["train", "val", "test"])].copy()
    o["is_frame"] = o["filename"].map(lambda f: C.parse_name(f)["is_frame"])
    o["split"] = o["source"]
    o["weight"] = 1.0 / o.groupby(["split", "group"])["md5"].transform("size")
    out = o[["path", "cls", "group", "md5", "split", "weight", "is_frame"]]
    out.to_csv(C.SPLITS / "official_split.csv", index=False)
    print("Saved splits/official_split.csv:", out.split.value_counts().to_dict())


def summarize(y, pred, w):
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=LAB, zero_division=0)
    per = pd.DataFrame(dict(cls=C.CLASSES, precision=p, recall=r, f1=f, support=s))
    m = dict(acc=accuracy_score(y, pred), macro_f1=f1_score(y, pred, labels=LAB, average="macro", zero_division=0),
             acc_group_balanced=accuracy_score(y, pred, sample_weight=w),
             macro_f1_group_balanced=f1_score(y, pred, labels=LAB, average="macro", sample_weight=w, zero_division=0))
    return m, per


def bootstrap_ci(y, pred, groups, n=1000, seed=C.SEED):
    """Cluster bootstrap: resample whole groups (videos/photos), because frames are not independent."""
    rng = np.random.RandomState(seed)
    ug = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in ug}
    accs, f1s = [], []
    for _ in range(n):
        idx = np.concatenate([idx_by[g] for g in rng.choice(ug, len(ug), replace=True)])
        accs.append((y[idx] == pred[idx]).mean())
        f1s.append(f1_score(y[idx], pred[idx], labels=np.unique(y[idx]), average="macro", zero_division=0))
    return [float(v) for v in np.percentile(accs, [2.5, 97.5])], [float(v) for v in np.percentile(f1s, [2.5, 97.5])]


def ece(conf, correct, bins=10):
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def confusion_fig(y, pred, title, fname):
    cm = confusion_matrix(y, pred, labels=LAB)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    for a_, mat, ttl, fmt in ((ax[0], cm, "counts", "d"), (ax[1], cmn, "row-normalised (recall)", ".2f")):
        a_.imshow(mat, cmap="Blues")
        a_.set_xticks(LAB); a_.set_yticks(LAB)
        a_.set_xticklabels(C.CLASSES, rotation=45, ha="right"); a_.set_yticklabels(C.CLASSES)
        a_.set_xlabel("predicted"); a_.set_ylabel("true"); a_.set_title(f"{title}: {ttl}")
        for i in LAB:
            for j in LAB:
                a_.text(j, i, format(mat[i, j], fmt), ha="center", va="center",
                        color="white" if mat[i, j] > mat.max() / 2 else "black", fontsize=8)
    plt.tight_layout(); plt.savefig(C.OUTPUTS / fname, dpi=130); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(C.MODELS / "best.pt"))
    ap.add_argument("--split-file", default=str(C.SPLITS / "split.csv"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--images-root", default=None)
    ap.add_argument("--make-official-split", action="store_true")
    a = ap.parse_args()
    if a.make_official_split:
        return make_official_split()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sp = pd.read_csv(a.split_file)
    te = sp[sp.split == "test"].reset_index(drop=True)
    model, ck = load_model(a.model, device)
    probs = predict_df(model, te, device, a.images_root)
    pred, conf = probs.argmax(1), probs.max(1)
    y = te["cls"].map({c: i for i, c in enumerate(C.CLASSES)}).values
    w = te["weight"].values
    isf = te["is_frame"].astype(bool).values
    groups = te["group"].astype(str).values
    correct = pred == y

    m, per = summarize(y, pred, w)
    ci_acc, ci_f1 = bootstrap_ci(y, pred, groups)
    m.update(n_test=len(te), n_groups=int(len(np.unique(groups))), acc_ci95_group_bootstrap=ci_acc,
             macro_f1_ci95_group_bootstrap=ci_f1, ece=ece(conf, correct),
             majority_class_baseline=float(np.bincount(y).max() / len(y)),
             mean_conf_correct=float(conf[correct].mean()), mean_conf_wrong=float(conf[~correct].mean()) if (~correct).any() else None)
    for name, mask in (("frames", isf), ("photos", ~isf)):
        if mask.any():
            mm, _ = summarize(y[mask], pred[mask], w[mask])
            m[f"{name}_n"] = int(mask.sum()); m[f"{name}_acc"] = mm["acc"]; m[f"{name}_macro_f1"] = mm["macro_f1"]
    # memorisation check: test files that also exist (identical bytes) in train
    if "md5" in sp.columns and (sp.split == "train").any():
        dup = te["md5"].isin(set(sp.loc[sp.split == "train", "md5"])).values
        m["test_in_train_exact_n"] = int(dup.sum())
        if dup.any():
            m["acc_on_files_also_in_train"] = float(correct[dup].mean())
        if (~dup).any():
            m["acc_on_files_not_in_train"] = float(correct[~dup].mean())

    # per-group table (each video separately, photos aggregated per class)
    key = np.where(isf, "video:" + groups, "photos:" + te["cls"].values)
    rows = []
    for k in np.unique(key):
        mk = key == k
        wrong = pd.Series(np.array(C.CLASSES)[pred[mk & ~correct]])
        rows.append(dict(key=k, cls=te["cls"].values[mk][0], n=int(mk.sum()), acc=float(correct[mk].mean()),
                         most_common_wrong=wrong.value_counts().idxmax() if len(wrong) else ""))
    pg = pd.DataFrame(rows).sort_values(["cls", "key"])

    pd.set_option("display.width", 220)
    print(f"\n=== TEST ({a.split_file}) model {a.model}: n={len(te)}, groups={m['n_groups']} ===")
    for k, v in m.items():
        print(f"  {k}: {round(v, 4) if isinstance(v, float) else v}")
    print("\nPer class:\n", per.round(3).to_string(index=False))
    print("\nPer video / photo-group:\n", pg.round(3).to_string(index=False))

    out_pred = te[["path", "cls", "group", "is_frame"]].copy()
    out_pred["pred"] = np.array(C.CLASSES)[pred]; out_pred["conf"] = conf.round(4)
    for i, c in enumerate(C.CLASSES):
        out_pred[f"p_{c}"] = probs[:, i].round(4)
    out_pred.to_csv(C.OUTPUTS / f"test_predictions{a.tag}.csv", index=False)
    err = out_pred[~correct].copy()
    err["p_true"] = probs[~correct, y[~correct]].round(4)
    err = err.sort_values("conf", ascending=False)
    err.to_csv(C.OUTPUTS / f"errors{a.tag}.csv", index=False)
    print(f"\nErrors: {len(err)} of {len(te)}; top-10 most confident mistakes:\n",
          err[["path", "cls", "pred", "conf"]].head(10).assign(path=lambda d: d.path.str[-40:]).to_string(index=False))
    per.to_csv(C.OUTPUTS / f"test_per_class{a.tag}.csv", index=False)
    pg.to_csv(C.OUTPUTS / f"test_per_group{a.tag}.csv", index=False)
    json.dump(m, open(C.OUTPUTS / f"test_metrics{a.tag}.json", "w"), indent=2)
    confusion_fig(y, pred, "official split" if "official" in a.tag else "own group split", f"fig_confusion{a.tag}.png")

    mine, off = C.OUTPUTS / "test_metrics.json", C.OUTPUTS / "test_metrics_official.json"
    if mine.exists() and off.exists():
        A, B = json.load(open(mine)), json.load(open(off))
        keys = ["acc", "macro_f1", "acc_group_balanced", "macro_f1_group_balanced", "frames_acc", "photos_acc", "n_test"]
        print("\n=== COMPARISON: own group-aware split vs official (leaky) split ===")
        print(pd.DataFrame({"own split": [A.get(k) for k in keys], "official split": [B.get(k) for k in keys]},
                           index=keys).round(4).to_string())


if __name__ == "__main__":
    main()
