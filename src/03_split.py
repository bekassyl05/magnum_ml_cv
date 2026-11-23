import itertools
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src import config as C

SPL = ["train", "val", "test"]
TARGET = {"train": 0.70, "val": 0.15, "test": 0.15}
PHOTO_SHARE = 0.20  # share of standalone photos put into EACH of val and test (min 3)


# --------------------------------------------------------------------- units
def load_units():
    cl = pd.read_csv(C.OUTPUTS / "clusters.csv")
    d = cl[cl.source == "all"].drop_duplicates("md5").reset_index(drop=True)
    info = d["filename"].map(C.parse_name).apply(pd.Series)
    d["video_id"] = info["video_id"]
    d["is_frame"] = info["is_frame"].astype(bool)
    stem = d["filename"].str.replace(r"\.\w+$", "", regex=True)
    d["unit"] = np.where(d["is_frame"], d["video_id"], "photo:" + stem)
    d["date"] = d["video_id"].str[:8]
    return d


def assign_class(g, rng):
    """Return {unit: split} for one class. Whole videos go to one split; test never shares a
    video with train. Video->split assignment minimises deviation from 70/15/15 (by images)."""
    vids = g[g.is_frame].groupby("unit").size()
    photos = sorted(g[~g.is_frame]["unit"].tolist())
    rng.shuffle(photos)
    N = len(g)
    tgt = {s: TARGET[s] * N for s in SPL}
    nv = max(3, int(round(PHOTO_SHARE * len(photos)))) if photos else 0
    ph_split = {u: ("val" if i < nv else "test" if i < 2 * nv else "train") for i, u in enumerate(photos)}
    ph_cnt = {s: sum(1 for v in ph_split.values() if v == s) for s in SPL}
    units, k = list(vids.index), len(vids)
    best, best_score = None, 1e18
    for combo in itertools.product(SPL, repeat=k):
        asg = dict(zip(units, combo))
        nvid = {s: combo.count(s) for s in SPL}
        if k >= 1 and nvid["train"] < 1:
            continue
        if k >= 2 and nvid["test"] < 1:
            continue
        cnt = {s: ph_cnt[s] + sum(vids[u] for u, sp in asg.items() if sp == s) for s in SPL}
        score = sum(abs(cnt[s] - tgt[s]) for s in SPL) / N
        if score < best_score:
            best, best_score = asg, score
    best = dict(best)
    best.update(ph_split)
    return best


def build_split(d):
    rng = np.random.RandomState(C.SEED)
    d["split"] = None
    for cls in C.CLASSES:
        g = d[d.cls == cls]
        d.loc[g.index, "split"] = g["unit"].map(assign_class(g, rng))
    assert d["split"].notna().all()
    d["weight"] = 1.0 / d.groupby("unit")["md5"].transform("size")
    # ---- verification
    for a, b in itertools.combinations(SPL, 2):
        sa, sb = d[d.split == a], d[d.split == b]
        assert not (set(sa.unit) & set(sb.unit)), f"group overlap {a}/{b}"
        assert not (set(sa.md5) & set(sb.md5)), f"md5 overlap {a}/{b}"
        assert not (set(sa.cluster_id) & set(sb.cluster_id)), f"near-dup cluster overlap {a}/{b}"
    print("VERIFIED: zero overlap of groups, MD5 and near-duplicate clusters between train/val/test.\n")
    return d


def report_balance(d):
    rows = []
    for (cls, sp), g in d.groupby(["cls", "split"]):
        rows.append(dict(cls=cls, split=sp, images=len(g), groups=g.unit.nunique(),
                         videos=g[g.is_frame].unit.nunique(), photos=int((~g.is_frame).sum()),
                         share_of_class=round(len(g) / (d.cls == cls).sum(), 3),
                         dates=",".join(sorted(g.date.unique()))))
    t = pd.DataFrame(rows)
    t["flag"] = np.where((t.split.isin(["val", "test"])) & ((t.groups < 3) | (t.images < 50)),
                         "SMALL(<3 groups or <50 imgs)", "")
    t.to_csv(C.OUTPUTS / "split_balance.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(t.to_string(index=False))
    tot = d.groupby("split").agg(images=("md5", "size"), groups=("unit", "nunique"))
    tot["share"] = (tot.images / tot.images.sum()).round(3)
    print("\nOverall:\n", tot.loc[SPL].to_string())
    print("\nPer-split class counts (images):\n", pd.crosstab(d.cls, d.split)[SPL].to_string())
    u = d.groupby("unit").agg(cls=("cls", "first"), date=("date", "first"), is_frame=("is_frame", "first"),
                              n=("md5", "size"), split=("split", "first")).reset_index()
    u.to_csv(C.SPLITS / "units.csv", index=False)
    d[["path", "cls", "unit", "md5", "split", "weight", "is_frame", "date", "cluster_id"]] \
        .rename(columns={"unit": "group"}).to_csv(C.SPLITS / "split.csv", index=False)


def build_folds(d):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=C.SEED)
    d["fold"] = -1
    for f, (_, te) in enumerate(sgkf.split(d, d.cls, d.unit)):
        d.loc[d.index[te], "fold"] = f
    d[["path", "cls", "unit", "md5", "fold"]].rename(columns={"unit": "group"}).to_csv(C.SPLITS / "folds.csv", index=False)
    print("\n5-fold StratifiedGroupKFold: images per fold x class\n", pd.crosstab(d.fold, d.cls).to_string())
    print("groups per fold x class\n", d.groupby(["fold", "cls"]).unit.nunique().unstack(fill_value=0).to_string())


# --------------------------------------------------------------- domain gap
def feats(path):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    a = np.asarray(im, dtype=np.float32)
    gray = a.mean(2)
    b = 10
    border = np.concatenate([a[:b].reshape(-1, 3), a[-b:].reshape(-1, 3),
                             a[:, :b].reshape(-1, 3), a[:, -b:].reshape(-1, 3)]).mean(0)
    tiny = np.asarray(im.resize((16, 16), Image.BILINEAR), dtype=np.float32).ravel() / 255
    return [gray.mean(), gray.std(), *border, ndimage.laplace(gray).var()], tiny


def group_cv_lr(X, y, groups):
    pred = np.empty(len(y), dtype=object)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=C.SEED).split(X, y, groups):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, C=0.05, class_weight="balanced"))
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def domain_checks(d):
    with ThreadPoolExecutor(8) as ex:
        res = list(tqdm(ex.map(feats, d["path"]), total=len(d), desc="domain features"))
    F = pd.DataFrame([r[0] for r in res], columns=["bright_mean", "bright_std", "border_r", "border_g", "border_b", "lap_var"])
    X = np.stack([r[1] for r in res])
    F = pd.concat([d[["path", "cls", "unit", "date", "is_frame", "split"]].reset_index(drop=True), F], axis=1)
    F.to_csv(C.OUTPUTS / "domain_features.csv", index=False)
    rows = []

    def add(check, key, metric, value, n=None, cls=""):
        rows.append(dict(check=check, cls=cls, key=key, metric=metric, value=round(float(value), 4), n=n))

    # 1) brightness / border colour per class and per capture date
    for (cls, date), g in F.groupby(["cls", "date"]):
        for m in ["bright_mean", "bright_std", "border_r", "border_g", "border_b"]:
            add("class_x_date", date, m, g[m].mean(), len(g), cls)
    # 2) video frames vs standalone photos
    for cls, g in list(F.groupby("cls")) + [("ALL", F)]:
        for m in ["bright_mean", "lap_var"]:
            fr, ph = g.loc[g.is_frame, m], g.loc[~g.is_frame, m]
            add("frame_vs_photo", "frame", m, fr.mean(), len(fr), cls)
            add("frame_vs_photo", "photo", m, ph.mean(), len(ph), cls)
    # 3) date alone predicts class?
    maj = F.groupby("date")["cls"].agg(lambda s: s.value_counts().idxmax())
    acc_date = (F["date"].map(maj) == F["cls"]).mean()
    add("date_shortcut", "majority-class-per-date", "accuracy", acc_date, len(F))
    add("date_shortcut", "always-biggest-class", "accuracy", F.cls.value_counts(normalize=True).iloc[0], len(F))
    # 4) background colour only (16x16 mean colour) with group-aware CV
    y_cls, y_date = F["cls"].values, F["date"].values
    for name, y in (("predict_class", y_cls), ("predict_date", y_date)):
        pred = group_cv_lr(X, y, F["unit"].values)
        add("tiny16_logreg_groupCV", name, "accuracy", accuracy_score(y, pred), len(y))
        add("tiny16_logreg_groupCV", name, "macro_f1", f1_score(y, pred, average="macro"), len(y))
        if name == "predict_class":
            for c in C.CLASSES:
                add("tiny16_logreg_groupCV", name, "recall", (pred[y == c] == c).mean(), int((y == c).sum()), c)
    out = pd.DataFrame(rows)
    out.to_csv(C.OUTPUTS / "domain_checks.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n== DOMAIN CHECKS ==")
    print(out[out.check == "date_shortcut"].to_string(index=False))
    print(out[out.check == "tiny16_logreg_groupCV"].to_string(index=False))
    print(out[(out.check == "frame_vs_photo") & (out.cls == "ALL")].to_string(index=False))
    print(F.groupby(["cls", "date"])[["bright_mean", "border_r", "border_g", "border_b"]].mean().round(1).to_string())


if __name__ == "__main__":
    d = build_split(load_units())
    report_balance(d)
    build_folds(d)
    domain_checks(d)
    print("\nDone. Files: splits/split.csv, splits/folds.csv, splits/units.csv, outputs/domain_checks.csv")
