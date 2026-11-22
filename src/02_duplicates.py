"""Step 2: duplicates & leakage. Usage: python src/02_duplicates.py [--tp 6 --td 6 --tc 0.97] [--rescan]
Needs outputs/inventory.csv (from 01_audit.py). First run downloads ResNet18 weights (~45 MB).
Outputs (outputs/): clusters.csv, leakage_report.txt, video_split_matrix.csv,
fig_threshold_cross_video.png, fig_dup_pairs.png, md5.csv, hashes.npz, emb.npy
"""
import argparse
import hashlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import imagehash
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image, ImageOps
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm

from src import config as C

OUT = C.OUTPUTS
POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
SPL = ["train", "val", "test"]
R = []


def say(s=""):
    print(s)
    R.append(s)


# ------------------------------------------------------------------ features
def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def get_md5(df, rescan):
    f = OUT / "md5.csv"
    cache = dict(pd.read_csv(f).values) if (f.exists() and not rescan) else {}
    todo = [p for p in df["path"] if p not in cache]
    if todo:
        with ThreadPoolExecutor(8) as ex:
            res = list(tqdm(ex.map(md5_file, todo), total=len(todo), desc="md5"))
        cache.update(zip(todo, res))
        pd.DataFrame(list(cache.items()), columns=["path", "md5"]).to_csv(f, index=False)
    return df["path"].map(cache)


def load_rgb(p):
    return ImageOps.exif_transpose(Image.open(p)).convert("RGB")


def hash_pair(p):
    im = load_rgb(p)
    return int(str(imagehash.phash(im)), 16), int(str(imagehash.dhash(im)), 16)


def embed(paths, bs=64):
    w = torchvision.models.ResNet18_Weights.DEFAULT
    m = torchvision.models.resnet18(weights=w)
    m.fc = torch.nn.Identity()
    m.eval()
    tf = w.transforms()
    torch.set_num_threads(os.cpu_count() or 4)
    outs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(paths), bs), desc="embeddings"):
            x = torch.stack([tf(load_rgb(p)) for p in paths[i:i + bs]])
            outs.append(F.normalize(m(x), dim=1).numpy().astype(np.float32))
    return np.concatenate(outs)


def get_features(umd5, rep, rescan):
    hf, ef = OUT / "hashes.npz", OUT / "emb.npy"
    if hf.exists() and ef.exists() and not rescan:
        z = np.load(hf, allow_pickle=False)
        E = np.load(ef)
        if len(z["md5"]) == len(umd5) and (z["md5"] == umd5).all() and len(E) == len(umd5):
            return z["P"], z["D"], E
    with ThreadPoolExecutor(8) as ex:
        hh = list(tqdm(ex.map(hash_pair, rep), total=len(rep), desc="phash/dhash"))
    P = np.array([a for a, _ in hh], dtype=np.uint64)
    D = np.array([b for _, b in hh], dtype=np.uint64)
    E = embed(list(rep))
    np.savez(hf, md5=umd5, P=P, D=D)
    np.save(ef, E)
    return P, D, E


def ham(a, b):
    x = np.bitwise_xor(a[:, None], b[None, :])
    return POP[x.view(np.uint8).reshape(x.shape[0], x.shape[1], 8)].sum(-1, dtype=np.uint8)


# ------------------------------------------------------------------ pair scan
def pair_scan(Pf, Df, Ef, grp, tp, td, tc, ch=128):
    n = len(Pf)
    ei, ej = [], []
    nn_dp = np.full(n, 64, np.uint8)
    nn_dp_j = np.zeros(n, np.int32)
    ar = np.arange(n)
    for s in tqdm(range(0, n, ch), desc="pair scan"):
        e = min(n, s + ch)
        dp, dd = ham(Pf[s:e], Pf), ham(Df[s:e], Df)
        cos = Ef[s:e] @ Ef.T
        cross = grp[s:e, None] != grp[None, :]
        dpc = np.where(cross, dp, 64)
        nn_dp[s:e] = dpc.min(1)
        nn_dp_j[s:e] = dpc.argmin(1)
        sel = (ar[None, :] > ar[s:e, None]) & (((dp <= tp) & (dd <= td)) | (cos >= tc))
        ii, jj = np.nonzero(sel)
        ei.append((ii + s).astype(np.int32))
        ej.append(jj.astype(np.int32))
    return np.concatenate(ei), np.concatenate(ej), nn_dp, nn_dp_j


# ------------------------------------------------------------------ figures
def pair_fig(pairs, paths, titles, fname, ncols_pairs=2):
    rows = int(np.ceil(len(pairs) / ncols_pairs))
    fig, axs = plt.subplots(rows, ncols_pairs * 2, figsize=(3.2 * ncols_pairs * 2, 3.4 * rows))
    axs = np.atleast_2d(axs)
    for ax in axs.ravel():
        ax.axis("off")
    for k, ((a, b), t) in enumerate(zip(pairs, titles)):
        r, c = divmod(k, ncols_pairs)
        for q, idx in enumerate((a, b)):
            ax = axs[r, c * 2 + q]
            ax.imshow(load_rgb(paths[idx]))
            ax.set_title(Path(paths[idx]).parent.parent.name[:5] + "/" + Path(paths[idx]).name[:24] if q == 0
                         else Path(paths[idx]).name[:28], fontsize=6)
        axs[r, c * 2].text(0, 215, t, fontsize=7, color="red")
    plt.tight_layout()
    plt.savefig(OUT / fname, dpi=120)
    plt.close()


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=6)
    ap.add_argument("--td", type=int, default=6)
    ap.add_argument("--tc", type=float, default=0.97)
    ap.add_argument("--rescan", action="store_true")
    a = ap.parse_args()

    df = pd.read_csv(OUT / "inventory.csv")
    df = df[~df["corrupt"].astype(bool)].reset_index(drop=True)
    df["md5"] = get_md5(df, a.rescan)
    umd5 = np.array(sorted(df["md5"].unique()))
    rep = df.drop_duplicates("md5").set_index("md5")["path"].reindex(umd5).values
    P, D, E = get_features(umd5, rep, a.rescan)
    inv = np.searchsorted(umd5, df["md5"].values)
    Pf, Df, Ef = P[inv], D[inv], E[inv]
    paths = df["path"].values
    cls = df["cls"].values
    src = df["source"].values
    grp = pd.factorize(np.where(df["video_id"].notna(), df["video_id"].astype(str),
                                "photo:" + df["filename"].str.replace(r"\.\w+$", "", regex=True)))[0]
    n = len(df)
    say(f"Files: {n}; unique by MD5: {len(umd5)}; thresholds: phash<={a.tp} & dhash<={a.td} OR cosine>={a.tc}")

    # (a) exact duplicates
    say("\n== (a) EXACT duplicates (MD5) ==")
    al = df[df.source == "all"]
    say(f"Dataset/: {len(al)} files, {al.md5.nunique()} unique -> {len(al) - al.md5.nunique()} exact duplicate files")
    mc = df.groupby("md5")["cls"].nunique()
    say(f"MD5 present in >1 class (label conflict): {int((mc > 1).sum())}")
    sets = {s: set(df.loc[df.source == s, "md5"]) for s in ["all"] + SPL}
    for x, y in [("test", "train"), ("val", "train"), ("test", "val")]:
        k = df[(df.source == x) & df.md5.isin(sets[y])]
        say(f"official {x} files with identical file in {y}: {len(k)} / {int((df.source == x).sum())}")
    off = df[df.source.isin(SPL)]
    say(f"official files whose MD5 is also in Dataset/: {int(off.md5.isin(sets['all']).sum())} / {len(off)}")
    say(f"Dataset/ files whose MD5 is in official: {int(al.md5.isin(set(off.md5)).sum())} / {len(al)}")
    nm = off.merge(al[["cls", "filename", "md5"]], on=["cls", "filename"], suffixes=("", "_all"))
    say(f"official files with the same class+filename as in Dataset/: {len(nm)} (identical bytes: {int((nm.md5 == nm.md5_all).sum())})")
    t = off.assign(in_all=off.md5.isin(sets["all"])).groupby(["source", "cls"])["in_all"].agg(["size", "sum"])
    say("official per split/class: n files, n found in Dataset/ by MD5\n" + t.to_string())

    # (b) near duplicates + clusters
    say("\n== (b) NEAR duplicates + clusters ==")
    ei, ej, nn_dp, nn_dp_j = pair_scan(Pf, Df, Ef, grp, a.tp, a.td, a.tc)
    g = coo_matrix((np.ones(len(ei), np.int8), (ei, ej)), shape=(n, n))
    ncomp, lab = connected_components(g, directed=False)
    df["cluster_id"] = lab
    off = df[df.source.isin(SPL)]
    cs = pd.Series(lab).value_counts()
    say(f"edges: {len(ei)}; clusters: {ncomp}; singletons: {int((cs == 1).sum())}; biggest sizes: {cs.head(8).tolist()}")
    gcl = df.groupby("cluster_id").agg(n_cls=("cls", "nunique"), n_grp=("filename", "size"))
    say(f"clusters spanning >1 class (label conflicts / too-loose threshold): {int((gcl.n_cls > 1).sum())}")
    say(f"cross-class edges: {int((cls[ei] != cls[ej]).sum())} of {len(ei)}")
    has_nb = np.zeros(n, bool)
    has_nb[ei] = True
    has_nb[ej] = True
    say(f"images with at least one near-duplicate neighbour: {has_nb.mean():.1%}")
    cross_vid = pd.Series(grp).groupby(lab).nunique()
    say(f"clusters that merge >1 video/photo group: {int((cross_vid > 1).sum())}")
    df["group"] = grp
    df[["source", "cls", "filename", "path", "md5", "video_id", "group", "cluster_id"]].to_csv(OUT / "clusters.csv", index=False)

    # (c) leakage in the official split
    say("\n== (c) LEAKAGE in official split ==")
    vv = off[off.video_id.notna()].groupby(["cls", "video_id", "source"]).size().unstack(fill_value=0)
    for s in SPL:
        if s not in vv.columns:
            vv[s] = 0
    vv["n_splits"] = (vv[SPL] > 0).sum(1)
    vv.to_csv(OUT / "video_split_matrix.csv")
    say(f"videos in official: {len(vv)}; present in >1 split: {int((vv.n_splits > 1).sum())}; in all 3 splits: {int((vv.n_splits == 3).sum())}")
    tr_vid = set(off.loc[(off.source == "train") & off.video_id.notna(), "video_id"])
    for s in ("val", "test"):
        f = off[(off.source == s) & off.video_id.notna()]
        say(f"{s} frames whose video_id also occurs in train: {f.video_id.isin(tr_vid).sum()} / {len(f)} ({f.video_id.isin(tr_vid).mean():.1%})")
    ph = off[~off["is_frame"].astype(bool)]
    say(f"standalone photos (no video id) in official: {len(ph)}")
    mult = off.groupby(["cls", "filename"])["source"].nunique()
    say(f"filenames that appear in >1 official split: {int((mult > 1).sum())}")
    oc = off.groupby("cluster_id")["source"].nunique()
    leak_cl = oc[oc > 1].index
    for s in ("val", "test"):
        f = off[off.source == s]
        say(f"{s} files in a near-dup cluster that also has train files: {f.cluster_id.isin(set(off[off.source=='train'].cluster_id)).sum()} / {len(f)}")
    say("")
    itr = np.where(src == "train")[0]
    for s in ("val", "test"):
        ia = np.where(src == s)[0]
        sim = Ef[ia] @ Ef[itr].T
        same = grp[ia][:, None] == grp[itr][None, :]
        c_same = np.where(same, sim, -1).max(1)
        c_diff = np.where(~same, sim, -1).max(1)
        best = np.maximum(c_same, c_diff)
        hd = ham(Pf[ia], Pf[itr]) + ham(Df[ia], Df[itr])
        say(f"[{s} -> train] nearest train image is from the SAME video/photo group: {(c_same >= c_diff).mean():.1%}")
        say(f"   median cosine to nearest train image: same-group {np.median(c_same[c_same>-1]):.3f} | other-groups {np.median(c_diff):.3f}")
        for th in (0.90, 0.95, 0.98, 0.99):
            say(f"   {s} images with a train image at cosine>={th}: {(best >= th).mean():.1%}")
        say(f"   {s} images with a train image at phash+dhash distance<={a.tp + a.td}: {(hd.min(1) <= a.tp + a.td).mean():.1%}")
        if s == "test":
            sel = np.where(best >= 0.98)[0]
            rng = np.random.RandomState(C.SEED)
            pick = rng.choice(sel, min(10, len(sel)), replace=False) if len(sel) else []
            pairs, titles = [], []
            for p in pick:
                j = itr[int(np.argmax(sim[p]))]
                pairs.append((ia[p], j))
                titles.append(f"cos={sim[p].max():.3f} same_group={grp[ia[p]] == grp[j]}")
            if pairs:
                pair_fig(pairs, paths, titles, "fig_dup_pairs.png")

    # threshold inspection: cross-video nearest pairs by hash distance bucket
    rng = np.random.RandomState(C.SEED)
    pairs, titles = [], []
    cand = np.where(src == "all")[0]
    for lo, hi in [(0, 2), (3, 5), (6, 8), (9, 12)]:
        ok = cand[(nn_dp[cand] >= lo) & (nn_dp[cand] <= hi)]
        say(f"Dataset/ images whose nearest OTHER-group image has phash dist {lo}-{hi}: {len(ok)}")
        for i in (rng.choice(ok, min(3, len(ok)), replace=False) if len(ok) else []):
            j = int(nn_dp_j[i])
            pairs.append((i, j))
            titles.append(f"phash={nn_dp[i]} cos={float(Ef[i] @ Ef[j]):.3f} cls={cls[i]}/{cls[j]}")
    if pairs:
        pair_fig(pairs, paths, titles, "fig_threshold_cross_video.png", ncols_pairs=3)

    (OUT / "leakage_report.txt").write_text("\n".join(R), encoding="utf-8")
    say("\nSaved: clusters.csv, leakage_report.txt, video_split_matrix.csv, fig_*.png")


if __name__ == "__main__":
    main()
