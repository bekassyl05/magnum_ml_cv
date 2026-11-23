import argparse
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

from src import config as C

INV = C.OUTPUTS / "inventory.csv"


def list_files():
    recs = []
    for cls in C.CLASSES:
        d = C.DATA_ALL / cls
        for p in sorted(d.iterdir()) if d.exists() else []:
            if p.is_file():
                recs.append(("all", cls, p))
    for sp in C.OFFICIAL_SPLITS:
        for cls in C.CLASSES:
            d = C.DATA_OFFICIAL / sp / cls
            for p in sorted(d.iterdir()) if d.exists() else []:
                if p.is_file():
                    recs.append((sp, cls, p))
    # folders that are not one of the 6 classes?
    for root in (C.DATA_ALL, *(C.DATA_OFFICIAL / s for s in C.OFFICIAL_SPLITS)):
        if root.exists():
            extra = [x.name for x in root.iterdir() if x.is_dir() and x.name not in C.CLASSES]
            if extra:
                print(f"[WARN] unexpected folders in {root}: {extra}")
    return recs


def inspect(rec):
    source, cls, p = rec
    d = dict(source=source, cls=cls, path=str(p), filename=p.name,
             ext=p.suffix.lower(), size_bytes=p.stat().st_size)
    d.update(C.parse_name(p.name))
    d["is_image_ext"] = d["ext"] in C.IMG_EXTS
    try:
        with Image.open(p) as im:
            im.load()
            ex = im.getexif()
            d.update(width=im.width, height=im.height, mode=im.mode, fmt=im.format,
                     exif_make=ex.get(0x010F), exif_model=ex.get(0x0110),
                     exif_orient=ex.get(0x0112), corrupt=False)
    except Exception as e:  # noqa
        d.update(corrupt=True, error=repr(e)[:120])
    return d


def build_inventory(rescan):
    if INV.exists() and not rescan:
        print(f"Loading cached {INV}")
        return pd.read_csv(INV)
    recs = list_files()
    print(f"Scanning {len(recs)} files ...")
    with ThreadPoolExecutor(max_workers=8) as ex:
        rows = list(tqdm(ex.map(inspect, recs), total=len(recs)))
    df = pd.DataFrame(rows)
    df.to_csv(INV, index=False)
    return df


def summarize(df):
    df = df.copy()
    df["has_id"] = df["video_id"].notna()
    df["is_frame"] = df["is_frame"].astype(bool)
    df["is_photo"] = df["has_id"] & ~df["is_frame"]
    df["aspect"] = df["width"] / df["height"]
    df["size_kb"] = df["size_bytes"] / 1024
    rows = []
    for (src, cls), g in df.groupby(["source", "cls"]):
        ok = g[~g["corrupt"].astype(bool)]
        fr = g[g["is_frame"]]
        per_vid = fr.groupby("video_id").size()
        rows.append(dict(
            source=src, cls=cls, n=len(g), corrupt=int(g["corrupt"].sum()),
            non_image_ext=int((~g["is_image_ext"]).sum()),
            n_frames=int(g["is_frame"].sum()), n_standalone_photos=int(g["is_photo"].sum()),
            n_unparsed_names=int((~g["has_id"]).sum()),
            n_videos=int(per_vid.size),
            frames_per_video_mean=round(per_vid.mean(), 1) if len(per_vid) else 0,
            frames_per_video_median=float(per_vid.median()) if len(per_vid) else 0,
            frames_per_video_max=int(per_vid.max()) if len(per_vid) else 0,
            n_independent_groups=int(per_vid.size + g["is_photo"].sum()),
            w_median=ok["width"].median(), h_median=ok["height"].median(),
            w_min=ok["width"].min(), w_max=ok["width"].max(),
            h_min=ok["height"].min(), h_max=ok["height"].max(),
            n_unique_sizes=int(ok.groupby(["width", "height"]).ngroups),
            aspect_median=round(ok["aspect"].median(), 3),
            non_rgb=int((ok["mode"] != "RGB").sum()),
            ext_jpg_share=round(ok["ext"].isin([".jpg", ".jpeg"]).mean(), 3),
            kb_mean=round(ok["size_kb"].mean(), 1), kb_median=round(ok["size_kb"].median(), 1),
            kb_min=round(ok["size_kb"].min(), 1), kb_max=round(ok["size_kb"].max(), 1),
            exif_orient_not1=int(ok["exif_orient"].notna().sum() - (ok["exif_orient"] == 1).sum()),
            exif_models=";".join(map(str, ok["exif_model"].dropna().unique()[:5])),
        ))
    s = pd.DataFrame(rows)
    s.to_csv(C.OUTPUTS / "audit_summary.csv", index=False)
    return df, s


def extra_tables(df):
    allx = df[df["source"] == "all"]
    vids = (df[df["is_frame"]].groupby(["source", "cls", "video_id"]).size()
            .rename("n_frames").reset_index())
    vids.to_csv(C.OUTPUTS / "videos.csv", index=False)
    dates = (allx[allx["has_id"]].assign(date=lambda x: x["video_id"].str[:8])
             .groupby(["date", "cls"]).size().unstack(fill_value=0))
    dates.to_csv(C.OUTPUTS / "dates_by_class.csv")
    res = (allx.dropna(subset=["width"]).groupby(["cls", "width", "height"]).size()
           .rename("n").reset_index().sort_values(["cls", "n"], ascending=[True, False])
           .groupby("cls").head(10))
    res.to_csv(C.OUTPUTS / "resolutions_top.csv", index=False)
    # same video_id appearing in more than one class? same filename in many classes?
    multi = allx[allx["has_id"]].groupby("video_id")["cls"].nunique()
    print("\nvideo_ids shared between classes (all photos):", int((multi > 1).sum()))
    print(multi[multi > 1].head(10).to_string() if (multi > 1).any() else "  none")
    same_name = allx.groupby("filename")["cls"].nunique()
    print("filenames present in >1 class:", int((same_name > 1).sum()))
    print("\nDates (all photos):\n", dates.to_string())
    print("\nUnparsed filenames (first 15):")
    print(allx[~allx["has_id"]]["filename"].head(15).to_string(index=False))
    print("\nEXIF camera models (all):", allx["exif_model"].value_counts(dropna=False).head(8).to_dict())
    print("Image modes:", allx["mode"].value_counts().to_dict(), "| formats:", allx["fmt"].value_counts().to_dict())
    print("Corrupt files total:", int(df["corrupt"].sum()))


def figures(df):
    allx = df[(df["source"] == "all") & (~df["corrupt"].astype(bool))]
    # 1) class counts: all vs official splits
    cnt = df.groupby(["cls", "source"]).size().unstack(fill_value=0)
    cnt = cnt[[c for c in ["all", "train", "val", "test"] if c in cnt.columns]]
    ax = cnt.plot.bar(figsize=(9, 4.5), rot=0)
    ax.set_title("Images per class: all photos vs official split")
    ax.set_ylabel("images")
    plt.tight_layout(); plt.savefig(C.OUTPUTS / "fig_class_counts.png", dpi=130); plt.close()

    # 2) independent groups (videos + standalone) per class
    g = (allx.assign(group=np.where(allx["has_id"], allx["video_id"], allx["filename"]))
         .groupby("cls")["group"].nunique())
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    g.plot.bar(ax=ax[0], rot=0); ax[0].set_title("Independent groups per class\n(videos + standalone photos)")
    fr = df[(df["source"] == "all") & df["is_frame"]].groupby(["cls", "video_id"]).size()
    for cls in C.CLASSES:
        if cls in fr.index.get_level_values(0):
            ax[1].hist(fr[cls].values, bins=30, histtype="step", label=cls)
    ax[1].set_yscale("log"); ax[1].set_xlabel("frames per video"); ax[1].legend(fontsize=7)
    ax[1].set_title("Frames per video")
    plt.tight_layout(); plt.savefig(C.OUTPUTS / "fig_groups_frames.png", dpi=130); plt.close()

    # 3) size / aspect / filesize histograms per class
    fig, axs = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (col, ttl) in zip(axs.ravel(), [("width", "width, px"), ("height", "height, px"),
                                            ("aspect", "aspect w/h"), ("size_kb", "file size, KB")]):
        for cls in C.CLASSES:
            v = allx.loc[allx["cls"] == cls, col].dropna()
            if len(v):
                ax.hist(v, bins=40, histtype="step", label=cls)
        ax.set_title(ttl); ax.set_yscale("log")
    axs[0, 0].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(C.OUTPUTS / "fig_sizes.png", dpi=130); plt.close()

    # 4) sample grid per class: 16 images from different groups
    rnd = random.Random(C.SEED)
    for cls in C.CLASSES:
        g = allx[allx["cls"] == cls].copy()
        g["group"] = np.where(g["has_id"], g["video_id"], g["filename"])
        groups = list(g["group"].unique()); rnd.shuffle(groups)
        picks = [g[g["group"] == k].sample(1, random_state=C.SEED).iloc[0] for k in groups[:16]]
        fig, axs = plt.subplots(4, 4, figsize=(10, 10))
        for ax in axs.ravel():
            ax.axis("off")
        for ax, r in zip(axs.ravel(), picks):
            try:
                im = ImageOps.exif_transpose(Image.open(r["path"]).convert("RGB"))
                im.thumbnail((320, 320)); ax.imshow(im)
                ax.set_title(r["filename"][:22], fontsize=6)
            except Exception:
                pass
        fig.suptitle(f"{cls}: 16 random groups (one image each)")
        plt.tight_layout(); plt.savefig(C.OUTPUTS / f"samples_{cls}.png", dpi=110); plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescan", action="store_true")
    args = ap.parse_args()
    df = build_inventory(args.rescan)
    df["corrupt"] = df["corrupt"].astype(bool)
    df, summary = summarize(df)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(summary[["source", "cls", "n", "corrupt", "n_frames", "n_standalone_photos",
                   "n_videos", "frames_per_video_median", "frames_per_video_max",
                   "n_independent_groups", "w_median", "h_median", "n_unique_sizes"]].to_string(index=False))
    extra_tables(df)
    figures(df)
    print("\nDone. See outputs/audit_summary.csv and outputs/fig_*.png, samples_*.png")
