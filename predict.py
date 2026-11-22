#!/usr/bin/env python
"""Vegetable classifier inference (6 classes), CPU-only and offline.

    python predict.py --input path/to/folder --output preds.csv

preds.csv columns: filename,label,confidence
  label       one of Capsicum, Carrot, Cucumber, Potato, Raddish, Squash, or "unknown"
              ("unknown" = unreadable file, or the image does not look like any of the 6 vegetables:
               score below the threshold in models/unknown_config.json; disable with --no-unknown)
  confidence  softmax probability of the label (2 decimals; 0.00 for unreadable files)

The preprocessing below must stay identical to src/common.py (Resize 224 -> CenterCrop 224 -> ImageNet
normalisation); src/07_predict_checks.py verifies that both give the same predictions.
Optional: --onnx models/best.onnx runs the exported model with onnxruntime instead of PyTorch.
"""
import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from PIL import Image, ImageOps
from torchvision import transforms as T

CLASSES = ["Capsicum", "Carrot", "Cucumber", "Potato", "Raddish", "Squash"]
IMG_SIZE = 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
HERE = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = [HERE / "models" / "final.pt", HERE / "models" / "best.pt"]  # first existing file is used
UNKNOWN_CFG = HERE / "models" / "unknown_config.json"

_TF = T.Compose([T.Resize(IMG_SIZE), T.CenterCrop(IMG_SIZE), T.ToTensor(), T.Normalize(MEAN, STD)])


def build_model(arch, num_classes=len(CLASSES)):
    """Architecture without downloading anything (weights=None); trained weights come from the checkpoint."""
    tv = torchvision.models
    if arch == "resnet18":
        m = tv.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "mobilenet_v3_small":
        m = tv.mobilenet_v3_small(weights=None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    elif arch == "efficientnet_b0":
        m = tv.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"unknown architecture {arch}")
    return m


def find_weights(path=None):
    if path:
        return Path(path)
    for p in DEFAULT_WEIGHTS:
        if p.exists():
            return p
    raise FileNotFoundError("no weights found: put models/best.pt (or models/final.pt) next to predict.py or pass --weights")


def load_model(path=None):
    path = find_weights(path)
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # checkpoint also stores a plain config dict; fall back for older torch versions
        ck = torch.load(path, map_location="cpu", weights_only=False)
    if list(ck.get("classes", CLASSES)) != CLASSES:
        raise ValueError(f"unexpected class list in checkpoint: {ck.get('classes')}")
    model = build_model(ck["arch"])
    model.load_state_dict(ck["state_dict"])
    return model.eval()


def load_unknown_cfg(path=None, disabled=False):
    """{"score": "msp" | "msp_T" | "energy", "threshold": float, "temperature": float} or None (feature off)."""
    if disabled:
        return None
    p = Path(path) if path else UNKNOWN_CFG
    if not p.exists():
        return None
    cfg = json.loads(p.read_text(encoding="utf-8"))
    if cfg["score"] not in ("msp", "msp_T", "energy"):
        raise ValueError(f"bad unknown score {cfg['score']}")
    return cfg


def decide(logits, cfg):
    """logits (N x 6) -> (class index, confidence, is_unknown). Confidence = (temperature-scaled) max softmax."""
    t = float(cfg.get("temperature", 1.0)) if cfg else 1.0
    conf, idx = torch.softmax(logits / t, dim=1).max(1)
    if cfg is None:
        unk = torch.zeros_like(conf, dtype=torch.bool)
    elif cfg["score"] == "energy":
        unk = torch.logsumexp(logits, dim=1) < float(cfg["threshold"])
    elif cfg["score"] == "msp":
        unk = torch.softmax(logits, dim=1).max(1).values < float(cfg["threshold"])
    else:
        unk = conf < float(cfg["threshold"])
    return idx, conf, unk


def load_tensor(path):
    """Read any supported image robustly -> normalised 3x224x224 tensor. Raises on unreadable files."""
    with Image.open(path) as im:
        try:
            im.draft("RGB", (2 * IMG_SIZE, 2 * IMG_SIZE))  # JPEG only: faster decoding of large photos
        except Exception:
            pass
        im = ImageOps.exif_transpose(im)  # phone photos carry their orientation in EXIF
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")
        return _TF(im)


def list_images(folder, recursive=False):
    it = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in EXTS)


def _safe_load(p):
    try:
        return load_tensor(p)
    except Exception as e:  # corrupt / truncated / unsupported file
        return e


@torch.inference_mode()
def predict_files(forward, files, batch_size=32, workers=4, cfg=None, root=None):
    """forward: callable(batch tensor) -> logits tensor. Returns [(filename, label, confidence, error_or_None)]."""
    out = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for i in range(0, len(files), batch_size):
            chunk = files[i:i + batch_size]
            loaded = list(ex.map(_safe_load, chunk))
            good = [(f, t) for f, t in zip(chunk, loaded) if not isinstance(t, Exception)]
            res = {}
            if good:
                idx, conf, unk = decide(forward(torch.stack([t for _, t in good])).float(), cfg)
                res = {f: ("unknown" if bool(u) else CLASSES[int(k)], float(c)) for (f, _), k, c, u in zip(good, idx, conf, unk)}
            for f, t in zip(chunk, loaded):
                name = str(f.relative_to(root)) if root else f.name
                if isinstance(t, Exception):
                    out.append((name, "unknown", 0.0, repr(t)))
                else:
                    out.append((name, *res[f], None))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="folder with images")
    ap.add_argument("--output", required=True, help="CSV file to write")
    ap.add_argument("--weights", default=None, help="PyTorch checkpoint (default: models/final.pt or models/best.pt)")
    ap.add_argument("--onnx", default=None, help="use this ONNX model with onnxruntime instead of PyTorch")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = all cores)")
    ap.add_argument("--workers", type=int, default=4, help="image-decoding threads")
    ap.add_argument("--recursive", action="store_true", help="also look into sub-folders")
    ap.add_argument("--no-unknown", action="store_true", help="never output 'unknown' for readable images")
    ap.add_argument("--unknown-config", default=None, help="JSON with score/threshold (default: models/unknown_config.json)")
    a = ap.parse_args()

    folder = Path(a.input)
    if not folder.is_dir():
        sys.exit(f"error: input folder not found: {folder}")
    threads = a.threads or os.cpu_count() or 1
    torch.set_num_threads(threads)  # CPU only, no network access anywhere
    if a.onnx:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")  # idle ORT threads must not compete with image decoding
        sess = ort.InferenceSession(a.onnx, so, providers=["CPUExecutionProvider"])
        forward = lambda x: torch.from_numpy(sess.run(None, {"input": x.numpy()})[0])  # noqa: E731
    else:
        forward = load_model(a.weights)
    cfg = load_unknown_cfg(a.unknown_config, a.no_unknown)
    files = list_images(folder, a.recursive)
    if not files:
        print(f"warning: no images ({', '.join(sorted(EXTS))}) in {folder}", file=sys.stderr)

    t0 = time.perf_counter()
    rows = predict_files(forward, files, a.batch_size, a.workers, cfg, root=folder)
    dt = time.perf_counter() - t0

    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    with open(a.output, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label", "confidence"])
        for name, label, conf, _ in rows:
            w.writerow([name, label, f"{conf:.2f}"])
    bad = [(n, e) for n, _, _, e in rows if e]
    for n, e in bad:
        print(f"warning: could not read {n}: {e}", file=sys.stderr)
    n_unk = sum(1 for _, lab, _, e in rows if lab == "unknown" and not e)
    print(f"{len(rows)} images ({len(bad)} unreadable, {n_unk} unknown) in {dt:.1f}s "
          f"({1000 * dt / max(len(rows), 1):.0f} ms/image, {threads} threads, {'onnxruntime' if a.onnx else 'pytorch'}) -> {a.output}")


if __name__ == "__main__":
    main()
