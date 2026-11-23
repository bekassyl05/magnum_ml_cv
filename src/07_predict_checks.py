import csv
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

import predict  # root predict.py
from src import config as C
from src.common import load_model as load_eval_model
from src.common import predict_df

OUT = C.OUTPUTS
TEST_DIR = OUT / "predict_test_input"


def stats(a):
    a = np.asarray(a) * 1000
    return float(a.mean()), float(np.median(a)), float(np.percentile(a, 95))


def check_cli():
    sp = pd.read_csv(C.SPLITS / "split.csv")
    te = sp[sp.split == "test"].sample(20, random_state=C.SEED).reset_index(drop=True)
    shutil.rmtree(TEST_DIR, ignore_errors=True)
    TEST_DIR.mkdir(parents=True)
    for p in te.path:
        shutil.copy(p, TEST_DIR / Path(p).name)
    (TEST_DIR / "corrupt.jpg").write_bytes(os.urandom(2000))     # random bytes with an image extension
    (TEST_DIR / "empty.png").write_bytes(b"")                    # zero-byte file
    (TEST_DIR / "notes.txt").write_text("not an image")          # must be ignored
    out_csv = OUT / "preds_test20.csv"
    r = subprocess.run([sys.executable, str(ROOT / "predict.py"), "--input", str(TEST_DIR), "--output", str(out_csv), "--no-unknown"],
                       capture_output=True, text=True, cwd=ROOT)
    print("predict.py exit code:", r.returncode)
    print("stdout:", r.stdout.strip())
    print("stderr (warnings expected for the 2 broken files):", r.stderr.strip()[:400])
    ok = r.returncode == 0
    with open(out_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    ok &= rows[0] == ["filename", "label", "confidence"]
    body = rows[1:]
    names = {x[0] for x in body}
    valid = set(C.CLASSES) | {"unknown"}
    ok &= len(body) == 22 and "notes.txt" not in names
    ok &= all(x[1] in valid for x in body)
    ok &= all(len(x[2].split(".")[-1]) == 2 and 0 <= float(x[2]) <= 1 for x in body)
    broken = {x[0]: x for x in body if x[0] in ("corrupt.jpg", "empty.png")}
    ok &= all(v[1] == "unknown" and v[2] == "0.00" for v in broken.values()) and len(broken) == 2
    print("first rows:", body[:3])
    print("FORMAT CHECK:", "PASS" if ok else "FAIL")

    # parity with the evaluation pipeline (same weights)
    w = predict.find_weights()
    model, _ = load_eval_model(w, "cpu")
    pr = predict_df(model, te, "cpu")
    ref = pd.DataFrame({"filename": te.path.map(lambda p: Path(p).name), "ref_label": np.array(C.CLASSES)[pr.argmax(1)],
                        "ref_conf": pr.max(1)})
    got = pd.DataFrame(body, columns=["filename", "label", "confidence"]).merge(ref, on="filename")
    same_label = (got.label == got.ref_label).mean()
    max_diff = (got.confidence.astype(float) - got.ref_conf).abs().max()
    print(f"PARITY with src/common.py on {len(got)} images: identical labels {same_label:.0%}, max |conf diff| {max_diff:.4f}",
          "PASS" if same_label == 1 and max_diff <= 0.0051 else "FAIL")
    return te


def latency(te):
    model = predict.load_model()
    paths = [Path(p) for p in te.path]
    sp = pd.read_csv(C.SPLITS / "split.csv")
    paths = [Path(p) for p in sp[sp.split == "test"].sample(100, random_state=C.SEED).path]
    ncpu = os.cpu_count() or 1
    meta = dict(cpu=platform.processor() or platform.machine(), logical_cores=ncpu, os=platform.platform(),
                python=platform.python_version(), torch=torch.__version__, weights=str(predict.find_weights().name))
    rows = []
    for th in sorted({1, 2, 4, ncpu} & set(range(1, ncpu + 1))):
        torch.set_num_threads(th)
        with torch.inference_mode():
            for p in paths[:10]:  # warm-up
                model(predict.load_tensor(p).unsqueeze(0))
            pre, inf, e2e = [], [], []
            for p in paths[:100]:
                t0 = time.perf_counter()
                x = predict.load_tensor(p).unsqueeze(0)
                t1 = time.perf_counter()
                model(x)
                t2 = time.perf_counter()
                pre.append(t1 - t0); inf.append(t2 - t1); e2e.append(t2 - t0)
            xb = torch.stack([predict.load_tensor(p) for p in paths[:32]])
            model(xb)  # warm-up for batch shape
            bt = []
            for _ in range(5):
                t0 = time.perf_counter(); model(xb); bt.append((time.perf_counter() - t0) / 32)
        for stage, v in (("preprocess (decode+resize)", pre), ("model forward, batch 1", inf),
                         ("end-to-end per image, batch 1", e2e), ("model forward per image, batch 32", bt)):
            m, md, p95 = stats(v)
            rows.append(dict(threads=th, stage=stage, mean_ms=round(m, 2), median_ms=round(md, 2), p95_ms=round(p95, 2),
                             n=len(v), **meta))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "latency.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nLATENCY (CPU)", {k: meta[k] for k in ("cpu", "logical_cores", "torch")})
    print(df[["threads", "stage", "mean_ms", "median_ms", "p95_ms", "n"]].to_string(index=False))


if __name__ == "__main__":
    te = check_cli()
    latency(te)
