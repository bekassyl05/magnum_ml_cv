import argparse
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch

import predict
from src import config as C


def st(v):
    v = np.asarray(v) * 1000
    return round(float(v.mean()), 2), round(float(np.median(v)), 2), round(float(np.percentile(v, 95)), 2)


def bench(fwd, paths, th, runtime):
    for p in paths[:10]:  # warm-up
        fwd(predict.load_tensor(p).unsqueeze(0))
    inf, e2e = [], []
    for p in paths[:100]:
        t0 = time.perf_counter()
        x = predict.load_tensor(p).unsqueeze(0)
        t1 = time.perf_counter()
        fwd(x)
        t2 = time.perf_counter()
        inf.append(t2 - t1)
        e2e.append(t2 - t0)
    xb = torch.stack([predict.load_tensor(p) for p in paths[:32]])
    fwd(xb)
    bt = []
    for _ in range(5):
        t0 = time.perf_counter()
        fwd(xb)
        bt.append((time.perf_counter() - t0) / 32)
    rows = []
    for stage, v in (("model forward, batch 1", inf), ("end-to-end per image, batch 1", e2e),
                     ("model forward per image, batch 32", bt)):
        m, md, p95 = st(v)
        rows.append(dict(runtime=runtime, threads=th, stage=stage, mean_ms=m, median_ms=md, p95_ms=p95, n=len(v)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "models" / "best.pt"))
    ap.add_argument("--out", default=str(ROOT / "models" / "best.onnx"))
    a = ap.parse_args()
    model = predict.load_model(a.weights)
    torch.onnx.export(model, torch.randn(1, 3, 224, 224), a.out, input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17,
                      do_constant_folding=True)
    onnx.checker.check_model(onnx.load(a.out))
    print(f"exported {a.out}: {os.path.getsize(a.out) / 1e6:.1f} MB (checkpoint: {os.path.getsize(a.weights) / 1e6:.1f} MB)")

    sp = pd.read_csv(C.SPLITS / "split.csv")
    paths = [Path(p) for p in sp[sp.split == "test"].sample(100, random_state=C.SEED).path]
    X = torch.stack([predict.load_tensor(p) for p in paths[:64]])
    with torch.inference_mode():
        lt = model(X).numpy()
    lo = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"]).run(None, {"input": X.numpy()})[0]

    def sm(z):
        e = np.exp(z - z.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    print(f"PARITY on {len(X)} images: max|logit diff| {np.abs(lt - lo).max():.2e}, mean {np.abs(lt - lo).mean():.2e}, "
          f"max|softmax diff| {np.abs(sm(lt) - sm(lo)).max():.2e}, identical argmax {(lt.argmax(1) == lo.argmax(1)).mean():.0%}")

    ncpu, rows = os.cpu_count() or 1, []
    for th in sorted({1, 2, 4, ncpu} & set(range(1, ncpu + 1))):
        torch.set_num_threads(th)
        with torch.inference_mode():
            rows += bench(lambda x: model(x), paths, th, "pytorch")
        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = th, 1
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")  # idle ORT threads must not compete with image decoding
        sess = ort.InferenceSession(a.out, so, providers=["CPUExecutionProvider"])
        rows += bench(lambda x: sess.run(None, {"input": x.numpy()}), paths, th, "onnxruntime")
    df = pd.DataFrame(rows)
    df.to_csv(C.OUTPUTS / "latency_onnx.csv", index=False)
    pv = df.pivot_table(index=["threads", "stage"], columns="runtime", values="mean_ms")
    pv["speedup_x"] = (pv["pytorch"] / pv["onnxruntime"]).round(2)
    print("\nCPU:", platform.processor(), "| cores", ncpu, "| torch", torch.__version__, "| onnxruntime", ort.__version__)
    print(pv.round(2).to_string())


if __name__ == "__main__":
    main()
