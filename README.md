# Vegetable recognition from photos (Track A: CNN transfer learning)

Test task "Junior Computer Vision / Machine Learning": classify a photo into Capsicum, Carrot, Cucumber, Potato,
Raddish, Squash (Kaggle `preethi866/vegetable-dataset`). The emphasis is on an honest evaluation: the data turned out to contain
heavy leakage in the official split, so everything is evaluated on a group-aware split that I built.

* Report (2 pages): [`reports/report.md`](reports/report.md)
* Working log with every number and decision: [`PROGRESS.md`](PROGRESS.md)

## Headline results (held-out `models/best.pt`, own group-aware test split, 966 images from 42 groups)

| | own group-aware split | official Kaggle split (retrained, same config) |
|---|---|---|
| accuracy | **93.2 %** (95 % CI 78-100 %, group bootstrap) | 100 % (test files identical to train: 717 of 799) |
| macro-F1 | 0.926 | 1.000 |

66 of the 966 images are misclassified, 63 of them in one held-out Capsicum video. Expect a clear drop on real shop photos
(see the report).

## Setup

Developed with Python 3.12.5 on Windows (Python 3.10-3.12 should work; torch 2.5.1 has no wheels for 3.13).

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt   # one command, CPU build of torch (extra index is inside the file)
```

Pinned versions: torch 2.5.1 (CPU), torchvision 0.20.1, numpy 1.26.4, pandas 2.2.3, scikit-learn 1.5.2, pillow 10.4.0,
imagehash 4.3.1, matplotlib 3.9.2, seaborn 0.13.2, tqdm 4.66.5, onnx 1.17.0, onnxruntime 1.20.1, grad-cam 1.5.4.
`predict.py` itself needs only torch, torchvision, pillow (and onnxruntime if `--onnx` is used).

## Prediction (what you run on your closed set)

```bash
python predict.py --input path/to/folder --output preds.csv
```

* Output `filename,label,confidence`; label is one of the 6 classes or `unknown`; confidence has 2 decimals.
* CPU only, no internet. Weights: `models/final.pt` (trained on all 3879 images) if present, otherwise `models/best.pt`.
* `unknown`: unreadable file (confidence `0.00`), or the image does not look like one of the 6 vegetables
  (energy score below the threshold in `models/unknown_config.json`). Switch off with `--no-unknown`.
* Other options: `--weights`, `--onnx models/final.onnx` (onnxruntime, ~1.5-1.8x faster), `--batch-size`, `--threads`,
  `--workers`, `--recursive`.
* Speed on the development PC (16 logical cores, ResNet18, 224 px): end-to-end ~18-28 ms per image with PyTorch and ~11 ms with
  ONNX Runtime (16 threads); ~55-118 ms / ~37 ms with 1 thread. Timings on this shared PC varied up to 2x between runs
  (see `outputs/latency.csv`, `outputs/latency_onnx.csv`; ONNX vs PyTorch was measured in the same run).

## Reproduce everything from scratch

1. Download the Kaggle dataset (https://www.kaggle.com/datasets/preethi866/vegetable-dataset, or
   `kaggle datasets download -d preethi866/vegetable-dataset` with the Kaggle CLI) and unzip it so that it contains `Dataset/Dataset/<class>/` and
   `final_dataset/final_dataset/{train,val,test}/<class>/`. Tell the code where it is:
   `set MAGNUM_DATA_ROOT=D:\path\to\archive` (PowerShell: `$env:MAGNUM_DATA_ROOT="D:\path\to\archive"`).
   The default in `src/config.py` is my local path. On Colab pass `--images-root /content/archive/Dataset/Dataset` to
   `04_train.py`, `05_eval.py`.
2. Run the scripts in this order (times on a 16-core CPU, no GPU needed):

```bash
python src/01_audit.py                         # data audit, filename parsing         ~15 s
python src/02_duplicates.py                    # MD5 / pHash / embeddings, leakage    ~2 min (downloads ResNet18 ImageNet weights once)
python src/03_split.py                         # own group-aware split + domain checks ~1 min
python src/04_train.py                         # ResNet18, 2 stages -> models/best.pt  ~9 min
python src/05_eval.py                          # test metrics, errors.csv, confusion matrix
python src/05_eval.py --make-official-split    # comparison with the official split
python src/04_train.py --split-file splits/official_split.csv --out models/official.pt --tag _official   # ~33 min
python src/05_eval.py --model models/official.pt --split-file splits/official_split.csv --tag _official
python src/06_gradcam.py                       # Grad-CAM + background-shortcut tests
python src/tools_error_sheet.py                # contact sheet of the 66 errors
python src/07_predict_checks.py                # predict.py format / parity / CPU latency
python src/08_get_ood.py                       # dev-only OOD images from Wikimedia (needs internet)
python src/08_unknown.py                       # threshold for "unknown" -> models/unknown_config.json
python src/08_onnx.py                          # ONNX export, parity, speed
python src/04_train.py --final --out models/final.pt --tag _final     # all data, shipped model  ~13 min
python src/08_onnx.py --weights models/final.pt --out models/final.onnx
```

Seeds are fixed (42), but thread scheduling on CPU is not bit-exact, so the last digits of metrics can differ slightly. The numbers
in the report come from one run; the raw outputs are kept in `outputs/` and `splits/`.

## Repository layout

```
predict.py                  inference script (root, as requested)
src/config.py, common.py    paths/constants, shared model + preprocessing code
src/01_audit.py ... 08_onnx.py   one script per step (see above), tools_error_sheet.py
models/                     best.pt (held-out), final.pt (shipped), *.onnx, unknown_config.json
splits/                     split.csv (own split), official_split.csv, folds.csv, units.csv
outputs/                    metrics, figures, error lists, latency tables, Grad-CAM images
reports/                    report.md
PROGRESS.md                 working log (decisions, findings, numbers)
```

`models/best.pt` is the model behind every reported metric. `models/final.pt` was trained on all images (train+val+test) with the
same recipe so it has seen more recording sessions, which is why it is the default for `predict.py`; its own numbers on these
images are therefore not held-out. The "unknown" threshold was fitted on `best.pt` and is reused for `final.pt`.

## Known limits (details in the report)

* Only 19 videos and 185 photos exist in the whole dataset (204 independent groups); 2-6 videos per class; one phone, four
  recording days; Raddish comes from one session.
* The test set contains 6 held-out videos, so the accuracy has a wide confidence interval.
* The model partly uses background/lighting; frames without a visible vegetable (label noise) exist in the data.
* The "unknown" detector is only moderate (AUROC 0.87 against other vegetables; 55 % of other-vegetable photos flagged with `best.pt`,
  37 % with the shipped `final.pt`); no real shop photos were available.
