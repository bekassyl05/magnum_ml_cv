# PROGRESS

## Task summary
Junior CV/ML test task (Magnum): recognise vegetables in photos, 6 classes
(Capsicum, Carrot, Cucumber, Potato, Raddish, Squash). Track A (CNN transfer
learning) + bonuses (unknown threshold, ONNX). Honest evaluation matters more
than accuracy. Deliverables: repo, README, scripts/notebook, 2-page report,
predict.py (CPU, offline, `--input DIR --output preds.csv`).
Deadline: 4 days from receipt.

## Decisions
- Track A chosen.
- Dataset stays outside the repo (read-only); paths in src/config.py
  (DATA_ALL, DATA_OFFICIAL, env override MAGNUM_DATA_ROOT).
- Group key = filename timestamp (video_id); each standalone photo is its own group.
  Step 2 confirmed near-dup clusters never merge groups or classes, so this key is consistent.
- Official final_dataset split is NOT used for evaluation (leakage). It is evaluated only
  for comparison, to show how much it inflates numbers.
- Near-dup thresholds: phash<=6 & dhash<=6 or ResNet18 cosine>=0.97 (different groups: no pair
  at phash<=5; median cosine to nearest other-group image 0.88).
- OWN SPLIT (src/03_split.py, splits/split.csv, SEED=42), unit = group, unique files only
  (3879, dedup by MD5). Policy: (1) whole videos go to one split, test never shares a video with
  train; video->split assignment is chosen by exhaustive search to be closest to 70/15/15 by
  image count, with >=1 video in train and >=1 in test for classes with >=2 videos;
  (2) 20% of each class's standalone photos (min 3) go to val and 20% to test, because photos
  are closest to real shop photos; (3) row weight = 1/(images in group) so group-balanced
  metrics can be reported next to plain metrics (frames are highly correlated);
  (4) splits/folds.csv: 5-fold StratifiedGroupKFold for robustness (see limitation).
  Result: train 2612 (67.3%), val 301 (7.8%), test 966 (24.9%); zero overlap of groups,
  MD5 and near-dup clusters verified by assert.
- Limitations of the split (go to the report): only 2-6 videos per class, so exact 70/15/15 is
  impossible. Val is small (301) and, for Capsicum/Cucumber/Potato/Raddish, photos only
  (8/3/9/5 images): use val only for early stopping, not for per-class decisions.
  Test = 6 held-out videos + 36 photos: high variance, report group-balanced metrics too.
  Cucumber, Potato, Raddish train on a single video each. Raddish is a single session
  (2025-12-10): test video is from the same day as train, so session leakage cannot be removed.
  folds.csv folds are badly balanced (e.g. fold 4 has 1-2 images of some classes): use only as a
  rough check, not as the main number.

## Findings
### Step 1: data audit
- Dataset/ = 3879 images: Capsicum 615, Carrot 905, Cucumber 308, Potato 422, Raddish 286,
  Squash 1343 (imbalance up to 4.7:1). All JPEG, RGB, 200x200 px, none corrupt.
- Filenames YYYYMMDD_HHMMSS[_frame_N].jpg. 3694 images are frames of 19 videos (Capsicum 3,
  Carrot 6, Cucumber 2, Potato 2, Raddish 2, Squash 4); only 185 are standalone phone photos
  (EXIF Samsung SM-A536E; frames have no EXIF). So only 204 independent groups.
- Dates: 2025-11-13, 11-17, 12-10, 2026-01-01. Raddish exists only on 2025-12-10.
- Surprise: official final_dataset has 5380 images vs 3879 in Dataset/.
### Step 2: duplicates and leakage
- Dataset/: 0 exact duplicates, 0 label conflicts. All 5380 official files are byte-identical to
  Dataset/ files (1501 extra copies); 1450 filenames occur in more than one official split.
- Official test: 717/799 (90%) images identical to train files; val 727/802 (91%); 108 test
  images also in val. 99.1% of test and 99.5% of val images share a video/photo group with
  train (nearest-train cosine median 1.000 vs 0.88 for other groups).
- Conclusion: official split measures memorisation; unusable for evaluation.
### Step 3: domain gap (outputs/domain_checks.csv)
- Frames vs photos: frames mean brightness 99.7 vs photos 89.4; Laplacian variance (high-frequency
  energy) 1062 vs 87, i.e. photos are ~12x smoother (likely downscaled from full-res), frames
  sharper/noisier (video compression). 95% of training data are frames, so photos are a shifted domain.
  Report metrics separately for frames and photos; use blur/sharpness augmentation.
- Background colour differs by capture day (border RGB, e.g. Potato 70/64/51 on 2025-11-13 vs
  109/95/84 on 2026-01-01). A logistic regression on 16x16 mean colours predicts the capture date
  with 69.2% accuracy (group CV) and the class with 41.8% (macro-F1 0.31; Cucumber recall 5%,
  Raddish 12%, Carrot 58%, Squash 55%). Guessing the majority class per date gives 42.0% vs 34.6%
  for always "Squash". So colour/background alone gives a weak but real class signal (the
  floor to beat; a CNN must clearly exceed ~42%) and identifies the session well.
- Other differences from real shop photos: 200 px images, clean single-object scenes, one phone,
  4 capture days, 6 vegetables only (closed set, no empty shelf, packaging, price tags,
  several vegetables or mixed shelves), video frames dominate.

## Training (Step 4; train+val only, test untouched)
- Config: ResNet18, ImageNet weights, 224 px (eval: Resize+CenterCrop, shared in src/common.py), AdamW
  wd 1e-4, batch 32, label smoothing 0.05, SEED 42, CPU (~1 min/epoch, 9 epochs, 9.0 min).
  Class-weighted CE (inverse frequency): Capsicum 1.00, Carrot 0.69, Cucumber 2.46, Potato 1.42,
  Raddish 2.58, Squash 0.48. The 113 train photos (4.3%) are sampled 3x more often (closest to real photos).
  Stage 1: head only, 3 epochs, lr 1e-3. Stage 2: layer4 + head, lr 1e-4 / 3e-4, cosine, BatchNorm stats of
  frozen layers kept fixed; early stopping (patience 4) on group-balanced val macro-F1 (tie-break: weighted loss).
- Augmentations (why): RandomResizedCrop scale 0.5-1 (framing/scale differ on shelves; >=0.5 keeps the
  vegetable in view); horizontal flip (no handedness); rotation +-20 deg p=0.5 (tilted camera);
  ColorJitter with tiny hue (shop lighting, and fights the per-day colour/background shortcut from Step 3,
  but colour is a true class cue so hue<=0.03); grayscale p=0.05 (reduce colour reliance, not remove);
  GaussianBlur p=0.5 (photos are ~12x smoother than the video frames that make up 95% of train);
  RandomErasing p=0.25 (price tags, hands, neighbouring produce).
  Not used: vertical flip (gravity direction is kept in real photos), 90-degree rotations, scale<0.5,
  heavy hue shifts (would destroy the colour cue), mixup/cutmix (hard to audit on tiny data, no time).
- Val result of the selected epoch 5: accuracy 94.4%, macro-F1 0.906, group-balanced macro-F1 0.998;
  frames 93.6% (n=265), photos 100% (n=36). Recall 1.00 for all classes except Squash 0.91
  (17 of 189 val Squash frames predicted as Capsicum). Train acc 99.8% while val loss rises after epoch 5
  (0.35 -> 0.43-0.55): overfitting to the few training videos.
- Caveat: val is NOT a reliable performance estimate or a sharp selection tool: 39 groups only, the
  group-balanced F1 is dominated by photos (each group weighs 1), differences between epochs are 1-2
  groups, and val photos come from the same sessions as train photos. Final numbers come from test (Step 5).

## Evaluation (Step 5; models/best.pt, test used once)
| | Own group-aware split | Official Kaggle split (retrained, same config) |
|---|---|---|
| train / test files | 2612 / 966 (42 groups) | 3779 / 799 |
| Accuracy | 93.2% (95% CI 78.1-100%, group bootstrap) | 100% |
| Macro-F1 | 0.926 | 1.000 |
| Group-balanced acc / macro-F1 | 0.990 / 0.988 | 1.000 / 1.000 |
| Frames / photos accuracy | 92.9% (n=930) / 100% (n=36) | 100% / 100% |
| Test files byte-identical in train | 0 | 717 of 799 (90%); acc 100% on them, and also 100% on the other 82 |
| ECE | 0.114 | 0.065 |
- Own split, per class (P/R/F1): Capsicum 0.99/0.63/0.77, Carrot 1/1/1, Cucumber 0.67/1.00/0.80, Potato 0.99/0.98/0.99,
  Raddish 0.99/1.00/1.00, Squash 1.00/1.00/1.00. 66 errors, 63 of them in ONE held-out video
  (Capsicum 20251117_134205: 61.6% of its 164 frames predicted Cucumber, confidence up to 0.87);
  the 5 other test videos are 98-100%. The Capsicum/Cucumber numbers are therefore one video, not a class property.
- Interpretation: (1) the official split gives 100% on everything, even on the 82 test files that are not byte-identical to train,
  because 99% of its test frames come from videos that are also in train: it measures recognition of already seen videos,
  not generalisation. (2) Holding out whole videos costs 7 accuracy points and exposes a real failure, so the official
  number would have hidden it. (3) The gap is not purely leakage: the two test sets differ in composition, but the
  direction and the 717/799 duplicates make leakage the main cause. (4) Even the honest number is optimistic and
  uncertain: only 42 groups (6 test videos), CI 78-100%, and 5 of 6 test videos are perfectly recognised although
  they come from the same phone, sessions and clean backgrounds as train. (5) The model is under-confident overall
  (mean confidence 0.84 when right) but systematically confident when a whole video is wrong (0.58 on average, up to 0.87),
  so a confidence threshold will only partly catch this kind of error.
- Expected drop on the closed Magnum shop set: likely substantial and not measurable here. Reasons: closed shop photos have
  shelves, packaging, price tags, several vegetables, different cameras/resolution/lighting; training had one phone,
  4 capture days, 200 px clean scenes, 2-6 videos per class; one unseen-condition video alone already lost 38% of its
  frames. Reference points: majority-class baseline 26.5%, colour-only logistic regression 41.8%.

## Grad-CAM and background reliance (Step 6; models/best.pt, test images; outputs/gradcam/)
- CAM mass outside the central 60%x60% box (uniform would be 0.64): 0.50 on average (ratio 0.79), so attention is
  moderately centre-biased. Correctly classified images: 0.46 (ratio 0.73); misclassified: 0.62 (ratio 0.97, almost uniform).
  By class: Cucumber 0.35, Carrot 0.44, Potato 0.50, Raddish 0.54, Capsicum 0.54, Squash 0.59. (The centre box is a rough
  proxy: vegetables do not always sit in the centre; the misclassified group is dominated by one video.)
- Removing parts of the image (all 966 test images; orig 93.2%): vegetable box only on grey background 87.2%, on blurred
  background 84.6%; centre box hidden, only periphery kept 76.7% (chance 16.7%, majority 26.5%, colour-only LR 41.8%).
  So the model uses both the centre and the surroundings, and a lot of its accuracy survives without the centre.
  Caution: a close-up vegetable often extends beyond the central box, so the centre-hidden score is NOT a pure background score.
- Per class: Capsicum depends on the centre (recall 0.63 -> 0.14 when hidden), while Carrot 0.93, Raddish 0.88, Potato 0.87,
  Squash 0.86, Cucumber 1.00 stay high without it. Raddish drops to 0.81/0.75 when only the centre is kept, and Raddish
  is a single capture session: a background/session cue is plausible but unproven.
- 'Cucumber' acts as a fallback class: with the centre hidden, Capsicum -> Cucumber in 146/172 cases, and 12-16 images of
  Carrot/Potato/Raddish/Squash also go to Cucumber. The weak test video (Capsicum 20251117_134205, the only Capsicum video
  from 2025-11-17) was also predicted Cucumber. Its border colour/brightness (RGB 76/78/61, brightness 91) is far from both
  Capsicum training videos (both 2026-01-01: 94/104/70 and 93/99/75, brightness 95-100) and close to the Cucumber training
  video (73/77/49, brightness 92). This is consistent with reliance on session lighting/background colour, but with one
  held-out video it is a correlation, not proof.
- Meaning for real shelf photos: the model has not learned a clean 'shape of the vegetable' concept; it mixes object and context
  cues from 4 capture sessions. On a shelf, backgrounds, lighting and neighbouring produce will all be new, so errors would be
  systematic (whole shelves wrong, often 'Cucumber') and confident. Mitigations: train the shipped model on all videos
  (includes the 2025-11-17 session), more colour/brightness/background augmentation, and collect real shop photos.
- Visual review (overview.png, compare_capsicum_failing_vs_train.png):
  * Correct examples: CAM sits on the vegetable for Capsicum, Cucumber, Raddish, Carrot, but partly also on neighbouring items
    or context (Squash: strongest region in the gap between pieces; Raddish: dark area beside the object).
  * Errors: attention is on the background, edges, glare or hand instead of the vegetable. Failing Capsicum video: frame 1 has only
    slivers of pepper at the top/bottom edges and CAM on a glossy countertop reflection (Cucumber 0.87); frame 2 is covered by a hand,
    CAM on the hand edge; frame 3 shows a clear large green pepper in a hand, yet CAM is on the dark/glare area at the right edge
    (Cucumber 0.29): a genuine failure. Potato->Cucumber (0.52): CAM on the noisy right edge, not on the pale potato.
    Squash->Capsicum (0.37): CAM on a partly visible object at the left edge, not on the squash in the centre.
  * Training Capsicum references (other videos, plus one photo of the same 2025-11-17 session) are correct (0.89-0.95) with CAM on
    the pepper. The failing video differs: dark counter with glare and specks, hand holding the pepper, pepper often partly out of view.
  * Label noise: frame labels are inherited from the video, so some frames show no (or almost no) vegetable (hand, empty counter). They
    cannot be classified by content, so part of the 63 errors in that video is label noise rather than model failure, and such frames
    in train teach the model to predict a class from hand/background/lighting, a plausible cause of the context reliance above.
    Manual count on outputs/gradcam/error_sheet.jpg (66 errors, rough, +-3): ~22 frames show no vegetable (empty counter / hand only),
    ~17 show only a part of a pepper at the image edge, ~27 show a clearly visible vegetable (about 11 of them a lime-yellow pepper,
    a colour not present in training, hypothesis: read as cucumber; ~13 green peppers held in a hand; 3 errors from other videos).
    Excluding the ~22 no-vegetable frames, test accuracy would be roughly 95% (900/944); this is an estimate, not a reported metric.
  * Task relevance: Magnum also asks whether a vegetable is present at all. Frames without a vegetable are exactly 'empty shelf' cases,
    and the model answers them with a confident class (0.87). The unknown bonus (Step 8) should address this.

## predict.py (Step 7)
- `python predict.py --input DIR --output preds.csv`: CPU-only, offline, weights from models/final.pt else models/best.pt (relative to the
  script), preprocessing identical to evaluation (Resize 224, CenterCrop 224, ImageNet norm) plus EXIF orientation fix, RGBA->white,
  jpg/jpeg/png/webp/bmp, batch inference with threaded decoding. Unreadable files do not crash it: row `unknown,0.00` + warning.
  Optional `--unknown-threshold` (bonus), `--recursive`, `--threads`, `--batch-size`.
- Checks (src/07_predict_checks.py): 20 test images + corrupt.jpg + empty.png + notes.txt -> 22 rows, header exactly
  `filename,label,confidence`, valid labels, 2-decimal confidence, .txt ignored: PASS. Parity with src/common.py: identical labels,
  max |confidence difference| 0.0047 (= 2-decimal rounding): PASS.
- CPU latency (AMD64 Family 25 Model 68, 16 logical cores, torch 2.5.1+cpu, ResNet18, 224 px, 100 images after 10 warm-ups; outputs/latency.csv):
  | threads | end-to-end per image, batch 1 (mean / median / p95) | model forward, batch 32, per image |
  |---|---|---|
  | 1 | 66.4 / 52.9 / 173.6 ms | 46.9 ms |
  | 2 | 47.6 / 44.8 / 60.9 ms | 41.9 ms |
  | 4 | 34.0 / 34.6 / 35.9 ms | 24.8 ms |
  | 16 | 20.3 / 18.2 / 19.6 ms | 9.0 ms |
  Preprocessing is ~3-5 ms because the dataset images are 200 px; real shop photos (several MP) will add decoding time (JPEG draft
  mode reduces it). The single-thread p95 outlier comes from a few slow first iterations (OS noise), not from the model.
  A whole folder of 22 files took 0.3 s (14 ms/image with 16 threads).
  A later re-run (final predict.py, busier PC) gave end-to-end 117.8 / 62.2 / 48.5 / 27.9 ms at 1/2/4/16 threads; outputs/latency.csv holds this
  latest run. Timings on this shared PC vary up to 2x between runs, so they are reported as ranges. CLI check after the Step 8 rewrite:
  format PASS, parity PASS (max confidence difference 0.0050 = rounding), default (final.pt, unknown on) and --onnx outputs identical on 20 images.

## Bonuses and shipped model (Step 8)
### Bonus 1: 'unknown' (src/08_get_ood.py, 08_unknown.py; outputs/ood_threshold.png, ood_thresholds.csv, models/unknown_config.json)
- Data: in-distribution = val (301 images). OOD dev set = 188 Wikimedia Commons photos found by keyword (tomato, onion, garlic, broccoli, cabbage,
  cauliflower, eggplant, apple, banana, lemon, orange, eggs, bread, supermarket shelf, kitchen counter, empty table; 12 per keyword,
  10 for banana and orange after HTTP 429 errors). Web search results, not checked image by image (outputs/ood_sheet.jpg).
  Independent checks, not used to choose the threshold: test images and the 22 frames without any vegetable (taken from the model's
  own errors, so a hard and biased set).
- Score comparison, AUROC val-ID vs other vegetables: max softmax 0.811, temperature-scaled max softmax 0.824, energy (logsumexp of
  logits) 0.870 (AUPR 0.877 / 0.887 / 0.900) -> energy. Temperature scaling fitted on val: T = 0.44; ECE improves 0.175 -> 0.029
  on val and 0.114 -> 0.031 on test. Confidence in preds.csv uses this temperature.
- Criterion: keep >=95% of val images, then reject as many OOD images as possible -> energy threshold 0.325.
  | val ID kept | threshold | other veg rejected | no-vegetable frames rejected | test ID kept | accuracy of kept test images |
  |---|---|---|---|---|---|
  | 99% | 0.045 | 35.6% | 0% | 99.8% | 93.2% |
  | 97.5% | 0.154 | 41.0% | 4.5% | 99.2% | 93.3% |
  | **95%** | **0.325** | **55.3%** | **9.1%** | 97.9% | 93.6% |
  | 90% | 0.519 | 62.2% | 27.3% | 94.7% | 94.5% |
- Honest reading: the detector is only moderate (AUROC 0.87). At the chosen point it rejects 55% of other vegetables, only 2 of 22 frames without a
  vegetable, and almost none of the errors of the weak Capsicum video (96.3% of its frames are kept: its errors are confident), so the
  accuracy of the kept predictions rises only from 93.2% to 93.6%. The Raddish test video loses 11% of its frames (kept 88.8%).
  Commons photos differ visually from the dataset, so 55% is probably optimistic for real shelves; on real shop photos with a domain shift
  logits may shrink and valid vegetables could be rejected (switch off with --no-unknown). A threshold cannot replace a trained
  'no vegetable / other' class (outlier exposure with real shelf negatives): not done, goes into 'what I would do next'.
- The threshold was calibrated on best.pt (held-out val) and is reused for final.pt, which cannot be recalibrated because it saw val.
  Check with predict.py on data/ood/other_veg: final.pt labels 70 of 188 (37.2 %) as unknown, while best.pt rejects 55.3 % at the same
  threshold. So the threshold transfers only partly (final.pt is more confident): the shipped setting is more conservative (fewer false
  'unknown' on valid vegetables, fewer detected others). On the 22 no-vegetable frames final.pt gives 0 unknown; this is not informative
  because those frames belong to its training data (best.pt: 2 of 22).
### Bonus 2: ONNX (src/08_onnx.py; models/best.onnx 44.7 MB, models/final.onnx; outputs/latency_onnx.csv)
- Parity on 64 test images: max |logit difference| 6.9e-6 (final: 5.7e-6), identical argmax 100%.
- CPU speed, mean ms (best.pt; PyTorch vs onnxruntime):
  | threads | end-to-end per image, batch 1: PyTorch / ORT (speed-up) | forward per image, batch 32: PyTorch / ORT (speed-up) |
  |---|---|---|
  | 1 | 55.0 / 36.8 (1.49x) | 47.3 / 32.0 (1.48x) |
  | 2 | 37.5 / 23.4 (1.60x) | 27.0 / 17.6 (1.53x) |
  | 4 | 28.3 / 16.1 (1.76x) | 18.3 / 12.5 (1.47x) |
  | 16 | 17.6 / 10.7 (1.64x) | 8.4 / 5.3 (1.58x) |
  ONNX Runtime is 1.5-1.8x faster end-to-end and 1.5-2.1x faster in the model forward (batch 1: 6.6 vs 13.7 ms at 16 threads). A first run
  (without session.intra_op.allow_spinning=0) had a 46.6 ms ORT end-to-end time at 16 threads and batch 1 although its forward was
  as fast as PyTorch's; switching the spinning of idle ORT threads off removed it (10.7 ms), consistent with the idle ORT threads
  competing with image decoding. PyTorch numbers vary between runs by ~10-20% (Step 7: 66/48/34/20 ms at 1/2/4/16 threads).
### Shipped model
- models/final.pt: trained on all 3879 images (train+val+test) with the epoch count selected on the held-out run (3 head + 2 fine-tune
  epochs, 12.7 min CPU). Its val/test numbers are not held-out; all reported metrics come from models/best.pt. predict.py uses
  models/final.pt when present.

## Artifacts
- src/config.py, common.py, 01_audit.py, 02_duplicates.py, 03_split.py, 04_train.py, 05_eval.py, 06_gradcam.py, 07_predict_checks.py,
  08_get_ood.py, 08_unknown.py, 08_onnx.py,
  tools_error_sheet.py, predict.py (repo root), requirements.txt (single-command install, CPU torch), .gitignore
- outputs/latency.csv, preds_test20.csv, predict_test_input/
- outputs/gradcam/ (overview.png, ex*.png, compare_capsicum_failing_vs_train.png), gradcam_stats.csv, shortcut_tests.csv, gradcam_summary.json
- models/best.pt (own split), models/official.pt (official split, comparison only), outputs/train_log*.csv, train_summary*.json,
  fig_train_curves*.png, test_metrics*.json, test_per_class*.csv, test_per_group*.csv, test_predictions*.csv, errors*.csv, fig_confusion*.png
- outputs/: inventory.csv, audit_summary.csv, clusters.csv, leakage_report.txt,
  video_split_matrix.csv, split_balance.csv, domain_features.csv, domain_checks.csv, fig_*.png
- splits/: split.csv, folds.csv, units.csv

## Next step
Steps 1-8 done (data audit, leakage, own split, training, test evaluation, Grad-CAM, predict.py, bonuses, final.pt).
Step 9: README + 2-page report + clean-environment check (fresh venv, predict.py with final.pt + unknown_config.json).
Note: official.pt was trained under the 30-min budget (13 epochs, val perfect from epoch 4).
