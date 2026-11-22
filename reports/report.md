# Report: vegetable recognition from photos (Track A)

## 1. What I found in the data
* 3879 images (Capsicum 615, Carrot 905, Cucumber 308, Potato 422, Raddish 286, Squash 1343), all 200x200 JPEG, none corrupt.
* The file names (`YYYYMMDD_HHMMSS[_frame_N].jpg`) show that 3694 images are **frames of only 19 videos**; the other 185 are phone photos
  (EXIF: Samsung SM-A536E). So there are only **204 independent groups**, 2-6 videos per class, recorded on four days
  (2025-11-13, 11-17, 12-10, 2026-01-01). Raddish exists only on one day (12-10), so class and session are confounded.
* Exact duplicates inside `Dataset/`: none. But the official `final_dataset` (5380 files) is not a partition of `Dataset/`: **every
  file is a byte-identical copy of a `Dataset/` file** (3879 unique files, 1501 extra copies). **717 of 799 official test files (90 %) are
  byte-identical to a train file**, and 99 % of test frames come from a video that is also in train. I checked with MD5, perceptual
  hashes (pHash+dHash), ResNet18 embeddings and filename/video-id overlap; near-duplicate clusters never mix two groups or two classes.
* **The official split cannot be trusted**: it measures memorisation of videos, not generalisation.

## 2. How I split the data
Unit = group (video, or a single photo). Whole videos go to one split, so no test video appears in train. A search over all
video-to-split assignments picks the one closest to 70/15/15; 20 % of the photos of each class go to val and to test (photos are
closest to real photos). Result: train 2612, val 301, test 966 (42 groups); zero overlap of groups, MD5 and near-duplicate
clusters, checked by assertions. Limits: val is tiny (for four classes it contains photos only), the test set contains 6 videos, and
Cucumber/Potato/Raddish train on one video each. Metrics are therefore reported with a group-bootstrap interval and also
group-balanced.

## 3. Model
ResNet18 (ImageNet weights), 224 px, two stages: head only (3 epochs), then last block + head with lower learning rates;
class-weighted loss, label smoothing 0.05, photos sampled 3x more often. Model selection on the group-balanced val macro-F1 (epoch 5).
Augmentations were chosen from what the data analysis showed: random-resized crop (framing on shelves), flip, rotation +-20 deg,
colour jitter with tiny hue (lighting differs per day, but colour is a true class cue), Gaussian blur (photos are 12x smoother than
video frames, 95 % of training data are frames), random erasing (hands, price tags). Not used: vertical flip, strong hue shifts, mixup/CutMix.

## 4. Results on the held-out test split
| | own group-aware split | official split (same recipe, retrained) |
|---|---|---|
| accuracy | **93.2 %** (95 % CI 78.1-100 %) | 100 % |
| macro-F1 | 0.926 | 1.000 |
| group-balanced acc / F1 | 0.990 / 0.988 | 1.000 / 1.000 |
| frames / photos | 92.9 % / 100 % (36 photos) | 100 % / 100 % |

Per class (precision/recall): Carrot, Raddish, Squash, Potato ~1.00; Capsicum 0.99/0.63, Cucumber 0.67/1.00. **63 of the 66 errors
come from a single held-out Capsicum video** (61.6 % of its frames predicted Cucumber); the other five test videos are 98-100 %.
The official split would have hidden this failure completely. The gap (100 % vs 93.2 %) is mainly leakage, but the two test sets are
not identical in composition. Calibration: ECE 0.114 -> 0.031 after temperature scaling.

## 5. Why the model fails (Grad-CAM and tests)
* In correct cases the CAM is on the vegetable (partly also on neighbouring items); in errors it is on background, glare, edges or a
  hand (share of CAM mass outside the central box 0.97 of uniform for errors vs 0.73 for correct predictions).
* Hiding the central box leaves 76.7 % accuracy (keeping only the box: 87.2 %), so the model uses context as well as the object.
  Capsicum depends on the centre (recall 0.63 -> 0.14 when hidden); "Cucumber" acts as a fallback class.
* The weak video differs in lighting (dark counter, glare, hand holding the pepper). By manual count (rough) ~22 of the 66 errors show
  **no vegetable at all** (video-frame labels are inherited from the video: label noise), ~17 only a part of one, ~27 a clearly visible
  vegetable, among them a lime-yellow pepper that has no counterpart in training (hypothesis: read as cucumber).

## 6. predict.py and the bonuses
`python predict.py --input DIR --output preds.csv`: CPU, offline, batched, EXIF-aware, corrupt files give `unknown,0.00` instead of a
crash. ~18-28 ms per image end-to-end with PyTorch (16 threads), 55-118 ms with 1 thread (runs on this shared PC differ by up to 2x). **ONNX** export matches PyTorch (max logit
difference 7e-6) and runs 1.5-1.8x faster (e.g. 10.7 vs 17.6 ms, 16 threads).
**Unknown**: energy score on logits (AUROC 0.87 vs 0.81 for max softmax) between val images and 188 other-vegetable/shelf photos from
Wikimedia; threshold = the value that keeps 95 % of val images (fixed before looking at test). It rejects 55 % of the other-vegetable
photos and keeps 97.9 % of test images, but only 2 of 22 frames without a vegetable and almost none of the confident errors; accuracy
of kept test images rises only from 93.2 % to 93.6 %. It is a modest safety net, not a "no vegetable" detector. With the shipped `final.pt` (more confident, trained on all data) the same
threshold flags 37 % of the other-vegetable photos (70 of 188) instead of 55 %, i.e. a more conservative setting.

## 7. Expected drop on your closed store set
I expect a clearly lower accuracy than 93 %, but cannot measure how much. Reasons: shelves, packaging, price tags and several products per
photo (training: one phone, four days, clean close-ups, one object, 200 px); backgrounds and lighting will be new, and one unseen
condition already cost 38 % of a class's frames here; the model partly relies on context; errors will be systematic (a whole shelf
wrong, often as Cucumber) and confident; the unknown threshold may reject valid vegetables when the logits shrink under domain shift.

## 8. What did not work / not done
The unknown detector is only moderate; no real store photos for validation; val too small for sharp model selection; test accuracy rests
on 6 videos; the lime-yellow-pepper explanation and the context reliance are hypotheses supported by Grad-CAM, not proven.
The shipped `final.pt` (all data, same recipe) cannot be evaluated; all numbers come from `best.pt`.

## 9. What I would do next
(1) Collect 100-200 real shelf photos per class (several stores, cameras, lighting) and evaluate on them first. (2) Add an explicit
"no vegetable / other" class trained with real negatives instead of a threshold. (3) Clean video labels (drop or relabel frames without a
visible vegetable). (4) Second held-out split with different videos to measure the video-level variance. (5) Stronger colour/background
augmentation and multi-crop inference for wide shelf photos.
