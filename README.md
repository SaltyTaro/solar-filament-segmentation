# Solar filament instance segmentation

This repository holds our entry to the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026). The task is to find each solar filament in a 2048 × 2048 full-disk H-alpha image and return one mask per filament, with no pixel shared between masks.

The final submission (Kaggle 57348584) scores 0.47467 pooled five-fold out-of-fold PQ under the organizer's scorer and 0.41 on the public leaderboard. Training uses only the supplied training images and annotations, and inference uses only the supplied test image. We did not use MAGFiLO annotations or any other external labels.

## Method

Five model families are trained on each of five folds. The folds group images by UTC date, so every annotator's drawing of an image, and every image from the same day, stays in one fold.

| Family | Input | Notes |
|---|---|---|
| YOLO11m-seg | 1536 px | 60 epochs, cosine schedule, EMA, scale and shift jitter |
| Mask R-CNN, ResNet-50 FPN v2 | 1536 px | 36 epochs, COCO initialization |
| YOLO11m-seg | 2048 px | 60 epochs |
| YOLO26m-seg | 1536 px | 60 epochs, end-to-end head |
| Dense ConvNeXt-Small U-Net | native 1024 crops | 80 epochs; predicts the annotators' soft consensus of filament pixels and of the supplied spines |

The dense family exists because of how the labels look. All 8,199 annotations are single polygons, 99.6% of them form one connected region, and only 59 of 36,064 instance pairs drawn by the same annotator come within 5 px of each other. Instances are therefore the connected components of a good filament map. The dense model draws that map at full resolution, while the box and query heads predict masks at a quarter of the input size, where a typical filament (about 11 px wide) is roughly 2 mask pixels across at 1536 px input.

For each fold, test-time prediction runs in five steps:

1. Every model predicts all eight dihedral views of the image.
2. Candidates from the 40 sources (5 families × 8 views) are clustered by mask IoU and fused by pixel voting. Each pixel then belongs to one instance.
3. A ResNet-34 U-Net crop refiner, cross-fitted on out-of-fold instances, redraws each fused mask. The refined masks are re-owned in fused order.
4. A gradient-boosted rescorer, also trained on out-of-fold instances, estimates how likely the annotators are to have drawn each instance.
5. The five fold ensembles are fused once more, pixels are owned in probability order, and instances below the out-of-fold threshold are dropped.

Every number below is pooled out-of-fold PQ over all 1,154 annotator records (IoU > 0.5, counts pooled as in the organizer's notebook) together with the public score of the matching submission.

| Ensemble | OOF PQ | Public |
|---|---|---|
| YOLO11m-1536 and Mask R-CNN, 8-view TTA, fusion, rescorer | 0.46271 | 0.40 |
| adding YOLO11m-2048 | 0.46550 | 0.41 |
| adding YOLO26m | 0.46582 | 0.41 |
| adding the crop refiner | 0.47219 | 0.41 |
| adding the dense family (final) | 0.47467 | 0.41 |

Human annotators agree with each other at 0.34 PQ, and the objects two annotators both drew score 0.33 against the third, so most of the remaining error is disagreement between annotators. The report covers the ablations, including the branches that did not help (CLAHE, native 2048 input, pseudo-labels, longer schedules, a CNN verifier, Mask2Former, per-instance mask variants).

## Reproduce the submitted predictions

You need Python 3.12 and a CUDA GPU. We ran this on one NVIDIA L40S.

```bash
pip install -r requirements.txt
export SOLAR_STORAGE_ROOT=/path/to/storage
```

Put the supplied competition data in `$SOLAR_STORAGE_ROOT/data/raw/MAGFiLO_1.0_Kaggle_2026/` (with its `train/` and `test/` folders) and the released weights, the Kaggle dataset [`notoverkil/solar-filament-weights`](https://www.kaggle.com/datasets/notoverkil/solar-filament-weights), in `$SOLAR_STORAGE_ROOT/weights/final/`. The weights folder holds the 25 fold models (config plus inference-only weights each), the five refiner models, the rescorer, the frozen fold split, the submitted CSV (`submission.csv`) and a `manifest.json` with every file's SHA-256. Then run:

```bash
python infra/reproduce-submission.py --weights $SOLAR_STORAGE_ROOT/weights/final \
    --output $SOLAR_STORAGE_ROOT/runs/reproduction
```

The script checks each released file against the manifest, runs the 25 TTA passes (52 minutes on one L40S), then fusion, refinement, adoption and the level-2 fusion (about 20 minutes on 8 CPU cores), and writes `predictions.csv` with a `reproduction.json` that compares it with the submitted CSV's hash and matches the two files mask by mask. Each stage resumes from its outputs if interrupted. On a fresh Linux worker, this rebuilt 1,286 of the 1,287 submitted masks at IoU above 0.5 (mean IoU 0.9998). The files are not byte-identical because the original test candidates came from several GPU types.

## Notebook

`pipeline.ipynb` runs every stage of the pipeline on one test image with the released weights: TTA for all 25 fold models, fusion within each fold, crop refinement, rescoring and the fusion across folds, ending with the predicted masks drawn on the image. It also covers the data, the folds and the metric. `infra/generate-notebook.py` writes the notebook.

## Train from scratch

The frozen split is in the weights release (`splits/`); copy it to `$SOLAR_STORAGE_ROOT/data/splits/` to reuse it exactly, or rebuild it with `python -m solar_filament.cli ingest` and `python -m solar_filament.cli folds`. Pretrained weights go in `$SOLAR_STORAGE_ROOT/weights/pretrained/`: `yolo11m-seg.pt` and `yolo26m-seg.pt` from the Ultralytics releases and `convnext_small-0c510722.pth` from TorchVision. Mask R-CNN downloads its COCO weights itself. The configs record each file's SHA-256.

Each fold model trains and then writes its held-out and test TTA candidates:

```bash
python infra/train-and-tta.py --family yolo --config configs/phase2/yolo11m1536-e60-f1.yaml --output $SOLAR_STORAGE_ROOT/runs/p2-yolo11m1536-e60-f1
```

| Family | `--family` | Fold 0 | Folds 1 to 4 |
|---|---|---|---|
| YOLO11m-1536 | `yolo` | `configs/phase1/yolo11m1536-long-f0.yaml` | `configs/phase2/yolo11m1536-e60-f{1,2,3}.yaml`, `configs/phase2/yolo11m1536-e60-f4-b4.yaml` |
| Mask R-CNN | `maskrcnn` | `configs/phase1/maskrcnn1536-long-f0.yaml` | `configs/phase2/maskrcnn1536-e36-f{1..4}.yaml` |
| YOLO11m-2048 | `yolo` | `configs/phase3/yolo11m2048-e60-f0.yaml` | `configs/phase3/yolo11m2048-e60-f{1..4}.yaml` |
| YOLO26m | `yolo` | `configs/phase3/yolo26m1536-e60-f0.yaml` | `configs/phase3/yolo26m1536-e60-f{1..4}.yaml` |
| Dense U-Net | `dense` | `configs/phase4/dense-convnexts1024-e80-f0.yaml` | `configs/phase4/dense-convnexts1024-e80-f{1..4}.yaml` |

A fold model, including its TTA, takes between one and four hours on one L40S. Fold 4 of YOLO11m-1536 used batch 4 with gradient accumulation to fit a 24 GB GPU, keeping the same effective batch.

With all 25 models trained, the remaining steps build the out-of-fold tables, train the refiner and rescorer, and produce the test predictions:

```bash
# Level-1 fusion and rescorer features for each fold (one comma-separated list of tta-validation folders per fold)
python infra/phase2-oof.py --fold-dirs F0 F1 F2 F3 F4 --models hgb --output $ROOT/artifacts/oof
# Cross-fitted crop refiner on the out-of-fold instances (GPU)
python infra/refine-masks.py train --features $ROOT/artifacts/oof --output $ROOT/runs/refiner
# Swap in the refined masks and relearn selection on them
python infra/refine-masks.py adopt --source $ROOT/runs/refiner/features --pattern "features-f*.csv" --output $ROOT/artifacts/adopted
python infra/rescore.py $ROOT/artifacts/adopted/features-f*.csv --model hgb --output $ROOT/artifacts/adopted/rescore-hgb.json \
    --save-model $ROOT/artifacts/adopted/rescorer-hgb.pkl
```

`infra/export-weights.py` then packs the trained models into the release format, and `infra/reproduce-submission.py` builds the test predictions from it.

## Evaluate a prediction file

```bash
python -m solar_filament.cli evaluate predictions.csv --fold 0 --output evaluation/
```

This scores one prediction set per image against every annotator record separately and pools the matched IoU and the TP, FP and FN counts, as the organizer's notebook does. It also writes per-object IoU, union Dice, and the counts of annotated objects touched by several predictions and of predictions touching several objects. `python -m solar_filament.cli validate-submission predictions.csv` checks the submission format and that masks do not overlap.

## Repository layout

- `src/solar_filament/`: the package (data access, folds, metrics, fusion, Mask R-CNN training, submission I/O).
- `infra/`: training loops (`train-yolo.py`, `train-dense.py`, `family_training.py`), TTA inference, fusion and rescoring, the crop refiner, and the release and reproduction scripts.
- `configs/`: the training configuration of each of the 25 fold models, grouped by project phase.
- `tests/`: unit tests (`python -m pytest -q`).
- `pipeline.ipynb`: the demonstration notebook.

The code reads all data and writes all outputs under one storage root: `SOLAR_STORAGE_ROOT`, or `storage/` inside the checkout when the variable is unset. The same commands work on Linux and Windows.

## License

The code is released under the GNU Affero General Public License v3.0 (`LICENSE`). That is the license of Ultralytics, which the YOLO models are built on, and the released weights use it too because the YOLO weights are fine-tuned from Ultralytics checkpoints. The license is OSI-approved and allows commercial use.
