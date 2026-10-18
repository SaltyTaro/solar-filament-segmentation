"""Write the demonstration notebook (``pipeline.ipynb``) that walks the final pipeline end to end.

The notebook runs every stage of the submitted pipeline for one supplied test
image with the released weights (25 fold models, the crop refiner and the
rescorer), using the same functions as the full run, and gives the one-line
command that rebuilds all 180 test predictions. Execute it on a CUDA machine
after downloading the weights:

    python infra/generate-notebook.py
    jupyter nbconvert --to notebook --execute --inplace pipeline.ipynb
"""

from __future__ import annotations

import json
from pathlib import Path

OUTPUT = Path(__file__).resolve().parents[1] / "pipeline.ipynb"

CELLS = [
    ("markdown", """# Solar filament instance segmentation: the final pipeline

This notebook walks through the submitted pipeline (Kaggle submission 57348584) on one supplied test image,
running each stage with the released weights and the same code as the full run, then shows how to rebuild
all 180 test predictions with one command.

**Pipeline.** Five model families are trained on each of five date-grouped folds: YOLO11m-seg at 1536 px,
Mask R-CNN (ResNet-50 FPN v2) at 1536 px, YOLO11m-seg at 2048 px, YOLO26m-seg at 1536 px, and a spine-aware
dense ConvNeXt-Small U-Net at native resolution. For each fold:
1. **TTA.** Every model predicts all eight dihedral views of the image.
2. **Instance fusion.** Candidates from all views and families are clustered by mask IoU and fused by pixel
   voting; each pixel is owned by one instance.
3. **Crop refinement.** A cross-fitted ResNet-34 U-Net redraws each fused mask from an image crop.
4. **Rescoring.** A gradient-boosted rescorer, trained on out-of-fold instances, scores the refined instances.

The five fold ensembles are then fused once more (level 2) and thresholded.

**Results.** Pooled five-fold out-of-fold PQ is 0.47467 (organizer scorer: IoU > 0.5, counts pooled over every
annotator record); public leaderboard 0.41. Inference uses only the supplied test image. No MAGFiLO or other
external annotations are used."""),
    ("code", """import importlib.util, json, logging, pickle, sys, time
from argparse import Namespace
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from pycocotools import mask as mask_utils

REPOSITORY = Path.cwd()
sys.path[:0] = [str(REPOSITORY / 'src'), str(REPOSITORY / 'infra')]
from solar_filament.coco import CocoIndex
from solar_filament.metrics import reference_counts
from solar_filament.storage import annotations_path, dataset_root, storage_root
from solar_filament.submission import validate_predictions

def load(name):
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), REPOSITORY / 'infra' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

tta, ensemble, refine = load('tta-inference'), load('ensemble-test'), load('refine-masks')
root = storage_root()                      # SOLAR_STORAGE_ROOT
WEIGHTS = root / 'weights' / 'final'       # the downloaded Kaggle dataset notoverkil/solar-filament-weights
WORK = root / 'runs' / 'notebook-demo'
manifest = json.loads((WEIGHTS / 'manifest.json').read_text())
print('CUDA:', torch.cuda.get_device_name() if torch.cuda.is_available() else 'not available')
print(len(manifest['models']), 'fold models;', len(manifest['files']), 'released files')"""),
    ("markdown", """## Data and validation

The supplied training set has 707 physical images with 1,154 annotator records: an image can carry one, two
or three independent drawings. The organizer's PQ scores one prediction set against every annotator record
separately and pools the counts. Folds keep every record of an image, and all images from the same UTC date,
together, so no observation is split between training and validation."""),
    ("code", """index = CocoIndex(annotations_path(root))
annotators = pd.Series({name: len(records) for name, records in index.images_by_filename.items()})
print('physical images:', len(annotators), '| annotator records:', int(annotators.sum()),
      '| annotations:', len(index.data['annotations']))
print('images by annotator count:', annotators.value_counts().sort_index().to_dict())
folds = pd.read_csv(WEIGHTS / 'splits' / 'folds.csv')
assert folds.groupby('capture_date').fold.nunique().max() == 1
folds.groupby('fold').agg(images=('filename', 'nunique'), records=('annotator_records', 'sum'))"""),
    ("markdown", """## The metric

A prediction matches an annotated filament when IoU > 0.5; PQ = (sum of matched IoU) / (TP + FP/2 + FN/2), with
counts pooled across every annotator-image record. In this toy example two of three annotated objects are matched
(IoU 0.6 and 0.8), so PQ = 1.4 / (2 + 0.5 + 0.5)."""),
    ("code", """toy = np.array([[0.6, 0.0], [0.0, 0.8], [0.3, 0.0]])   # rows: annotated objects, columns: predictions
reference_counts(toy).scores()"""),
    ("markdown", """## The released models

Each fold model is a folder with its training configuration and inference-only weights (exponential moving
average). The table lists fold 0's five families; folds 1 to 4 repeat them, each trained without its own
held-out fold."""),
    ("code", """rows = []
for model in manifest['models']:
    config = yaml.safe_load((WEIGHTS / model['run'] / 'config.yaml').read_text())
    rows.append({'fold': model['fold'], 'family': model['family'], 'run': model['run'], 'input': config.get('image_size', 'native crops'),
                 'epochs': config['epochs']})
models = pd.DataFrame(rows)
models[models.fold == 0]"""),
    ("markdown", """## Stage 1: eight-view TTA for every fold model

All 25 fold models predict the eight dihedral views of one test image. Each candidate keeps its score and its
native-resolution mask, mapped back to the original orientation."""),
    ("code", """test_images = sorted((dataset_root(root) / 'test/test_images').glob('*.jpeg'))
image_path = next(path for path in test_images if path.stem == '20140420195854Bh')   # most submitted filaments (17)
stem = image_path.stem
image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
from ultralytics.utils import LOGGER
LOGGER.setLevel(logging.ERROR)   # hide Ultralytics deprecation notices
options = Namespace(candidate_threshold=0.05, maximum_candidates=100, low_mask_threshold=0.3, image_size=None)

def source_for(model):
    folder = WEIGHTS / model['run']
    if model['tta_family'] == 'maskrcnn':
        return tta.MaskRCNNSource(folder, 'final', options)
    if model['tta_family'] == 'dense':
        return tta.DenseSource(folder, options)
    return tta.YoloSource(folder, 'final', options)

counts, started = {}, time.time()
for model in manifest['models']:
    source = source_for(model)
    records = [record for view in manifest['views'] for record in source(image, view)]
    target = WORK / 'tta' / model['run'] / 'candidates'
    target.mkdir(parents=True, exist_ok=True)
    if hasattr(source, 'finish'):
        source.finish(target.parent, stem)
    (target / f'{stem}.json').write_text(json.dumps(records))
    counts[(model['fold'], model['family'])] = len(records)
    del source
    torch.cuda.empty_cache()
print(f'{stem}: TTA for 25 models in {time.time() - started:.0f} s')
pd.Series(counts).unstack()"""),
    ("markdown", """## Stage 2: instance fusion within each fold (level 1)

Within a fold, candidates from the 40 sources (5 families × 8 views) are clustered greedily by mask IoU ≥ 0.5,
with at most one member per source. A fused mask keeps pixels drawn by at least half of its members. Its score
is the mean member score over all sources, so a source that missed the instance contributes zero. Pixels are
then owned in score order, and the instance features for the rescorer are computed."""),
    ("code", """setting = tuple(manifest['level1_setting'][key] for key in ('cluster_iou', 'pixel_vote', 'minimum_area', 'maximum_detections'))
level1 = {}
for fold in sorted(models.fold.unique()):
    directories = [WORK / 'tta' / run / 'candidates' for run in models[models.fold == fold].run]
    _, rows = ensemble.fold_instances((directories, stem, setting, manifest['views']))
    level1[fold] = pd.DataFrame(rows)
{fold: len(frame) for fold, frame in level1.items()}"""),
    ("markdown", """## Stage 3: crop refinement and adoption

Each fused instance is cropped from the native image with context, and the five refiner fold models, averaged,
redraw its mask from the image and the fused mask. The refined masks then replace the fused ones and are
re-owned in fused order (instances below 150 px are dropped). The `matched` figure it prints is the share of
annotators who drew each instance, which is 0 here because test images carry no labels."""),
    ("code", """refiners, size = refine.load_models(WEIGHTS / 'refiner', 'cuda')
tables = WORK / 'level1'
tables.mkdir(parents=True, exist_ok=True)
for fold, frame in level1.items():
    inputs, _, _, boxes = refine.extract(frame, size, workers=1, with_targets=False)
    probabilities = refine.predict_crops(refiners, inputs, 'cuda', 64)
    refine.add_refined_columns(frame, probabilities, None, boxes)
    frame.to_csv(tables / f'level1-f{fold}.csv', index=False)
refine.adopt(Namespace(source=tables, pattern='level1-f*.csv', column=manifest['refiner']['column'],
                       minimum_area=manifest['refiner']['minimum_area'], output=WORK / 'adopted', workers=1))
adopted = {fold: pd.read_csv(WORK / 'adopted' / f'level1-f{fold}.csv') for fold in level1}"""),
    ("markdown", """## Stage 4: rescoring and fusion across folds (level 2)

The rescorer (gradient-boosted trees fitted on out-of-fold instances of all five folds) gives each refined instance
the probability that the annotators drew it. The five fold ensembles are fused once more: an instance missed by a
fold contributes zero. Pixels are owned in probability order, and instances below the out-of-fold threshold
are dropped."""),
    ("code", """saved = pickle.loads((WEIGHTS / 'rescorer-hgb.pkl').read_bytes())
rows = []
for fold, frame in adopted.items():
    frame['probability'] = saved['rescorer'].predict(frame)
    frame['fold'] = fold
    rows += frame[['stem', 'fold', 'probability', 'rle']].to_dict('records')
_, masks, owned = ensemble.combine((stem, rows, len(adopted), setting[0], setting[1], setting[2], manifest['threshold']))
validation = validate_predictions({stem: masks}, {stem})
print(f'{owned} owned instances, {len(masks)} above threshold {manifest["threshold"]}')
validation"""),
    ("code", """overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
colours = plt.get_cmap('tab20')
for number, rle in enumerate(masks):
    mask = mask_utils.decode(rle).astype(bool)
    overlay[mask] = (0.45 * overlay[mask] + 0.55 * np.array(colours(number % 20)[:3]) * 255).astype(np.uint8)
figure, axes = plt.subplots(1, 2, figsize=(14, 7))
axes[0].imshow(image, cmap='gray'); axes[0].set_title(f'{stem} (supplied test image)')
axes[1].imshow(overlay); axes[1].set_title(f'{len(masks)} predicted filaments')
for axis in axes:
    axis.axis('off')
plt.tight_layout()"""),
    ("markdown", """## Check against the submitted file

The release includes `submission.csv`, the file Kaggle scored. A different GPU can move a few mask pixels through
mixed-precision arithmetic, so the masks for this image are matched at IoU > 0.5 rather than compared byte for
byte."""),
    ("code", """from solar_filament.metrics import overlap_matrix
from solar_filament.submission import load_predictions

submitted = load_predictions(WEIGHTS / 'submission.csv').get(stem, [])
best = overlap_matrix(submitted, masks).max(axis=1) if submitted and masks else np.zeros(len(submitted))
matched = best > 0.5
print(f'submitted masks: {len(submitted)} | reproduced: {len(masks)} | matched at IoU > 0.5: {int(matched.sum())}')
print(f'median IoU of matched masks: {np.median(best[matched]) if matched.any() else 0:.4f}')"""),
    ("markdown", """## The full test set

The same stages over all 180 test images rebuild the submitted CSV. The script verifies every released file
against `manifest.json`, resumes stage by stage, and compares the result with the released `submission.csv`:

```
python infra/reproduce-submission.py --weights $SOLAR_STORAGE_ROOT/weights/final \\
    --output $SOLAR_STORAGE_ROOT/runs/reproduction
```

On one L40S GPU, the 25 TTA passes take about 50 minutes, and fusion, refinement and level 2 about 20 minutes more
on 8 CPU cores. Training commands for every model are in `configs/` and the README. Out-of-fold PQ for each stage is
in the README and the report."""),
    ("code", """RUN_FULL_REPRODUCTION = False   # about 70 minutes on one L40S
if RUN_FULL_REPRODUCTION:
    import subprocess
    subprocess.run([sys.executable, 'infra/reproduce-submission.py', '--weights', str(WEIGHTS),
                    '--output', str(root / 'runs' / 'reproduction')], check=True)"""),
]


def main():
    cells = []
    for number, (kind, text) in enumerate(CELLS):
        cell = {"cell_type": kind, "id": f"solar-{number:03d}", "metadata": {}, "source": text.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}}}
    OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
