"""Deterministic date-grouped folds and a buffered temporal stress holdout."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from solar_filament.coco import CocoIndex
from solar_filament.ingest import file_sha256
from solar_filament.storage import annotations_path


def observation_table(index: CocoIndex) -> pd.DataFrame:
    rows = []
    for filename, records in sorted(index.images_by_filename.items()):
        stem = Path(filename).stem
        counts = [len(index.annotations_by_image[record["id"]]) for record in records]
        areas = [
            annotation["area"]
            for record in records
            for annotation in index.annotations_by_image[record["id"]]
        ]
        rows.append(
            {
                "filename": filename,
                "image_stem": stem,
                "capture_time": datetime.strptime(stem[:14], "%Y%m%d%H%M%S").isoformat(),
                "capture_date": stem[:8],
                "year": int(stem[:4]),
                "site": stem[-2:],
                "annotator_records": len(records),
                "mean_filament_count": float(np.mean(counts)),
                "annotation_count": sum(counts),
                "median_annotation_area": float(np.median(areas)) if areas else 0.0,
            }
        )
    return pd.DataFrame(rows)


def balanced_date_folds(table: pd.DataFrame, n_folds: int = 5, seed: int = 20261003) -> pd.DataFrame:
    """Balance a continuous feature vector while keeping UTC dates intact."""
    if n_folds < 2 or len(table) < n_folds:
        raise ValueError("Not enough observations for the requested number of folds")
    metadata = pd.get_dummies(table[["site", "year", "annotator_records"]].astype(str), dtype=float)
    features = pd.concat(
        [
            pd.Series(1.0, index=table.index, name="observations"),
            table["mean_filament_count"].rename("filaments"),
            metadata,
        ],
        axis=1,
    )
    features["capture_date"] = table["capture_date"]
    grouped = features.groupby("capture_date", sort=True).sum()
    if len(grouped) < n_folds:
        raise ValueError("Not enough independent capture dates for the requested folds")
    values = grouped.to_numpy(dtype=float)
    target = values.sum(axis=0) / n_folds
    normalizer = np.maximum(target, 1.0)
    weights = np.ones(len(target), dtype=float)
    weights[:2] = [12.0, 4.0]
    best = None
    rng = np.random.default_rng(seed)
    # Seeded restarts help balance rare years without splitting physical groups.
    for _ in range(48):
        random_priority = rng.random(len(values))
        order = np.lexsort((random_priority, -values[:, 0]))
        sums = np.zeros((n_folds, values.shape[1]))
        assignment = np.empty(len(values), dtype=int)
        tie_order = rng.permutation(n_folds)
        for group_index in order:
            costs = []
            for fold in tie_order:
                updated = sums[fold] + values[group_index]
                change = ((updated - target) / normalizer) ** 2 - ((sums[fold] - target) / normalizer) ** 2
                costs.append(float((change * weights).sum()))
            fold = int(tie_order[int(np.argmin(costs))])
            assignment[group_index] = fold
            sums[fold] += values[group_index]
        cost = float(((((sums - target) / normalizer) ** 2) * weights).sum())
        if best is None or cost < best[0]:
            best = (cost, assignment.copy())
    date_to_fold = dict(zip(grouped.index, best[1].tolist()))
    result = table.copy()
    result["fold"] = result["capture_date"].map(date_to_fold).astype(int)
    if result.groupby("capture_date")["fold"].nunique().max() != 1:
        raise AssertionError("A capture date crossed folds")
    return result


def temporal_holdout(table: pd.DataFrame, seed: int = 20261003, buffer_days: int = 7) -> pd.DataFrame:
    """Hold out one populated month per year and exclude a seven-day buffer."""
    times = pd.to_datetime(table["capture_time"])
    months = times.dt.to_period("M").astype(str)
    rng = np.random.default_rng(seed)
    chosen = []
    for year in sorted(table["year"].unique()):
        available = sorted(months[table["year"] == year].unique())
        chosen.append(str(rng.choice(available)))
    validation = months.isin(chosen)
    val_dates = times[validation].dt.normalize().to_numpy(dtype="datetime64[D]")
    dates = times.dt.normalize().to_numpy(dtype="datetime64[D]")
    distance_days = (
        np.abs(dates[:, None] - val_dates[None, :]).astype("timedelta64[D]").astype(int).min(axis=1)
    )
    result = table[["filename", "capture_time", "capture_date", "year", "site"]].copy()
    result["split"] = np.where(
        validation, "validation", np.where(distance_days <= buffer_days, "excluded", "train")
    )
    result["nearest_validation_days"] = distance_days
    if (result.loc[result["split"] == "train", "nearest_validation_days"] <= buffer_days).any():
        raise AssertionError("Temporal exclusion buffer was violated")
    result.attrs["validation_months"] = chosen
    return result


def create_folds(root: Path, n_folds: int = 5, seed: int = 20261003) -> dict:
    annotation_file = annotations_path(root)
    index = CocoIndex(annotation_file)
    table = balanced_date_folds(observation_table(index), n_folds, seed)
    destination = root / "data" / "splits"
    destination.mkdir(parents=True, exist_ok=True)
    fold_path = destination / "folds.csv"
    signature = hashlib.sha256(
        "\n".join(f"{row.filename},{row.fold}" for row in table.itertuples()).encode()
    ).hexdigest()
    metadata_path = destination / "folds-metadata.json"
    if fold_path.exists():
        old = pd.read_csv(fold_path)
        old_signature = hashlib.sha256(
            "\n".join(f"{row.filename},{row.fold}" for row in old.itertuples()).encode()
        ).hexdigest()
        if old_signature != signature:
            raise ValueError("Existing frozen folds differ. Use a new versioned split directory.")
    table.to_csv(fold_path, index=False, lineterminator="\n")
    stress = temporal_holdout(table, seed)
    stress.to_csv(destination / "temporal-holdout.csv", index=False, lineterminator="\n")
    summaries = []
    for fold, group in table.groupby("fold"):
        summaries.append(
            {
                "fold": int(fold),
                "physical_images": len(group),
                "capture_dates": int(group["capture_date"].nunique()),
                "annotator_records": int(group["annotator_records"].sum()),
                "annotations": int(group["annotation_count"].sum()),
                "site_counts": group["site"].value_counts().sort_index().to_dict(),
                "year_counts": {
                    str(year): int(count) for year, count in group["year"].value_counts().sort_index().items()
                },
            }
        )
    metadata = {
        "seed": seed,
        "folds": n_folds,
        "grouping": "UTC capture date; all physical-image annotator records remain together",
        "annotation_sha256": file_sha256(annotation_file),
        "split_sha256": signature,
        "summary": summaries,
        "temporal_holdout": {
            "validation_months": stress.attrs["validation_months"],
            "buffer_days": 7,
            "counts": stress["split"].value_counts().to_dict(),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata
