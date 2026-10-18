"""Compare two fixed models using the same held-out dates and annotator records."""

from __future__ import annotations

import numpy as np
import pandas as pd

from solar_filament.report import pooled_scores


def paired_pq_comparison(
    baseline: pd.DataFrame, candidate: pd.DataFrame, replicates: int = 2000, seed: int = 20261003
) -> dict:
    keys = ["filename", "annotator_image"]
    for frame in (baseline, candidate):
        if frame.empty or frame.duplicated(keys).any():
            raise ValueError("Comparisons require nonempty, unique annotator-image records")
    baseline, candidate = (frame.set_index(keys).sort_index() for frame in (baseline, candidate))
    if not baseline.index.equals(candidate.index):
        raise ValueError("Models must cover exactly the same held-out images and annotator records")
    if not np.array_equal((baseline.tp + baseline.fn).to_numpy(), (candidate.tp + candidate.fn).to_numpy()):
        raise ValueError("Compared records must contain the same ground-truth objects")
    frames = [frame.reset_index() for frame in (baseline, candidate)]
    grouped = []
    for frame in frames:
        grouped.append(
            frame.assign(capture_date=frame.filename.str[:8])
            .groupby("capture_date")[["tp", "fp", "fn", "sum_iou"]]
            .sum()
            .to_numpy()
        )
    if replicates < 1:
        raise ValueError("At least one bootstrap replicate is required")
    rng = np.random.default_rng(seed)
    values = np.empty(replicates)
    for start in range(0, replicates, 200):
        number = min(200, replicates - start)
        draws = rng.integers(len(grouped[0]), size=(number, len(grouped[0])))
        scores = []
        for counts in grouped:
            sums = counts[draws].sum(axis=1)
            denominator = sums[:, 0] + 0.5 * (sums[:, 1] + sums[:, 2])
            scores.append(np.divide(sums[:, 3], denominator, out=np.zeros(number), where=denominator > 0))
        values[start : start + number] = scores[1] - scores[0]
    base_score, candidate_score = (pooled_scores(frame) for frame in frames)
    lower, median, upper = np.quantile(values, [0.025, 0.5, 0.975])
    return {
        "baseline": base_score,
        "candidate": candidate_score,
        "delta": {
            key: candidate_score[key] - base_score[key]
            for key in ("pq", "sq", "rq", "precision", "recall", "tp", "fp", "fn")
        },
        "paired_date_bootstrap": {
            "date_groups": len(grouped[0]),
            "replicates": replicates,
            "seed": seed,
            "delta_pq_lower_95": float(lower),
            "delta_pq_median": float(median),
            "delta_pq_upper_95": float(upper),
            "fraction_delta_positive": float(np.mean(values > 0)),
        },
        "interpretation": "Paired fixed-model uncertainty retains every annotator and uses shared date resamples. It does not correct development-fold model or parameter selection bias.",
    }
