"""Organizer-compatible pooled PQ and complementary one-to-one diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment


@dataclass
class PQCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    sum_iou: float = 0.0

    def __add__(self, other: PQCounts) -> PQCounts:
        return PQCounts(
            self.tp + other.tp, self.fp + other.fp, self.fn + other.fn, self.sum_iou + other.sum_iou
        )

    def scores(self) -> dict:
        denominator = self.tp + 0.5 * (self.fp + self.fn)
        return {
            **asdict(self),
            "pq": self.sum_iou / denominator if denominator else 0.0,
            "sq": self.sum_iou / self.tp if self.tp else 0.0,
            "rq": self.tp / denominator if denominator else 0.0,
        }


def overlap_matrix(gt_rles: list[dict], pred_rles: list[dict]) -> np.ndarray:
    if not gt_rles or not pred_rles:
        return np.zeros((len(gt_rles), len(pred_rles)), dtype=np.float64)
    return np.asarray(mask_utils.iou(gt_rles, pred_rles, [0] * len(pred_rles)), dtype=np.float64)


def reference_counts(iou: np.ndarray, threshold: float = 0.5) -> PQCounts:
    """Count every qualifying pair, exactly as the organizer reference notebook."""
    if iou.ndim != 2 or not np.isfinite(iou).all() or np.any((iou < 0) | (iou > 1)):
        raise ValueError("IoU matrix must contain finite values in [0, 1]")
    hits = iou > threshold
    return PQCounts(
        tp=int(hits.sum()),
        fp=int((~hits.any(axis=0)).sum()),
        fn=int((~hits.any(axis=1)).sum()),
        sum_iou=float(iou[hits].sum()),
    )


def one_to_one_counts(iou: np.ndarray, threshold: float = 0.5) -> PQCounts:
    """Maximum-cardinality matching, then maximize matched IoU."""
    n_gt, n_pred = iou.shape
    if not n_gt or not n_pred:
        return PQCounts(fp=n_pred, fn=n_gt)
    valid = iou > threshold
    # Each qualifying pair gets a bonus exceeding any possible IoU sum change.
    value = valid * (min(n_gt, n_pred) + 1 + iou)
    rows, columns = linear_sum_assignment(value, maximize=True)
    matched = valid[rows, columns]
    tp = int(matched.sum())
    return PQCounts(
        tp=tp, fp=n_pred - tp, fn=n_gt - tp, sum_iou=float(iou[rows[matched], columns[matched]].sum())
    )


def overlap_diagnostics(iou: np.ndarray) -> dict:
    positive = iou > 0
    gt_degree = positive.sum(axis=1)
    pred_degree = positive.sum(axis=0)
    nonzero = iou[positive]
    return {
        "gt_overlapping_multiple_predictions": int((gt_degree > 1).sum()),
        "predictions_overlapping_multiple_gt": int((pred_degree > 1).sum()),
        "gt_with_no_overlap": int((gt_degree == 0).sum()),
        "predictions_with_no_overlap": int((pred_degree == 0).sum()),
        "positive_pair_ious": nonzero.tolist(),
        "positive_pair_dice": (2 * nonzero / (1 + nonzero)).tolist(),
        "gt_overlap_degrees": gt_degree.tolist(),
        "prediction_overlap_degrees": pred_degree.tolist(),
    }
