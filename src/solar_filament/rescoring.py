"""Learned rescoring of fused instances from fusion features only."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EXCLUDED = {"stem", "column", "annotators", "matched_fraction", "best_iou_mean", "pairs", "parsed", "rle", "fold", "families"}
FUSION = ("score", "max_score", "votes", "members", "pixel_agreement")
MONOTONE = ("score", "max_score", "votes")


def feature_columns(frame: pd.DataFrame, kind: str = "hgb") -> list[str]:
    numeric = [name for name in frame.columns if name not in EXCLUDED and pd.api.types.is_numeric_dtype(frame[name])]
    if kind == "hgb":
        return numeric
    return [name for name in numeric if name in FUSION or name.split("_m")[0] in ("votes", "score", "max_score")]


class Rescorer:
    """Predicts the matched-annotator fraction; ``logistic`` fits the binary majority label."""

    def __init__(self, kind: str, columns: list[str], seed: int = 0):
        self.kind, self.columns = kind, columns
        if kind == "logistic":
            self.model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        elif kind == "hgb-monotone":
            constraints = [1 if name.split("_m")[0] in MONOTONE else 0 for name in columns]
            self.model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=7,
                                                       min_samples_leaf=40, monotonic_cst=constraints, random_state=seed)
        else:
            self.model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.03, max_leaf_nodes=15,
                                                       min_samples_leaf=20, l2_regularization=1.0, random_state=seed)

    def fit(self, frame: pd.DataFrame):
        target = frame.matched_fraction >= 0.5 if self.kind == "logistic" else frame.matched_fraction
        self.model.fit(frame[self.columns].fillna(0), target, **{
            ("logisticregression__sample_weight" if self.kind == "logistic" else "sample_weight"): frame.annotators})
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        inputs = frame[self.columns].fillna(0)
        return self.model.predict_proba(inputs)[:, 1] if self.kind == "logistic" else self.model.predict(inputs)


