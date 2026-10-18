import pandas as pd
import pytest

from solar_filament.report import bootstrap_pq, pooled_scores


def test_pooled_report_retains_per_annotator_counts():
    records = pd.DataFrame(
        [
            {"filename": "20200101120000Bh.jpeg", "tp": 1, "fp": 0, "fn": 0, "sum_iou": 1.0},
            {"filename": "20200101120000Bh.jpeg", "tp": 0, "fp": 1, "fn": 1, "sum_iou": 0.0},
        ]
    )
    scores = pooled_scores(records)
    assert scores["pq"] == pytest.approx(0.5)
    assert scores["physical_images"] == 1
    assert scores["annotator_records"] == 2


def test_bootstrap_keeps_same_date_observations_and_annotators_together():
    records = pd.DataFrame(
        [
            {"filename": "20200101120000Bh.jpeg", "tp": 1, "fp": 0, "fn": 0, "sum_iou": 1.0},
            {"filename": "20200101120100Ch.jpeg", "tp": 0, "fp": 0, "fn": 9, "sum_iou": 0.0},
        ]
    )
    result = bootstrap_pq(records, replicates=100)
    assert result["date_groups"] == 1
    assert result["lower_95"] == pytest.approx(1 / 5.5)
    assert result["upper_95"] == result["lower_95"]
