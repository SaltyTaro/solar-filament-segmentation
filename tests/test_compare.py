import pandas as pd
import pytest

from solar_filament.compare import paired_pq_comparison


def sample_records():
    return pd.DataFrame(
        [
            {"filename": "20110101A.jpeg", "annotator_image": "a", "tp": 1, "fp": 1, "fn": 1, "sum_iou": 0.8},
            {"filename": "20110101A.jpeg", "annotator_image": "b", "tp": 1, "fp": 0, "fn": 0, "sum_iou": 0.6},
            {"filename": "20110102B.jpeg", "annotator_image": "c", "tp": 1, "fp": 0, "fn": 0, "sum_iou": 0.7},
        ]
    )


def test_paired_comparison_handles_reordered_records_without_artificial_uncertainty():
    frame = sample_records()
    result = paired_pq_comparison(frame, frame.iloc[::-1], replicates=100)
    assert result["delta"]["pq"] == 0
    assert result["paired_date_bootstrap"]["delta_pq_lower_95"] == 0
    assert result["paired_date_bootstrap"]["delta_pq_upper_95"] == 0
    assert result["paired_date_bootstrap"]["date_groups"] == 2


def test_paired_comparison_retains_ground_truth_and_detects_consistent_improvement():
    baseline = sample_records()
    candidate = baseline.copy()
    candidate["sum_iou"] += 0.1
    result = paired_pq_comparison(baseline, candidate, replicates=100)
    assert result["delta"]["pq"] == pytest.approx(0.075)
    assert result["paired_date_bootstrap"]["delta_pq_lower_95"] > 0
    with pytest.raises(ValueError, match="exactly the same"):
        paired_pq_comparison(baseline, candidate.iloc[:2])
    candidate.loc[0, "fn"] += 1
    with pytest.raises(ValueError, match="ground-truth"):
        paired_pq_comparison(baseline, candidate)
