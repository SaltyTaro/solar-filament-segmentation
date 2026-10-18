import numpy as np
import pytest

from solar_filament.metrics import PQCounts, one_to_one_counts, reference_counts


def test_threshold_is_strict_and_bad_detections_are_penalized():
    result = reference_counts(np.array([[0.5, 0.0], [0.0, 0.75]])).scores()
    assert result == {"tp": 1, "fp": 1, "fn": 1, "sum_iou": 0.75, "pq": 0.375, "sq": 0.75, "rq": 0.5}


@pytest.mark.parametrize("shape,expected", [((2, 0), (0, 2)), ((0, 3), (3, 0)), ((0, 0), (0, 0))])
def test_missing_instances_are_counted_even_when_one_side_is_empty(shape, expected):
    result = reference_counts(np.zeros(shape))
    assert (result.fp, result.fn) == expected
    assert result.scores()["pq"] == 0


def test_pooled_counts_do_not_average_observation_scores():
    easy = reference_counts(np.eye(1))
    hard = reference_counts(np.zeros((9, 0)))
    pooled = (easy + hard).scores()["pq"]
    assert pooled == pytest.approx(1 / 5.5)
    assert pooled != (easy.scores()["pq"] + hard.scores()["pq"]) / 2


def test_reference_and_one_to_one_diverge_on_overlapping_annotation_matches():
    matrix = np.array([[0.7], [0.6]])
    assert reference_counts(matrix).tp == 2
    conventional = one_to_one_counts(matrix)
    assert conventional == PQCounts(tp=1, fp=0, fn=1, sum_iou=0.7)


def test_one_to_one_matching_prefers_two_valid_pairs_over_one_large_iou():
    matrix = np.array([[0.99, 0.51], [0.51, 0.49]])
    result = one_to_one_counts(matrix)
    assert result.tp == 2
    assert result.sum_iou == pytest.approx(1.02)


def test_invalid_overlap_values_are_rejected():
    with pytest.raises(ValueError):
        reference_counts(np.array([[np.nan]]))
