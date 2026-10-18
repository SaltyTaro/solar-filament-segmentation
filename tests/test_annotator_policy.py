import numpy as np

from solar_filament.annotator_policy import agreement_medoid, agreement_medoids


def test_medoid_selects_record_most_consistent_with_other_independent_drawings():
    agreement = np.array([[1.0, 0.8, 0.3], [0.8, 1.0, 0.7], [0.3, 0.7, 1.0]])
    assert agreement_medoid(["record-c", "record-b", "record-a"], agreement) == "record-b"


def test_two_annotator_tie_is_stable_without_using_validation_performance():
    assert agreement_medoid(["record-c", "record-a"], np.array([[1.0, 0.7], [0.7, 1.0]])) == "record-a"


def test_tied_policy_retains_both_two_annotator_records():
    assert agreement_medoids(["record-c", "record-a"], np.array([[1.0, 0.7], [0.7, 1.0]])) == [
        "record-a", "record-c"
    ]


def test_tied_policy_retains_only_the_most_central_three_record_candidates():
    agreement = np.array([[1.0, 0.8, 0.3], [0.8, 1.0, 0.3], [0.3, 0.3, 1.0]])
    assert agreement_medoids(["record-c", "record-b", "record-a"], agreement) == ["record-b", "record-c"]
    unique = np.array([[1.0, 0.8, 0.3], [0.8, 1.0, 0.7], [0.3, 0.7, 1.0]])
    assert agreement_medoids(["record-c", "record-b", "record-a"], unique) == ["record-b"]


def test_tied_policy_retains_all_fully_equal_three_record_drawings():
    agreement = np.full((3, 3), 0.7)
    np.fill_diagonal(agreement, 1.0)
    assert agreement_medoids(["record-c", "record-b", "record-a"], agreement) == [
        "record-a", "record-b", "record-c"
    ]
