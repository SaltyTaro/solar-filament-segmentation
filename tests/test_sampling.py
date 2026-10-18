from collections import Counter

from solar_filament.sampling import coverage_index


def test_physical_image_coverage_has_no_omissions_before_repeats():
    size = 37
    sequence = [coverage_index(size, presentation, seed=12) for presentation in range(size * 3)]
    for cycle in range(3):
        assert sorted(sequence[cycle * size : (cycle + 1) * size]) == list(range(size))
    assert set(Counter(sequence).values()) == {3}


def test_partial_pass_and_restart_reconstruct_the_same_image_schedule():
    size, seed = 566, 20261003
    uninterrupted = [coverage_index(size, presentation, seed) for presentation in range(570)]
    resumed = [coverage_index(size, presentation, seed) for presentation in range(320, 570)]
    assert uninterrupted[320:] == resumed
    assert len(set(uninterrupted[:566])) == 566
