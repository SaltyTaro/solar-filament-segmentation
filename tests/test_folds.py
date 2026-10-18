import numpy as np
import pandas as pd

from solar_filament.folds import balanced_date_folds, temporal_holdout


def synthetic_observations():
    rows = []
    for index, day in enumerate(pd.date_range("2021-01-01", periods=180, freq="3D")):
        for view in range(2 if index % 11 == 0 else 1):
            rows.append(
                {
                    "filename": f"{day:%Y%m%d}1200{view:02d}Ch.jpeg",
                    "capture_date": day.strftime("%Y%m%d"),
                    "capture_time": day.isoformat(),
                    "year": day.year,
                    "site": ["Ch", "Bh", "Mh"][index % 3],
                    "annotator_records": index % 3 + 1,
                    "mean_filament_count": index % 7 + 2,
                }
            )
    return pd.DataFrame(rows)


def test_physical_dates_remain_together_and_splits_are_reproducible():
    table = synthetic_observations()
    first = balanced_date_folds(table, n_folds=5, seed=11)
    second = balanced_date_folds(table, n_folds=5, seed=11)
    assert np.array_equal(first["fold"], second["fold"])
    assert first.groupby("capture_date")["fold"].nunique().max() == 1
    assert set(first["fold"]) == set(range(5))
    counts = first["fold"].value_counts()
    assert counts.max() - counts.min() <= 5


def test_temporal_stress_training_respects_the_exclusion_buffer():
    result = temporal_holdout(synthetic_observations(), seed=11, buffer_days=7)
    assert set(result["split"]) == {"train", "validation", "excluded"}
    assert (result.loc[result["split"] == "train", "nearest_validation_days"] > 7).all()
    assert result.groupby("capture_date")["split"].nunique().max() == 1
