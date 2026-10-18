"""Deterministic physical-image coverage independent of annotation record counts."""

from __future__ import annotations

import numpy as np


def coverage_index(image_count: int, presentation: int, seed: int) -> int:
    if image_count < 1 or presentation < 0:
        raise ValueError("Physical-image count must be positive and presentation nonnegative")
    cycle, offset = divmod(presentation, image_count)
    generator = np.random.default_rng(np.random.SeedSequence([seed, cycle, 9013]))
    return int(generator.permutation(image_count)[offset])
