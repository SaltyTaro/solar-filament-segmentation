"""Image-only detector preprocessing shared by training and every inference path."""

from __future__ import annotations

import math

import cv2
import numpy as np


class DetectorPreprocessing:
    def __init__(self, config: dict):
        self.kind = config.get("image_preprocessing", "raw")
        if self.kind not in ("raw", "clahe"):
            raise ValueError("Detector image preprocessing must be raw or clahe")
        self.transform = None
        if self.kind == "clahe":
            limit = float(config.get("clahe_clip_limit", 2.0))
            grid = config.get("clahe_grid_size", 8)
            if not math.isfinite(limit) or not 0 < limit <= 8:
                raise ValueError("CLAHE clip limit must be finite and in (0, 8]")
            if isinstance(grid, bool) or not isinstance(grid, int) or not 1 <= grid <= 32:
                raise ValueError("CLAHE grid size must be an integer in [1, 32]")
            self.transform = cv2.createCLAHE(clipLimit=limit, tileGridSize=(grid, grid))

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray) or image.ndim != 2 or image.dtype != np.uint8:
            raise ValueError("Detector preprocessing requires uint8 grayscale pixels")
        if self.transform is None:
            return image
        return self.transform.apply(image)
