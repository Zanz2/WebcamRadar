"""Optional per-pixel noise threshold (off unless grid.noise_sigma > 0).

A single motion_threshold treats every pixel alike, but camera noise is several times
stronger on sharp edges than on flat areas, so motionless edges keep casting rays.
This learns each pixel's normal frame-to-frame change while nothing moves, then keeps
only changes above max(floor, k * sigma) for that pixel.

The thresholds are frozen after learning: a model that kept adapting would learn
something that moves in one place all the time (a playing screen) as noise.
"""

from __future__ import annotations

import numpy as np

MAX_SAMPLES = 90  # frames kept per camera while learning (bounds memory: ~83 MB at 1280x720)


class NoiseModel:
    def __init__(self, k: float, floor: float, learn_seconds: float):
        self.k, self.floor, self.learn_seconds = k, floor, learn_seconds
        self.threshold: np.ndarray | None = None  # per pixel, once learned
        self._t0: float | None = None
        self._last = -np.inf
        self._samples: list[np.ndarray] = []

    @property
    def ready(self) -> bool:
        return self.threshold is not None

    def learn(self, diff: np.ndarray, timestamp: float) -> bool:
        """Feed one |frame difference| of a static scene; True once learning is complete."""
        if self._t0 is None:
            self._t0 = timestamp
        if timestamp - self._last >= self.learn_seconds / MAX_SAMPLES:
            self._samples.append(np.clip(diff, 0, 255).astype(np.uint8))  # grey diffs are whole numbers
            self._last = timestamp
        if timestamp - self._t0 >= self.learn_seconds and len(self._samples) >= 10:
            # Median, not standard deviation: a brief real movement must not raise the bar.
            sigma = 1.4826 * np.median(np.stack(self._samples), axis=0)
            self.threshold = np.maximum(self.floor, self.k * sigma).astype(np.float32)
            self._samples = []
        return self.ready

    def apply(self, diff: np.ndarray) -> np.ndarray:
        """`diff` with everything that is within the pixel's normal noise set to 0."""
        return np.where(diff > self.threshold, diff, np.float32(0.0))
