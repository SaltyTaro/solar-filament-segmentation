"""Learning rates indexed by completed training steps for reliable resumption."""

from __future__ import annotations

import math


def validate_learning_rate_schedule(config: dict) -> None:
    policy = config.get("learning_rate_policy", "constant")
    if policy not in ("constant", "cosine"):
        raise ValueError(f"Unknown learning-rate policy: {policy}")
    base = config["learning_rate"]
    if isinstance(base, bool) or not math.isfinite(base) or base <= 0:
        raise ValueError("Learning rate must be finite and positive")
    if policy == "constant":
        return
    final = config.get("final_learning_rate")
    if not isinstance(final, (int, float)) or isinstance(final, bool):
        raise ValueError("Cosine learning rates require a numeric final learning rate")
    if not math.isfinite(final) or not 0 < final <= base:
        raise ValueError("Final learning rate must be positive and no larger than the initial rate")
    for field in ("epochs", "steps_per_epoch"):
        value = config[field]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{field} must be a positive integer")


def learning_rate_at_step(config: dict, absolute_step: int) -> float:
    """Return the rate for a zero-indexed step, including after checkpoint resume.

    Cosine decay reaches the configured floor on the last presentation. A
    one-presentation smoke uses the initial rate. No scheduler state or
    validation score participates in the schedule.
    """
    validate_learning_rate_schedule(config)
    base = config["learning_rate"]
    if config.get("learning_rate_policy", "constant") == "constant":
        return base
    total = config["epochs"] * config["steps_per_epoch"]
    if not isinstance(absolute_step, int) or isinstance(absolute_step, bool):
        raise ValueError("Absolute training step must be an integer")
    if not 0 <= absolute_step < total:
        raise ValueError("Absolute training step is outside the configured training horizon")
    if total == 1:
        return base
    floor = config["final_learning_rate"]
    return floor + 0.5 * (base - floor) * (1 + math.cos(math.pi * absolute_step / (total - 1)))
