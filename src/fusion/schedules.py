"""Epoch schedules for the fusion curriculum.

The schedule returns lambda_t in [0,1]: how much the *learned* gate is trusted at
epoch t, with the remainder falling back to a fixed prior. Early training therefore
fuses with a stable constant and hands control to the data-driven gate as it becomes
reliable.

This is deliberately not the same thing as annealing a model out of the ensemble. The
supervisor's illustrative form, w1 = max(0.8, t/T) with hm = w1*m1 + (1-w1)*m2, drives
the second model's contribution to exactly zero at the final epoch, so the "hybrid"
degenerates into model 1 at precisely the point it gets evaluated, and model 1's weight
never drops below 0.8 regardless of which model is actually better. Applying the same
mechanism to gate *trust* keeps both backbones contributing at convergence while
preserving the idea it was meant to express. `supervisor` below reproduces the original
curve so the comparison is empirical rather than rhetorical.
"""
from __future__ import annotations

import math
from typing import Callable

Schedule = Callable[[int, int], float]


def constant(epoch: int, total_epochs: int) -> float:
    """No curriculum: trust the learned gate from the first step."""
    return 1.0


def linear(epoch: int, total_epochs: int) -> float:
    if total_epochs <= 1:
        return 1.0
    return min(1.0, (epoch - 1) / (total_epochs - 1))


def cosine(epoch: int, total_epochs: int) -> float:
    if total_epochs <= 1:
        return 1.0
    progress = min(1.0, (epoch - 1) / (total_epochs - 1))
    return 0.5 * (1 - math.cos(math.pi * progress))


def step(epoch: int, total_epochs: int, switch_at: float = 0.5) -> float:
    """Prior only, then learned gate only."""
    return 0.0 if (epoch - 1) < switch_at * total_epochs else 1.0


def supervisor(epoch: int, total_epochs: int) -> float:
    """The originally suggested curve, max(0.8, t/T), kept for direct comparison."""
    if total_epochs <= 0:
        return 1.0
    return max(0.8, epoch / total_epochs)


SCHEDULES: dict[str, Schedule] = {
    "constant": constant,
    "linear": linear,
    "cosine": cosine,
    "step": step,
    "supervisor": supervisor,
}


def get_schedule(name: str) -> Schedule:
    if name not in SCHEDULES:
        raise ValueError(f"Unknown schedule {name!r}; expected one of {sorted(SCHEDULES)}")
    return SCHEDULES[name]
