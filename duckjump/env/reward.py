"""Long-jump reward. Tweak the weights or rewrite `compute_reward` freely.

The environment calls `compute_reward(step, weights, dt)` once per control step
with a `StepInfo` describing what just happened.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class RewardWeights:
    runup_velocity: float = 1.0      # per (m/s) per second, only before the take-off line
    max_runup_velocity: float = 1.0  # forward speed above this earns nothing extra
    airtime: float = 2.0             # per second with both feet off the ground
    jump_distance: float = 100.0     # per metre jumped, paid on landing
    fall: float = -2.0               # falling before landing
    fall_after_landing: float = 0.5  # fraction of the jump reward taken back if it falls after landing
    upright: float = 0.1             # per second, scaled by how upright the trunk is
    action_rate: float = -0.02       # per second, times squared change in action


@dataclass
class StepInfo:
    phase: str               # "runup" (includes the flight) or "landed"
    forward_velocity: float  # world-frame x velocity of the base (m/s)
    airborne_time: float     # seconds of this step with no foot touching the ground
    uprightness: float       # 1 upright, 0 lying flat
    action_delta: np.ndarray
    just_landed: bool
    jump_distance: float     # valid when just_landed
    fell: bool


def compute_reward(s: StepInfo, w: RewardWeights, dt: float) -> tuple[float, dict]:
    terms = {
        "runup_velocity": 0.0,
        "airtime": 0.0,
        "jump_distance": 0.0,
        "fall": 0.0,
        "upright": w.upright * s.uprightness * dt,
        "action_rate": w.action_rate * float(np.sum(s.action_delta**2)) * dt,
    }
    if s.phase == "runup":
        terms["runup_velocity"] = w.runup_velocity * np.clip(s.forward_velocity, -w.max_runup_velocity, w.max_runup_velocity) * dt
        terms["airtime"] = w.airtime * s.airborne_time
    if s.just_landed:
        terms["jump_distance"] = w.jump_distance * s.jump_distance
    if s.fell:
        if s.phase == "landed" or s.just_landed:
            terms["fall"] = -w.fall_after_landing * w.jump_distance * s.jump_distance
        else:
            terms["fall"] = w.fall
    return float(sum(terms.values())), terms
