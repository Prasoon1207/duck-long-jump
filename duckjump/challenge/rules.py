"""Official challenge rules: the budget, the evaluation seeds and the frozen env settings."""

import dataclasses

from duckjump.env.arena import ArenaConfig

STEP_BUDGET = 10_000_000          # environment steps of training per entry (self-reported)
EVAL_SEEDS = tuple(range(20))
MAX_CHECKPOINT_BYTES = 20 * 2**20

# LongJumpConfig fields every entry is evaluated with, whatever its checkpoint or defaults say.
# Everything else (reward, start_distance_range, fields a submission adds) is free.
FROZEN_ENV_FIELDS = {
    "motor_strength": 2.0,
    "action_scale": 0.5,
    "sim_dt": 0.002,
    "control_dt": 0.02,
    "max_time": 4.0,
    "settle_time": 0.3,
    "joint_noise": 0.03,
    "fall_height": 0.08,
    "fall_uprightness": 0.5,
}


def official_env_config(cfg):
    """cfg with the frozen fields forced to their official values (arena.py itself is hash-frozen)."""
    return dataclasses.replace(cfg, arena=ArenaConfig(), **FROZEN_ENV_FIELDS)
