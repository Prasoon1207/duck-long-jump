"""Plays the official attempts. Runs in a subprocess inside a submission workspace, so
`duckjump` here is the submission's code (frozen parts verified before this starts).

    python -m duckjump.challenge.rollout <checkpoint.pt> <result.json>
"""

import json
import sys

import numpy as np
import torch

from duckjump.challenge.rules import EVAL_SEEDS, STEP_BUDGET, official_env_config
from duckjump.env.longjump_env import LongJumpConfig, LongJumpEnv
from duckjump.infra.checkpoint import load_policy


def attempt(env: LongJumpEnv, agent, seed: int) -> dict:
    obs, _ = env.reset(seed=seed, options={"start_distance": env.cfg.arena.runup_length})
    while True:
        action = np.asarray(agent.act(obs[None], deterministic=True)[0][0], dtype=np.float64)
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            ep = info["episode"]
            jump = float(ep["jump_distance"]) if ep["outcome"] == "jump" else 0.0
            return {"seed": seed, "outcome": ep["outcome"], "jump_distance": jump}


def run(checkpoint: str) -> dict:
    torch.set_num_threads(1)
    agent, ckpt = load_policy(checkpoint)
    steps = ckpt.get("steps")
    if not isinstance(steps, int):
        return {"status": "rejected", "reason": "checkpoint does not record its training steps"}
    if steps > STEP_BUDGET:
        return {"status": "rejected", "steps": steps,
                "reason": f"trained for {steps:,} steps, over the {STEP_BUDGET:,} step budget"}

    env = LongJumpEnv(official_env_config(LongJumpConfig()))
    obs_dim = env.observation_space.shape[0]
    if ckpt.get("obs_dim", obs_dim) != obs_dim:
        return {"status": "rejected", "steps": steps,
                "reason": f"policy expects {ckpt['obs_dim']} observations, env produces {obs_dim}"}
    episodes = [attempt(env, agent, seed) for seed in EVAL_SEEDS]
    env.close()
    jumps = [e["jump_distance"] for e in episodes]
    return {
        "status": "ok",
        "score": float(np.mean(jumps)),
        "success_rate": float(np.mean([e["outcome"] == "jump" for e in episodes])),
        "best": float(np.max(jumps)),
        "steps": steps,
        "episodes": episodes,
    }


if __name__ == "__main__":
    result = run(sys.argv[1])
    with open(sys.argv[2], "w") as f:
        json.dump(result, f)
