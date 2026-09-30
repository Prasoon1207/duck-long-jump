"""Save/load a trained agent together with the configs needed to rebuild it."""

import dataclasses
import json
from pathlib import Path

import torch

from duckjump.algo.config import PPOConfig
from duckjump.algo.ppo import PPO
from duckjump.env.arena import ArenaConfig
from duckjump.env.longjump_env import LongJumpConfig
from duckjump.env.reward import RewardWeights


def env_config_from_dict(d: dict) -> LongJumpConfig:
    d = dict(d)
    d["arena"] = ArenaConfig(**d["arena"])
    d["reward"] = RewardWeights(**d["reward"])
    return LongJumpConfig(**d)


def ppo_config_from_dict(d: dict) -> PPOConfig:
    d = dict(d)
    d["actor_hidden"], d["critic_hidden"] = tuple(d["actor_hidden"]), tuple(d["critic_hidden"])
    return PPOConfig(**d)


def save_configs(run_dir: Path, env_cfg: LongJumpConfig, ppo_cfg: PPOConfig) -> None:
    cfg = {"env": dataclasses.asdict(env_cfg), "ppo": dataclasses.asdict(ppo_cfg)}
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))


def save_checkpoint(path: Path, agent: PPO, env_cfg: LongJumpConfig, obs_dim: int, act_dim: int, extra: dict) -> None:
    torch.save({
        "agent": agent.state_dict(),
        "env_cfg": dataclasses.asdict(env_cfg),
        "ppo_cfg": dataclasses.asdict(agent.cfg),
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        **extra,
    }, path)


def load_checkpoint(path: str | Path) -> tuple[PPO, LongJumpConfig, dict]:
    path = Path(path)
    if path.is_dir():
        path = path / "best.pt" if (path / "best.pt").exists() else path / "latest.pt"
    ckpt = torch.load(path, weights_only=False)
    agent = PPO(ckpt["obs_dim"], ckpt["act_dim"], ppo_config_from_dict(ckpt["ppo_cfg"]))
    agent.load_state_dict(ckpt["agent"])
    return agent, env_config_from_dict(ckpt["env_cfg"]), ckpt
