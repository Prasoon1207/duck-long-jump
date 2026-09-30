"""Training loop: runs parallel environments, feeds PPO, logs progress, saves checkpoints."""

import csv
import time
from collections import Counter, deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from duckjump.algo.config import PPOConfig
from duckjump.algo.ppo import PPO
from duckjump.env.longjump_env import LongJumpConfig, LongJumpEnv
from duckjump.infra.checkpoint import save_checkpoint, save_configs


def make_vector_env(env_cfg: LongJumpConfig, num_envs: int, seed: int) -> gym.vector.VectorEnv:
    envs = gym.vector.AsyncVectorEnv(
        [lambda: LongJumpEnv(env_cfg) for _ in range(num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    envs.reset(seed=seed)
    return envs


class EpisodeTracker:
    """Rolling statistics over the most recently finished episodes."""

    def __init__(self, window: int = 200):
        self.returns = deque(maxlen=window)
        self.infos = deque(maxlen=window)
        self.best_jump = 0.0

    def add(self, ret: float, info: dict) -> None:
        self.returns.append(ret)
        self.infos.append(info)
        self.best_jump = max(self.best_jump, info["jump_distance"])

    def summary(self) -> dict:
        if not self.infos:
            return {}
        outcomes = Counter(i["outcome"] for i in self.infos)
        n = len(self.infos)
        jumps = [i["jump_distance"] for i in self.infos]
        out = {
            "ep_return": float(np.mean(self.returns)),
            "ep_length": float(np.mean([i["length"] for i in self.infos])),
            "jump_mean": float(np.mean(jumps)),
            "jump_max_recent": float(np.max(jumps)),
            "jump_best_ever": self.best_jump,
            "flight_time": float(np.mean([i["flight_time"] for i in self.infos])),
        }
        for o in ["jump", "stepped_over", "fell", "fell_after_landing", "timeout"]:
            out[f"frac_{o}"] = outcomes[o] / n
        for k in self.infos[0]:
            if k.startswith("reward/"):
                out[k] = float(np.mean([i[k] for i in self.infos]))
        return out


def train(env_cfg: LongJumpConfig, ppo_cfg: PPOConfig, run_dir: Path, log_every: int = 1) -> PPO:
    torch.set_num_threads(1)  # leave the CPU cores to the simulation workers
    run_dir.mkdir(parents=True, exist_ok=True)
    save_configs(run_dir, env_cfg, ppo_cfg)

    envs = make_vector_env(env_cfg, ppo_cfg.num_envs, ppo_cfg.seed)
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]
    agent = PPO(obs_dim, act_dim, ppo_cfg)

    tracker = EpisodeTracker()
    running_return = np.zeros(ppo_cfg.num_envs)
    obs, _ = envs.reset(seed=ppo_cfg.seed)
    num_updates = ppo_cfg.total_steps // ppo_cfg.batch_size
    best_score = -np.inf
    log_file = open(run_dir / "progress.csv", "w", newline="")
    writer = None
    start = time.time()

    try:
        for update in range(1, num_updates + 1):
            agent.set_progress((update - 1) / num_updates)

            for _ in range(ppo_cfg.rollout_steps):
                action, log_prob, value, norm_obs = agent.act(obs)
                next_obs, reward, terminated, truncated, info = envs.step(action.numpy())
                done = terminated | truncated
                running_return += reward

                if done.any():
                    final_obs = np.stack([o if d else np.zeros(obs_dim, np.float32)
                                          for o, d in zip(info["final_obs"], done)])
                    reward = agent.bootstrap_truncated(reward, truncated, final_obs)
                    for i in np.flatnonzero(done):
                        ep = {k: v[i] for k, v in info["final_info"]["episode"].items() if not k.startswith("_")}
                        tracker.add(running_return[i], ep)
                        running_return[i] = 0.0

                agent.store(norm_obs, action, log_prob, value, reward, done)
                obs = next_obs

            stats = agent.update(obs)
            steps = update * ppo_cfg.batch_size
            row = {"update": update, "steps": steps, "time": time.time() - start,
                   "fps": steps / (time.time() - start), **tracker.summary(), **stats}

            if writer is None and "jump_mean" in row:
                writer = csv.DictWriter(log_file, fieldnames=list(row.keys()))
                writer.writeheader()
            if writer is not None:
                writer.writerow(row)
                log_file.flush()

            if update % log_every == 0 and "jump_mean" in row:
                print(f"[{update:4d}/{num_updates}] steps {steps:>9,d} | fps {row['fps']:5.0f} | "
                      f"return {row['ep_return']:7.2f} | jump mean {row['jump_mean'] * 100:5.1f} cm "
                      f"best {row['jump_best_ever'] * 100:5.1f} cm | jumps {row['frac_jump']:4.0%} "
                      f"fell {row['frac_fell']:4.0%} step-over {row['frac_stepped_over']:4.0%} | "
                      f"std {stats['action_std']:.2f} kl {stats['approx_kl']:.3f}", flush=True)

            extra = {"steps": steps, "stats": row}
            save_checkpoint(run_dir / "latest.pt", agent, env_cfg, obs_dim, act_dim, extra)
            score = row.get("jump_mean", -np.inf)
            if score > best_score and len(tracker.infos) >= 50:
                best_score = score
                save_checkpoint(run_dir / "best.pt", agent, env_cfg, obs_dim, act_dim, extra)
    finally:
        log_file.close()
        envs.close()
    return agent
