"""Run a trained policy: evaluate it, record an MP4, or watch it in the MuJoCo viewer."""

import time
from pathlib import Path

import imageio
import numpy as np

from duckjump.algo.ppo import PPO
from duckjump.env.longjump_env import LongJumpConfig, LongJumpEnv


def run_episode(env: LongJumpEnv, agent: PPO | None, seed: int, deterministic: bool = True, on_step=None) -> dict:
    """Full run-up attempt. agent=None holds the home pose. on_step(env) runs after every control step."""
    obs, _ = env.reset(seed=seed, options={"start_distance": env.cfg.arena.runup_length})
    total = 0.0
    while True:
        if agent is None:
            action = np.zeros(env.action_space.shape)
        else:
            action = agent.act(obs[None], deterministic=deterministic)[0][0].numpy()
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        if on_step is not None:
            on_step(env)
        if terminated or truncated:
            return {**info["episode"], "return": total}


def evaluate(agent: PPO, env_cfg: LongJumpConfig, episodes: int = 10, seed: int = 1000) -> list[dict]:
    env = LongJumpEnv(env_cfg)
    results = [run_episode(env, agent, seed + i) for i in range(episodes)]
    env.close()
    return results


def record_video(agent: PPO | None, env_cfg: LongJumpConfig, path: Path, episodes: int = 3,
                 camera: str = "follow", seed: int = 1000, hold_s: float = 0.6) -> list[dict]:
    env = LongJumpEnv(env_cfg)
    fps = round(1.0 / env_cfg.control_dt)
    frames, results = [], []
    for i in range(episodes):
        results.append(run_episode(env, agent, seed + i, on_step=lambda e: frames.append(e.render(camera))))
        frames += [frames[-1]] * round(hold_s * fps)
    env.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, macro_block_size=8)
    return results


def watch(agent: PPO | None, env_cfg: LongJumpConfig, episodes: int = 5, seed: int = 1000, slowdown: float = 1.0):
    import mujoco.viewer

    env = LongJumpEnv(env_cfg)
    results = []
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = env.model.camera("follow").id

        def sync(e):
            viewer.sync()
            time.sleep(env_cfg.control_dt * slowdown)

        for i in range(episodes):
            if not viewer.is_running():
                break
            result = run_episode(env, agent, seed + i, on_step=sync)
            results.append(result)
            print(format_result(i, result), flush=True)
            time.sleep(0.8)
    env.close()
    return results


def format_result(i: int, r: dict) -> str:
    return (f"attempt {i + 1}: {r['outcome']:<18} jump {r['raw_jump_distance'] * 100:5.1f} cm | "
            f"flight {r['flight_time'] * 1000:4.0f} ms | take-off gap {r['takeoff_gap'] * 100:5.1f} cm | "
            f"return {r['return']:6.2f}")
