"""Run a trained policy: evaluate it, record an MP4, or watch it in the MuJoCo viewer."""

import time
from pathlib import Path

import imageio
import mujoco
import numpy as np

from duckjump.algo.ppo import PPO
from duckjump.env.longjump_env import LongJumpConfig, LongJumpEnv

ISO = {"azimuth": 60.0, "elevation": -28.0, "distance": 1.35, "height": 0.18, "lead": 0.15, "smoothing": 0.15}


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


def iso_camera(env: LongJumpEnv, cam: mujoco.MjvCamera | None = None, track: bool = False) -> mujoco.MjvCamera:
    cam = cam or mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = ISO["azimuth"], ISO["elevation"], ISO["distance"]
    cam.lookat[:] = [*iso_target(env), ISO["height"]]
    if track:
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = env.robot.base_body
    else:
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    return cam


def iso_target(env: LongJumpEnv) -> np.ndarray:
    return env.robot.base_pos(env.data)[:2] + [ISO["lead"], 0.0]


def follow_iso(env: LongJumpEnv, cam: mujoco.MjvCamera, reset: bool = False):
    target = iso_target(env)
    cam.lookat[:2] = target if reset else cam.lookat[:2] + ISO["smoothing"] * (target - cam.lookat[:2])


class FrameGrabber:
    """Renders "iso" with a smoothed free camera, anything else with the model's named cameras."""

    def __init__(self, env: LongJumpEnv, camera: str):
        self.renderer = mujoco.Renderer(env.model, height=720, width=1280)
        self.cam = iso_camera(env) if camera == "iso" else camera
        self.fresh = True

    def __call__(self, env: LongJumpEnv) -> np.ndarray:
        if isinstance(self.cam, mujoco.MjvCamera):
            follow_iso(env, self.cam, reset=self.fresh)
        self.fresh = False
        self.renderer.update_scene(env.data, camera=self.cam)
        return self.renderer.render()

    def close(self):
        self.renderer.close()


def record_video(agent: PPO | None, env_cfg: LongJumpConfig, path: Path, episodes: int = 3,
                 camera: str = "iso", seed: int = 1000, hold_s: float = 0.6) -> list[dict]:
    env = LongJumpEnv(env_cfg)
    grab = FrameGrabber(env, camera)
    fps = round(1.0 / env_cfg.control_dt)
    frames, results = [], []
    for i in range(episodes):
        grab.fresh = True
        results.append(run_episode(env, agent, seed + i, on_step=lambda e: frames.append(grab(e))))
        frames += [frames[-1]] * round(hold_s * fps)
    grab.close()
    env.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, macro_block_size=8)
    return results


def watch(agent: PPO | None, env_cfg: LongJumpConfig, episodes: int = 5, seed: int = 1000,
          slowdown: float = 1.0, camera: str = "iso"):
    import mujoco.viewer

    env = LongJumpEnv(env_cfg)
    results = []
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        if camera == "iso":
            iso_camera(env, viewer.cam, track=True)
        else:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = env.model.camera(camera).id

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
    return (f"attempt {i + 1}: {r['outcome']:<18} jump {r['raw_jump_distance'] * 100:5.1f} cm "
            f"(first touch {r['first_touch_distance'] * 100:5.1f} cm) | "
            f"flight {r['flight_time'] * 1000:4.0f} ms | take-off gap {r['takeoff_gap'] * 100:5.1f} cm | "
            f"return {r['return']:6.2f}")
