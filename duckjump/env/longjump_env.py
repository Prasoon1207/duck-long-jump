"""Gymnasium environment: the duck runs up to the take-off line and jumps into the pit.

Rules (a simplified version of the real event):
  * The attempt ends the first time a foot touches the ground at or past the line.
  * The jump distance is measured from the line to the rearmost foot contact at
    that moment, clamped at 0. Stepping over the line while the other foot is
    still on the runway therefore scores 0; only a real flight scores.
  * After landing the duck must stay up for `settle_time` seconds.
  * Falling (trunk too low or tilted) ends the episode.
"""

from dataclasses import dataclass, field

import gymnasium as gym
import mujoco
import numpy as np

from duckjump.env import robot as duck
from duckjump.env.arena import FLOOR_GEOM, ArenaConfig, build_arena
from duckjump.env.reward import RewardWeights, StepInfo, compute_reward


@dataclass
class LongJumpConfig:
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    reward: RewardWeights = field(default_factory=RewardWeights)
    # Each episode starts this far (m) before the line, sampled uniformly. Short starts
    # make the jump easy to discover; the demo always uses the full arena run-up.
    start_distance_range: tuple[float, float] = (0.05, 0.4)
    motor_strength: float = 2.0   # 1.0 = real servos; the real duck can barely leave the ground
    action_scale: float = 0.5     # rad of joint offset from the home pose per unit action
    sim_dt: float = 0.002
    control_dt: float = 0.02
    max_time: float = 4.0
    settle_time: float = 0.3
    joint_noise: float = 0.03     # reset noise on leg joints (rad)
    fall_height: float = 0.08     # base height below which the duck has fallen (m)
    fall_uprightness: float = 0.5  # cos(tilt) below which the duck has fallen


class LongJumpEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, cfg: LongJumpConfig | None = None, render_mode: str | None = None):
        self.cfg = cfg or LongJumpConfig()
        self.render_mode = render_mode
        self.model = build_arena(self.cfg.arena)
        self.model.opt.timestep = self.cfg.sim_dt
        self.model.opt.iterations = 50
        self.model.opt.ls_iterations = 20
        duck.scale_motor_strength(self.model, self.cfg.motor_strength)
        self.data = mujoco.MjData(self.model)
        self.robot = duck.Robot(self.model)
        self.floor = self.model.geom(FLOOR_GEOM).id
        self.line_x = self.cfg.arena.runup_length
        self.n_substeps = round(self.cfg.control_dt / self.cfg.sim_dt)
        self.max_steps = round(self.cfg.max_time / self.cfg.control_dt)
        self.settle_steps = round(self.cfg.settle_time / self.cfg.control_dt)

        self.phase = "runup"
        n_act = len(duck.LEG_JOINTS)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (n_act,), np.float32)
        obs_dim = len(self._observe(np.zeros(n_act), np.zeros(2)))
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self._renderer = None

    # ------------------------------------------------------------------ helpers

    def _foot_contacts(self) -> np.ndarray:
        """Per-foot ground contact flags."""
        geoms = self.data.contact.geom
        return np.isin(self.robot.foot_geoms, geoms).astype(np.float64)

    def _ground_contact_xs(self) -> np.ndarray:
        """x of every ground contact point. Only the robot touches the ground, so any
        contact counts as a mark in the pit, whichever body part made it."""
        n = self.data.ncon
        return self.data.contact.pos[:n, 0] if n else np.empty(0)

    def _observe(self, last_action: np.ndarray, contacts: np.ndarray) -> np.ndarray:
        r, d = self.robot, self.data
        phase = [1.0, 0.0] if self.phase == "runup" else [0.0, 1.0]
        return np.concatenate([
            r.projected_gravity(d),
            r.gyro(d) * 0.25,
            r.local_linvel(d),
            r.leg_pos(d) - r.home_leg,
            r.leg_vel(d) * 0.05,
            last_action,
            contacts,
            [np.clip(r.base_pos(d)[0] - self.line_x, -1.0, 1.0)],
            [r.base_pos(d)[2] - duck.HOME_BASE_HEIGHT],
            phase,
        ]).astype(np.float32)

    def _fallen(self) -> bool:
        uprightness = -self.robot.projected_gravity(self.data)[2]
        return self.robot.base_pos(self.data)[2] < self.cfg.fall_height or uprightness < self.cfg.fall_uprightness

    # ---------------------------------------------------------------- gym API

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.start_distance = options.get("start_distance", self.np_random.uniform(*self.cfg.start_distance_range))
        self.robot.reset_pose(self.data, self.np_random, self.cfg.joint_noise, x=self.line_x - self.start_distance)
        self.phase = "runup"
        self.steps = 0
        self.steps_since_landing = 0
        self.last_action = np.zeros(self.action_space.shape)
        self.jump_distance = 0.0
        self.takeoff_x = np.nan
        self.flight_time = 0.0
        self.episode_terms: dict[str, float] = {}
        return self._observe(self.last_action, self._foot_contacts()), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.robot.apply_leg_targets(self.data, self.robot.home_leg + self.cfg.action_scale * action)

        just_landed = False
        airborne_time = 0.0
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
            if self.phase != "runup":
                continue
            contact_xs = self._ground_contact_xs()
            if contact_xs.size == 0:
                airborne_time += self.cfg.sim_dt
                self.flight_time += self.cfg.sim_dt
            elif contact_xs.max() >= self.line_x:
                self.phase = "landed"
                just_landed = True
                self.jump_distance = max(0.0, contact_xs.min() - self.line_x)
            else:
                self.takeoff_x = contact_xs.max()
                self.flight_time = 0.0

        self.steps += 1
        if self.phase == "landed" and not just_landed:
            self.steps_since_landing += 1

        fell = self._fallen()
        uprightness = float(np.clip(-self.robot.projected_gravity(self.data)[2], 0.0, 1.0))
        step_info = StepInfo(
            phase=self.phase if not just_landed else "runup",
            forward_velocity=float(self.robot.world_linvel(self.data)[0]),
            airborne_time=airborne_time,
            uprightness=uprightness,
            action_delta=action - self.last_action,
            just_landed=just_landed,
            jump_distance=self.jump_distance,
            fell=fell,
        )
        reward, terms = compute_reward(step_info, self.cfg.reward, self.cfg.control_dt)
        for k, v in terms.items():
            self.episode_terms[k] = self.episode_terms.get(k, 0.0) + v

        self.last_action = action
        obs = self._observe(action, self._foot_contacts())
        settled = self.phase == "landed" and self.steps_since_landing >= self.settle_steps
        terminated = fell or settled
        truncated = not terminated and self.steps >= self.max_steps

        info = {}
        if terminated or truncated:
            if self.phase == "landed":
                outcome = "fell_after_landing" if fell else ("jump" if self.jump_distance > 0 else "stepped_over")
            else:
                outcome = "fell" if fell else "timeout"
            info["episode"] = {
                "outcome": outcome,
                "jump_distance": self.jump_distance if outcome == "jump" else 0.0,
                "raw_jump_distance": self.jump_distance,
                "flight_time": self.flight_time if self.phase == "landed" else 0.0,
                "takeoff_gap": self.line_x - self.takeoff_x if not np.isnan(self.takeoff_x) else np.nan,
                "length": self.steps,
                "start_distance": self.start_distance,
                **{f"reward/{k}": v for k, v in self.episode_terms.items()},
            }
        return obs, reward, terminated, truncated, info

    # -------------------------------------------------------------- rendering

    def render(self, camera: str = "follow", width: int = 1280, height: int = 720):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
