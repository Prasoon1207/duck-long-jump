"""Open Duck Mini v2: model loading, joint bookkeeping and state readers."""

from pathlib import Path

import mujoco
import numpy as np

ROBOT_XML = Path(__file__).resolve().parents[2] / "assets" / "open_duck_mini_v2" / "open_duck_mini_v2.xml"

# Standing pose from the Open Duck Playground "home" keyframe, in actuator order.
HOME_CTRL = {
    "left_hip_yaw": 0.002, "left_hip_roll": 0.053, "left_hip_pitch": -0.63,
    "left_knee": 1.368, "left_ankle": -0.784,
    "neck_pitch": 0.0, "head_pitch": 0.0, "head_yaw": 0.0, "head_roll": 0.0,
    "right_hip_yaw": -0.003, "right_hip_roll": -0.065, "right_hip_pitch": 0.635,
    "right_knee": 1.379, "right_ankle": -0.796,
}

# The policy only drives the legs; the head is held at its home pose.
LEG_JOINTS = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]

FOOT_GEOMS = ["left_foot_bottom_tpu", "right_foot_bottom_tpu"]
BASE_BODY = "base"
HOME_BASE_HEIGHT = 0.16


def load_robot_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(ROBOT_XML))


def scale_motor_strength(model: mujoco.MjModel, strength: float) -> None:
    """Scale servo stiffness and torque limits (1.0 = real STS3215 servos)."""
    model.actuator_gainprm[:, 0] *= strength
    model.actuator_biasprm[:, 1] *= strength
    model.actuator_forcerange[:] *= strength


class Robot:
    """Index lookups and state readers for a compiled model containing the duck."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.base_body = model.body(BASE_BODY).id
        self.home_ctrl = np.array([HOME_CTRL[model.actuator(i).name] for i in range(model.nu)])
        self.leg_act = np.array([model.actuator(n).id for n in LEG_JOINTS])
        leg_jnt = [model.joint(n).id for n in LEG_JOINTS]
        self.leg_qpos = model.jnt_qposadr[leg_jnt]
        self.leg_qvel = model.jnt_dofadr[leg_jnt]
        self.home_leg = self.home_ctrl[self.leg_act]
        self.ctrl_low = model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = model.actuator_ctrlrange[:, 1].copy()
        self.foot_geoms = np.array([model.geom(n).id for n in FOOT_GEOMS])
        self._gyro = self._sensor_slice("gyro")
        self._local_linvel = self._sensor_slice("local_linvel")

    def _sensor_slice(self, name: str) -> slice:
        s = self.model.sensor(name)
        return slice(s.adr[0], s.adr[0] + s.dim[0])

    def reset_pose(self, data: mujoco.MjData, rng: np.random.Generator, joint_noise: float, x: float = 0.0) -> None:
        mujoco.mj_resetData(self.model, data)
        data.qpos[0:3] = [x, 0.0, HOME_BASE_HEIGHT]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        for i in range(self.model.nu):
            qadr = self.model.jnt_qposadr[self.model.actuator_trnid[i, 0]]
            data.qpos[qadr] = self.home_ctrl[i]
        data.qpos[self.leg_qpos] += rng.uniform(-joint_noise, joint_noise, len(self.leg_qpos))
        data.ctrl[:] = self.home_ctrl
        mujoco.mj_forward(self.model, data)

    def apply_leg_targets(self, data: mujoco.MjData, leg_targets: np.ndarray) -> None:
        ctrl = self.home_ctrl.copy()
        ctrl[self.leg_act] = leg_targets
        data.ctrl[:] = np.clip(ctrl, self.ctrl_low, self.ctrl_high)

    def base_pos(self, data) -> np.ndarray:
        return data.xpos[self.base_body]

    def projected_gravity(self, data) -> np.ndarray:
        """Gravity direction in the base frame; (0, 0, -1) when upright."""
        return data.xmat[self.base_body].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])

    def gyro(self, data) -> np.ndarray:
        return data.sensordata[self._gyro]

    def local_linvel(self, data) -> np.ndarray:
        return data.sensordata[self._local_linvel]

    def world_linvel(self, data) -> np.ndarray:
        return data.qvel[0:3]

    def leg_pos(self, data) -> np.ndarray:
        return data.qpos[self.leg_qpos]

    def leg_vel(self, data) -> np.ndarray:
        return data.qvel[self.leg_qvel]
