"""Long-jump arena: a runway, a take-off board and a landing pit, built around the robot.

Coordinates: the robot faces +x. The take-off line sits at x = runup_length, so a
full run-up starts at x = 0, and the landing pit extends beyond the line. Only the ground plane
collides; every other arena element is a visual marking.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from duckjump.env import robot as duck


@dataclass
class ArenaConfig:
    runup_length: float = 0.4     # start position to take-off line (m)
    pit_length: float = 0.6       # length of the landing pit (m)
    lane_width: float = 0.35
    board_depth: float = 0.03     # take-off board, ends exactly at the line
    marker_spacing: float = 0.05  # distance markings in the pit (m)
    ground_friction: float = 0.8


FLOOR_GEOM = "floor"
_MARK_Z = 0.0005  # visual markings float just above the ground


def _visual_box(body, name, center_xy, half_xy, rgba):
    body.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[center_xy[0], center_xy[1], _MARK_Z],
        size=[half_xy[0], half_xy[1], _MARK_Z],
        rgba=rgba,
        contype=0,
        conaffinity=0,
        group=1,
    )


def build_arena_spec(cfg: ArenaConfig) -> mujoco.MjSpec:
    spec = duck.load_robot_spec()
    world = spec.worldbody
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 720

    spec.add_texture(
        name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.55, 0.75, 0.95], rgb2=[0.1, 0.15, 0.25], width=512, height=512,
    )
    spec.add_texture(
        name="grass", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=[0.28, 0.5, 0.25], rgb2=[0.25, 0.46, 0.22], width=256, height=256,
    )
    grass = spec.add_material(name="grass", texrepeat=[8, 8], texuniform=True)
    grass.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grass"

    world.add_light(pos=[cfg.runup_length, -0.5, 2.0], dir=[0, 0.25, -1], diffuse=[0.8, 0.8, 0.8], castshadow=True)
    world.add_geom(
        name=FLOOR_GEOM, type=mujoco.mjtGeom.mjGEOM_PLANE, size=[4, 4, 0.05], material="grass",
        friction=[cfg.ground_friction, 0.005, 0.0001], condim=3,
    )

    line = cfg.runup_length
    half_w = cfg.lane_width / 2
    runway_start = -0.2
    _visual_box(world, "runway", [(runway_start + line) / 2, 0], [(line - runway_start) / 2, half_w], [0.72, 0.3, 0.22, 1])
    _visual_box(world, "board", [line - cfg.board_depth / 2, 0], [cfg.board_depth / 2, half_w], [0.95, 0.95, 0.95, 1])
    _visual_box(world, "pit", [line + cfg.pit_length / 2, 0], [cfg.pit_length / 2, half_w], [0.87, 0.76, 0.52, 1])

    for i, x in enumerate(np.arange(cfg.marker_spacing, cfg.pit_length + 1e-6, cfg.marker_spacing)):
        major = i % 2 == 1
        _visual_box(
            world, f"mark_{i}", [line + x, -half_w + (0.03 if major else 0.015)],
            [0.002, 0.03 if major else 0.015], [0.2, 0.2, 0.2, 1],
        )

    # Side view of the whole arena, and a camera that follows the duck.
    world.add_camera(
        name="side", pos=[line, -1.3, 0.45], xyaxes=[1, 0, 0, 0, 0.33, 1],
    )
    spec.body(duck.BASE_BODY).add_camera(
        name="follow", mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM,
        pos=[0.0, -0.75, 0.15], xyaxes=[1, 0, 0, 0, 0.2, 1],
    )
    return spec


def build_arena(cfg: ArenaConfig) -> mujoco.MjModel:
    return build_arena_spec(cfg).compile()
