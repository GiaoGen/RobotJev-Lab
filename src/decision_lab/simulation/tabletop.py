"""A table and free cube composed around the fixed official Panda model."""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from decision_lab.robots.panda import panda_spec


@dataclass(frozen=True)
class TabletopSettings:
    table_center_xy_m: tuple[float, float] = (0.5, 0.0)
    table_size_xy_m: tuple[float, float] = (0.6, 0.7)
    table_top_m: float = 0.2
    table_thickness_m: float = 0.04
    cube_position_xy_m: tuple[float, float] = (0.45, -0.05)
    cube_size_m: float = 0.04
    cube_mass_kg: float = 0.08
    cube_friction: float = 1.0
    initial_clearance_m: float = 0.001
    noslip_iterations: int = 5
    seed: int = 0
    cube_position_jitter_xy_m: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self):
        for name in ("table_center_xy_m", "table_size_xy_m", "cube_position_xy_m", "cube_position_jitter_xy_m"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (2,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must contain two finite numbers.")
        for name in ("table_top_m", "table_thickness_m", "cube_size_m", "cube_mass_kg"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite.")
        if min(self.table_size_xy_m) <= 0 or self.table_thickness_m >= self.table_top_m:
            raise ValueError("Table dimensions must provide a positive tabletop and leg height.")
        if not np.isfinite(self.initial_clearance_m) or self.initial_clearance_m < 0:
            raise ValueError("initial_clearance_m must be nonnegative and finite.")
        if not np.isfinite(self.cube_friction) or self.cube_friction < 0:
            raise ValueError("cube_friction must be nonnegative and finite.")
        if not isinstance(self.noslip_iterations, int) or self.noslip_iterations < 0:
            raise ValueError("noslip_iterations must be a nonnegative integer.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        if min(self.cube_position_jitter_xy_m) < 0:
            raise ValueError("cube_position_jitter_xy_m must be nonnegative.")


def tabletop_spec(scene_path: Path, settings: TabletopSettings):
    spec = panda_spec(scene_path, [])
    # Improve static-friction accuracy for sustained fingertip holding.
    spec.option.noslip_iterations = settings.noslip_iterations
    x, y = settings.table_center_xy_m
    half_x, half_y = np.asarray(settings.table_size_xy_m) / 2
    thickness = settings.table_thickness_m
    spec.worldbody.add_geom(
        name="tabletop", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[x, y, settings.table_top_m - thickness / 2],
        size=[half_x, half_y, thickness / 2], rgba=[0.55, 0.40, 0.26, 1],
    )
    leg_height = settings.table_top_m - thickness
    for index, (dx, dy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        spec.worldbody.add_geom(
            name=f"table_leg_{index}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=[x + dx * (half_x - 0.03), y + dy * (half_y - 0.03), leg_height / 2],
            size=[0.018, leg_height / 2, 0], rgba=[0.22, 0.25, 0.29, 1],
        )
    offset = np.random.default_rng(settings.seed).uniform(
        -np.asarray(settings.cube_position_jitter_xy_m), settings.cube_position_jitter_xy_m,
    )
    cube_position = [
        *(np.asarray(settings.cube_position_xy_m) + offset),
        settings.table_top_m + settings.cube_size_m / 2 + settings.initial_clearance_m,
    ]
    cube = spec.worldbody.add_body(name="cube", pos=cube_position)
    cube.add_freejoint(name="cube_free")
    cube.add_geom(
        name="cube_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[settings.cube_size_m / 2] * 3, mass=settings.cube_mass_kg,
        friction=[settings.cube_friction, 0.005, 0.0001], priority=1,
        # A frictionless surface has normal-only contact constraints.
        condim=1 if settings.cube_friction == 0 else 3,
        rgba=[0.9, 0.16, 0.10, 1],
    )
    return spec


def compile_tabletop(spec):
    model = spec.compile()
    # Extend the author's home keyframe with the free object's declared pose.
    cube_qpos = model.joint("cube_free").qposadr[0]
    model.key_qpos[model.key("home").id, cube_qpos:cube_qpos + 7] = model.qpos0[cube_qpos:cube_qpos + 7]
    return model


def create_tabletop(scene_path: Path, settings: TabletopSettings):
    return compile_tabletop(tabletop_spec(scene_path, settings))
