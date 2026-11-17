"""Tray geometry and independent physical placement verification."""

from dataclasses import dataclass

import mujoco
import numpy as np

from decision_lab.simulation.tabletop import compile_tabletop, tabletop_spec
from decision_lab.simulation.verification import GraspVerifier


@dataclass(frozen=True)
class PlacementSettings:
    target_center_xy_m: tuple[float, float] = (0.48, 0.18)
    target_size_xy_m: tuple[float, float] = (0.14, 0.14)
    floor_thickness_m: float = 0.01
    wall_height_m: float = 0.015
    wall_thickness_m: float = 0.008
    release_clearance_m: float = 0.001
    retract_height_m: float = 0.12
    release_timeout_s: float = 1.5
    verify_timeout_s: float = 2.0
    stable_window_s: float = 0.5
    maximum_linear_speed_m_s: float = 0.02
    maximum_angular_speed_rad_s: float = 0.2
    minimum_support_force_n: float = 0.05
    maximum_finger_contact_force_n: float = 0.02
    forbidden_contact_force_n: float = 0.05
    forbidden_penetration_m: float = 0.0001
    support_height_tolerance_m: float = 0.004
    released_opening_m: float = 0.065
    minimum_retraction_m: float = 0.06
    maximum_payload_drift_m: float = 0.008
    carry_via_xy_m: tuple[tuple[float, float], ...] = ()
    drop_delay_s: float = 0.35
    drop_gripper_force_n: float = 0.1
    obstacle_height_m: float = 0.122
    obstacle_size_xy_m: tuple[float, float] = (0.036, 0.05)
    obstacle_blocked_xy_m: tuple[float, float] = (0.465, 0.065)
    obstacle_parked_xy_m: tuple[float, float] = (0.70, -0.20)
    forbidden_xy_m: tuple[float, float] = (0.70, 0.27)

    def __post_init__(self):
        for name, value in vars(self).items():
            if name == "carry_via_xy_m":
                for waypoint in value:
                    vector = np.asarray(waypoint, dtype=float)
                    if vector.shape != (2,) or not np.isfinite(vector).all():
                        raise ValueError("Carry waypoints must contain finite XY pairs.")
            elif name.endswith("xy_m"):
                vector = np.asarray(value, dtype=float)
                if vector.shape != (2,) or not np.isfinite(vector).all():
                    raise ValueError(f"{name} must contain two finite numbers.")
                if "size" in name and min(vector) <= 0:
                    raise ValueError(f"{name} must be positive.")
            elif not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite.")
        if self.released_opening_m > 0.08:
            raise ValueError("Released opening must be within Panda's 0.08 m range.")


def create_pick_place_scene(scene_path, scene, placement):
    spec = tabletop_spec(scene_path, scene)
    center = np.asarray(placement.target_center_xy_m)
    half = np.asarray(placement.target_size_xy_m) / 2
    wall = placement.wall_thickness_m
    if np.any(np.abs(center - scene.table_center_xy_m) + half + wall
              > np.asarray(scene.table_size_xy_m) / 2):
        raise ValueError("The tray must fit on the table.")
    tray = spec.worldbody.add_body(name="tray", pos=[*center, scene.table_top_m])
    tray.add_geom(
        name="tray_floor", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0, 0, placement.floor_thickness_m / 2],
        size=[*(half + wall), placement.floor_thickness_m / 2],
        rgba=[0.12, 0.45, 0.65, 1],
    )
    wall_z = placement.floor_thickness_m + placement.wall_height_m / 2
    for axis in range(2):
        for sign in (-1, 1):
            position = [0.0, 0.0, wall_z]
            position[axis] = sign * (half[axis] + wall / 2)
            size = [half[0] + wall, half[1] + wall, placement.wall_height_m / 2]
            size[axis] = wall / 2
            tray.add_geom(
                name=f"tray_wall_{axis}_{sign}", type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=position, size=size, rgba=[0.08, 0.27, 0.40, 1],
            )
    obstacle = spec.worldbody.add_body(
        name="path_obstacle", pos=[*placement.obstacle_parked_xy_m,
                                  scene.table_top_m + placement.obstacle_height_m / 2],
    )
    obstacle.add_geom(
        name="path_obstacle_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[*(np.asarray(placement.obstacle_size_xy_m) / 2), placement.obstacle_height_m / 2],
        rgba=[0.95, 0.65, 0.10, 1],
    )
    forbidden = spec.worldbody.add_body(
        name="forbidden", pos=[*placement.forbidden_xy_m, scene.table_top_m + 0.04],
    )
    forbidden.add_geom(
        name="forbidden_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.025, 0.025, 0.04], rgba=[0.55, 0.20, 0.65, 1],
    )
    return compile_tabletop(spec)


class PlacementVerifier:
    def __init__(self, model, scene, settings):
        self.model = model
        self.settings = settings
        self.cube_id = model.body("cube").id
        self.site_id = model.site("grasp_center").id
        self.floor_geom = model.geom("tray_floor").id
        self.half_size = scene.cube_size_m / 2
        self.tray_top_m = scene.table_top_m + settings.floor_thickness_m
        self.reset_window()

    def reset_window(self):
        self.stable_since_s = None

    def measure(self, data, holding, constraint_violated=False):
        position = data.xpos[self.cube_id]
        rotation = data.xmat[self.cube_id].reshape(3, 3)
        extent = self.half_size * np.sum(np.abs(rotation), axis=1)
        inside = bool(np.all(
            np.abs(position[:2] - self.settings.target_center_xy_m) + extent[:2]
            <= np.asarray(self.settings.target_size_xy_m) / 2,
        ))
        support_force = 0.0
        contact_force = np.zeros(6)
        for index, contact in enumerate(data.contact):
            bodies = self.model.geom_bodyid[[contact.geom1, contact.geom2]]
            if self.floor_geom in (contact.geom1, contact.geom2) and self.cube_id in bodies:
                if abs(contact.frame[2]) >= 0.9:
                    mujoco.mj_contactForce(self.model, data, index, contact_force)
                    support_force += max(0.0, float(contact_force[0]))
        bottom = float(position[2] - extent[2])
        supported = (support_force >= self.settings.minimum_support_force_n
                     and abs(bottom - self.tray_top_m) <= self.settings.support_height_tolerance_m)
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_BODY, self.cube_id, velocity, 0)
        linear_speed = float(np.linalg.norm(velocity[3:]))
        angular_speed = float(np.linalg.norm(velocity[:3]))
        released = (
            holding["gripper_opening_m"] >= self.settings.released_opening_m
            and max(holding["left_contact_force_n"], holding["right_contact_force_n"])
            < self.settings.maximum_finger_contact_force_n
        )
        retracted = bool(
            data.site_xpos[self.site_id, 2] - (position[2] + extent[2])
            >= self.settings.minimum_retraction_m
        )
        eligible = (
            inside and supported and released and retracted and not constraint_violated
            and linear_speed <= self.settings.maximum_linear_speed_m_s
            and angular_speed <= self.settings.maximum_angular_speed_rad_s
        )
        if eligible:
            if self.stable_since_s is None:
                self.stable_since_s = float(data.time)
        else:
            self.reset_window()
        stable = 0.0 if self.stable_since_s is None else float(data.time - self.stable_since_s)
        return {
            "inside_target_region": inside, "tray_supported": supported,
            "tray_support_force_n": support_force,
            "cube_linear_speed_m_s": linear_speed, "cube_angular_speed_rad_s": angular_speed,
            "gripper_released": released, "gripper_retracted": retracted,
            "placement_stable_s": stable,
            "placement_confirmed": bool(eligible and stable >= self.settings.stable_window_s),
        }


class PickPlaceVerifier:
    def __init__(self, model, scene, grasp_settings, placement_settings):
        self.holding = GraspVerifier(model, scene.table_top_m, scene.cube_size_m, grasp_settings)
        self.placement = PlacementVerifier(model, scene, placement_settings)
        self.settings = grasp_settings
        self.model = model
        self.forbidden_id = model.body("forbidden").id
        self.watched_bodies = {model.body(name).id for name in
                               ("cube", "link0", "link1", "link2", "link3", "link4", "link5",
                                "link6", "link7", "hand", "left_finger", "right_finger")}
        self.constraint_violated = False

    def reset_window(self):
        self.holding.reset_window()
        self.placement.reset_window()

    def reset(self):
        self.reset_window()
        self.constraint_violated = False

    def measure(self, data):
        force = np.zeros(6)
        for index, contact in enumerate(data.contact):
            bodies = {int(self.model.geom_bodyid[contact.geom1]), int(self.model.geom_bodyid[contact.geom2])}
            if self.forbidden_id in bodies and bodies & self.watched_bodies:
                mujoco.mj_contactForce(self.model, data, index, force)
                if (contact.dist < -self.placement.settings.forbidden_penetration_m
                        or force[0] >= self.placement.settings.forbidden_contact_force_n):
                    self.constraint_violated = True
        holding = self.holding.measure(data)
        return {
            **holding, **self.placement.measure(data, holding, self.constraint_violated),
            "constraint_violated": self.constraint_violated,
            "constraint_violation_reason": "forbidden_contact" if self.constraint_violated else None,
        }
