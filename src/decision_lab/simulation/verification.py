"""Physical grasp evidence, isolated from the commands that try to grasp."""

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class GraspVerificationSettings:
    minimum_contact_force_n: float = 0.02
    minimum_clearance_m: float = 0.05
    stable_window_s: float = 0.5
    maximum_relative_position_drift_m: float = 0.003
    maximum_relative_angle_drift_rad: float = 0.1

    def __post_init__(self):
        for name, value in vars(self).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite.")


class GraspVerifier:
    def __init__(self, model, table_top_m, cube_size_m, settings):
        self.model = model
        self.settings = settings
        self.table_top_m = table_top_m
        self.cube_half_size_m = cube_size_m / 2
        self.cube_id = model.body("cube").id
        self.site_id = model.site("grasp_center").id
        self.finger_ids = (model.body("left_finger").id, model.body("right_finger").id)
        self.finger_joints = [model.joint(name).qposadr[0] for name in
                              ("finger_joint1", "finger_joint2")]
        self.reset_window()

    def reset_window(self):
        self.stable_since_s = None
        self.reference_position = None
        self.reference_rotation = None

    def reset(self):
        self.reset_window()

    def measure(self, data):
        contact_forces = [0.0, 0.0]
        force = np.zeros(6)
        for index, contact in enumerate(data.contact):
            first = int(self.model.geom_bodyid[contact.geom1])
            second = int(self.model.geom_bodyid[contact.geom2])
            if self.cube_id not in (first, second):
                continue
            for finger_index, finger_id in enumerate(self.finger_ids):
                if finger_id in (first, second):
                    mujoco.mj_contactForce(self.model, data, index, force)
                    contact_forces[finger_index] += max(0.0, float(force[0]))
        cube_position = data.xpos[self.cube_id].copy()
        cube_rotation = data.xmat[self.cube_id].reshape(3, 3)
        tcp_rotation = data.site_xmat[self.site_id].reshape(3, 3)
        relative_position = tcp_rotation.T @ (cube_position - data.site_xpos[self.site_id])
        relative_rotation = tcp_rotation.T @ cube_rotation
        # Exact lowest world Z of a rotated cube's oriented bounding box.
        vertical_half_extent = self.cube_half_size_m * np.sum(np.abs(cube_rotation[2, :]))
        clearance = float(cube_position[2] - vertical_half_extent - self.table_top_m)
        bilateral = min(contact_forces) >= self.settings.minimum_contact_force_n
        eligible = bilateral and clearance >= self.settings.minimum_clearance_m
        position_drift = 0.0
        angle_drift = 0.0
        if not eligible:
            self.reset_window()
        else:
            if self.reference_position is not None:
                position_drift = float(np.linalg.norm(relative_position - self.reference_position))
                delta_rotation = self.reference_rotation.T @ relative_rotation
                angle_drift = float(np.arccos(np.clip((np.trace(delta_rotation) - 1) / 2, -1, 1)))
                if (position_drift > self.settings.maximum_relative_position_drift_m
                        or angle_drift > self.settings.maximum_relative_angle_drift_rad):
                    self.reset_window()
            if self.reference_position is None:
                self.reference_position = relative_position.copy()
                self.reference_rotation = relative_rotation.copy()
                self.stable_since_s = float(data.time)
        stable_elapsed = 0.0
        if self.stable_since_s is not None:
            stable_elapsed = float(data.time - self.stable_since_s)
        return {
            "cube_position_m": cube_position.tolist(),
            "cube_height_m": float(cube_position[2]),
            "bottom_clearance_m": clearance,
            "left_contact_force_n": contact_forces[0],
            "right_contact_force_n": contact_forces[1],
            "bilateral_contact": bilateral,
            "gripper_opening_m": float(np.sum(data.qpos[self.finger_joints])),
            "relative_position_m": relative_position.tolist(),
            "relative_position_drift_m": position_drift,
            "relative_angle_drift_rad": angle_drift,
            "stable_elapsed_s": stable_elapsed,
            "holding_confirmed": eligible and stable_elapsed >= self.settings.stable_window_s,
        }
