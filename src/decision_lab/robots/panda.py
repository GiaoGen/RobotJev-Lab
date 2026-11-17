"""Panda pose IK and bounded position-actuator motion, independent of the UI."""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


@dataclass(frozen=True)
class PoseTarget:
    id: str
    position_m: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]

    def __post_init__(self):
        position = np.asarray(self.position_m, dtype=float)
        quaternion = np.asarray(self.quaternion_xyzw, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("Target position must contain three finite numbers.")
        if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
            raise ValueError("Target quaternion must contain four finite numbers.")
        if abs(np.linalg.norm(quaternion) - 1.0) > 1e-5:
            raise ValueError("Target quaternion must have unit length (xyzw).")
        object.__setattr__(self, "position_m", tuple(position))
        object.__setattr__(self, "quaternion_xyzw", tuple(quaternion))

    @property
    def quaternion_wxyz(self):
        x, y, z, w = self.quaternion_xyzw
        return np.array([w, x, y, z], dtype=float)


@dataclass(frozen=True)
class ControlSettings:
    control_period_s: float = 0.02
    damping: float = 0.03
    orientation_weight_m: float = 0.25
    ik_position_tolerance_m: float = 0.0002
    ik_orientation_tolerance_rad: float = 0.002
    ik_max_iterations: int = 250
    ik_max_joint_step_rad: float = 0.1
    joint_margin_rad: float = 0.01
    path_sample_rad: float = 0.025
    max_joint_speed_rad_s: float = 0.6
    minimum_motion_s: float = 1.5
    position_tolerance_m: float = 0.005
    orientation_tolerance_rad: float = 0.035
    joint_speed_tolerance_rad_s: float = 0.05
    stable_window_s: float = 0.25
    settling_timeout_s: float = 4.0
    collision_force_threshold_n: float = 0.05
    collision_penetration_threshold_m: float = 0.0001

    def __post_init__(self):
        for name, value in vars(self).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite.")
        if not isinstance(self.ik_max_iterations, int):
            raise ValueError("ik_max_iterations must be an integer.")


@dataclass
class IKResult:
    converged: bool
    reason: str
    joint_positions: np.ndarray
    iterations: int
    position_error_m: float
    orientation_error_rad: float


@dataclass
class CarriedBody:
    body_id: int
    qpos_address: int
    relative_position: np.ndarray
    relative_rotation: np.ndarray


def panda_spec(scene_path: Path, targets: list[PoseTarget]):
    """Extend the official model in memory with reference points only."""
    spec = mujoco.MjSpec.from_file(str(scene_path))
    # Finger root (0.0584 m) + fingertip pad center (about 0.045 m).
    spec.body("hand").add_site(
        name="grasp_center", pos=[0, 0, 0.1034],
        size=[0.008, 0, 0], rgba=[0.1, 0.9, 0.3, 1],
    )
    for target in targets:
        spec.worldbody.add_site(
            name=f"target_{target.id}", pos=target.position_m,
            quat=target.quaternion_wxyz, size=[0.012, 0, 0],
            rgba=[1, 0.7, 0.1, 0.6],
        )
    return spec


def load_panda(scene_path: Path, targets: list[PoseTarget]):
    return panda_spec(scene_path, targets).compile()


class PandaArm:
    def __init__(self, model: mujoco.MjModel, settings: ControlSettings):
        self.model = model
        self.settings = settings
        self.site_id = model.site("grasp_center").id
        self.home_id = model.key("home").id
        joints = [model.joint(f"joint{i}") for i in range(1, 8)]
        actuators = [model.actuator(f"actuator{i}") for i in range(1, 8)]
        for joint, actuator in zip(joints, actuators):
            if model.actuator_trnid[actuator.id, 0] != joint.id:
                raise ValueError(f"Unexpected actuator mapping for {joint.name}.")
        self.qpos_indices = np.array([joint.qposadr[0] for joint in joints])
        self.dof_indices = np.array([joint.dofadr[0] for joint in joints])
        self.actuator_indices = np.array([actuator.id for actuator in actuators])
        self.position_gains = model.actuator_gainprm[self.actuator_indices, 0].copy()
        if not np.all(self.position_gains > 0):
            raise ValueError("Panda position actuator gains must be positive.")
        self.gripper_actuator = model.actuator("actuator8").id
        self.gripper_target = 255.0
        self.allowed_contacts = frozenset()
        self.allowed_geom_contacts = frozenset()
        self.payload = None
        self.robot_bodies = {model.body("link0").id}
        for body_id in range(model.nbody):
            if model.body_parentid[body_id] in self.robot_bodies:
                self.robot_bodies.add(body_id)
        self.lower = np.maximum(
            model.jnt_range[[joint.id for joint in joints], 0],
            model.actuator_ctrlrange[self.actuator_indices, 0],
        ) + settings.joint_margin_rad
        self.upper = np.minimum(
            model.jnt_range[[joint.id for joint in joints], 1],
            model.actuator_ctrlrange[self.actuator_indices, 1],
        ) - settings.joint_margin_rad
        self.planning_data = mujoco.MjData(model)
        self.status = "ready"
        self.reason = "home"
        self.target = None
        self.ik_result = None
        self.last_ik_attempt = None
        self.start_time_s = 0.0
        self.duration_s = 0.0
        self.stable_since_s = None
        self.start_joints = np.zeros(7)
        self.goal_joints = np.zeros(7)

    def set_gripper(self, data, opening_m):
        """Command total finger opening in metres through the official mapping."""
        if not np.isfinite(opening_m) or not 0 <= opening_m <= 0.08:
            raise ValueError("Panda gripper opening must be within 0 to 0.08 m.")
        self.gripper_target = 255.0 * (opening_m / 0.08)
        data.ctrl[self.gripper_actuator] = self.gripper_target

    def reset(self, data):
        mujoco.mj_resetDataKeyframe(self.model, data, self.home_id)
        mujoco.mj_forward(self.model, data)
        self.status = "ready"
        self.reason = "home"
        self.target = None
        self.ik_result = None
        self.last_ik_attempt = None
        self.stable_since_s = None
        self.gripper_target = 255.0
        self.allowed_contacts = frozenset()
        self.allowed_geom_contacts = frozenset()
        self.payload = None

    def capture_payload(self, data, body_id):
        joint_id = int(self.model.body_jntadr[body_id])
        if (self.model.body_jntnum[body_id] != 1
                or self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE):
            raise ValueError("A carried body must have one free joint.")
        rotation = data.site_xmat[self.site_id].reshape(3, 3)
        return CarriedBody(
            body_id, int(self.model.jnt_qposadr[joint_id]),
            rotation.T @ (data.xpos[body_id] - data.site_xpos[self.site_id]),
            rotation.T @ data.xmat[body_id].reshape(3, 3),
        )

    def pose_error(self, data, target):
        position_error = np.asarray(target.position_m) - data.site_xpos[self.site_id]
        current_quaternion = np.empty(4)
        inverse_quaternion = np.empty(4)
        difference_quaternion = np.empty(4)
        rotation_error = np.empty(3)
        mujoco.mju_mat2Quat(current_quaternion, data.site_xmat[self.site_id])
        mujoco.mju_negQuat(inverse_quaternion, current_quaternion)
        mujoco.mju_mulQuat(
            difference_quaternion, target.quaternion_wxyz, inverse_quaternion,
        )
        # target * inverse(current) gives a world-frame rotation error,
        # matching the world-frame rotational Jacobian returned by MuJoCo.
        mujoco.mju_quat2Vel(rotation_error, difference_quaternion, 1.0)
        return position_error, rotation_error

    def solve_ik(self, data, target):
        """Solve on scratch MjData; actual physics state is preserved."""
        scratch = self.planning_data
        mujoco.mj_resetData(self.model, scratch)
        scratch.qpos[:] = data.qpos
        scratch.qpos[self.qpos_indices] = np.clip(
            scratch.qpos[self.qpos_indices], self.lower, self.upper,
        )
        jacobian_position = np.zeros((3, self.model.nv))
        jacobian_rotation = np.zeros((3, self.model.nv))
        settings = self.settings
        for iteration in range(settings.ik_max_iterations + 1):
            mujoco.mj_forward(self.model, scratch)
            position_error, rotation_error = self.pose_error(scratch, target)
            position_norm = float(np.linalg.norm(position_error))
            rotation_norm = float(np.linalg.norm(rotation_error))
            if (position_norm <= settings.ik_position_tolerance_m
                    and rotation_norm <= settings.ik_orientation_tolerance_rad):
                return IKResult(
                    True, "converged", scratch.qpos[self.qpos_indices].copy(),
                    iteration, position_norm, rotation_norm,
                )
            if iteration == settings.ik_max_iterations:
                break
            mujoco.mj_jacSite(
                self.model, scratch, jacobian_position, jacobian_rotation,
                self.site_id,
            )
            weight = settings.orientation_weight_m
            jacobian = np.vstack((
                jacobian_position[:, self.dof_indices],
                weight * jacobian_rotation[:, self.dof_indices],
            ))
            error = np.concatenate((position_error, weight * rotation_error))
            regularized = jacobian @ jacobian.T + settings.damping**2 * np.eye(6)
            increment = jacobian.T @ np.linalg.solve(regularized, error)
            scale = max(1.0, np.max(np.abs(increment)) / settings.ik_max_joint_step_rad)
            scratch.qpos[self.qpos_indices] = np.clip(
                scratch.qpos[self.qpos_indices] + increment / scale,
                self.lower, self.upper,
            )
        return IKResult(
            False, "ik_not_converged", scratch.qpos[self.qpos_indices].copy(),
            iteration, position_norm, rotation_norm,
        )

    def collision(self, data, allowed_contacts=frozenset(), *,
                  moving_bodies=frozenset(), allowed_geom_contacts=frozenset()):
        force = np.zeros(6)
        for index, contact in enumerate(data.contact):
            first_id = int(self.model.geom_bodyid[contact.geom1])
            second_id = int(self.model.geom_bodyid[contact.geom2])
            if not ({first_id, second_id} & (self.robot_bodies | moving_bodies)):
                continue
            if frozenset((first_id, second_id)) in allowed_contacts:
                continue
            if frozenset((int(contact.geom1), int(contact.geom2))) in allowed_geom_contacts:
                continue
            mujoco.mj_contactForce(self.model, data, index, force)
            if (contact.dist < -self.settings.collision_penetration_threshold_m
                    or force[0] >= self.settings.collision_force_threshold_n):
                first = self.model.body(first_id).name
                second = self.model.body(second_id).name
                return f"collision:{first}/{second}"
        return None

    def execution_collision(self, data):
        moving = frozenset() if self.payload is None else frozenset((self.payload.body_id,))
        return self.collision(
            data, self.allowed_contacts, moving_bodies=moving,
            allowed_geom_contacts=self.allowed_geom_contacts,
        )

    def check_path(self, data, goal_joints, allowed_contacts=frozenset(), *,
                   payload=None, allowed_geom_contacts=frozenset()):
        start = data.qpos[self.qpos_indices].copy()
        samples = max(1, int(np.ceil(
            np.max(np.abs(goal_joints - start)) / self.settings.path_sample_rad,
        )))
        scratch = self.planning_data
        scratch.qpos[:] = data.qpos
        for fraction in np.linspace(0.0, 1.0, samples + 1):
            scratch.qpos[self.qpos_indices] = start + fraction * (goal_joints - start)
            mujoco.mj_forward(self.model, scratch)
            moving_bodies = frozenset()
            if payload is not None:
                rotation = scratch.site_xmat[self.site_id].reshape(3, 3)
                address = payload.qpos_address
                scratch.qpos[address:address + 3] = (
                    scratch.site_xpos[self.site_id] + rotation @ payload.relative_position
                )
                mujoco.mju_mat2Quat(
                    scratch.qpos[address + 3:address + 7],
                    (rotation @ payload.relative_rotation).ravel(),
                )
                mujoco.mj_forward(self.model, scratch)
                moving_bodies = frozenset((payload.body_id,))
            reason = self.collision(
                scratch, allowed_contacts, moving_bodies=moving_bodies,
                allowed_geom_contacts=allowed_geom_contacts,
            )
            if reason:
                return reason
        return None

    def start(self, data, target, allowed_contacts=frozenset(), *,
              payload=None, allowed_geom_contacts=frozenset()):
        if self.status == "moving":
            return "busy"
        solution = self.solve_ik(data, target)
        self.last_ik_attempt = solution
        if not solution.converged:
            return solution.reason
        reason = self.check_path(
            data, solution.joint_positions, allowed_contacts, payload=payload,
            allowed_geom_contacts=allowed_geom_contacts,
        )
        if reason:
            return reason
        self.target = target
        self.allowed_contacts = frozenset(allowed_contacts)
        self.allowed_geom_contacts = frozenset(allowed_geom_contacts)
        self.payload = payload
        self.ik_result = solution
        self.start_joints = data.qpos[self.qpos_indices].copy()
        self.goal_joints = solution.joint_positions.copy()
        self.start_time_s = float(data.time)
        # A quintic smoothstep has peak derivative 1.875.
        self.duration_s = max(
            self.settings.minimum_motion_s,
            1.875 * np.max(np.abs(self.goal_joints - self.start_joints))
            / self.settings.max_joint_speed_rad_s,
        )
        self.stable_since_s = None
        self.status = "moving"
        self.reason = "trajectory_started"
        return None

    def stop(self, data, reason, status="failed"):
        data.ctrl[self.actuator_indices] = np.clip(
            data.qpos[self.qpos_indices], self.lower, self.upper,
        )
        self.status = status
        self.reason = reason

    def update(self, data):
        if self.status != "moving":
            return
        elapsed = float(data.time) - self.start_time_s
        fraction = np.clip(elapsed / self.duration_s, 0.0, 1.0)
        blend = 10 * fraction**3 - 15 * fraction**4 + 6 * fraction**5
        nominal = self.start_joints + blend * (self.goal_joints - self.start_joints)
        # The official position actuators have no gravity feedforward. A small
        # target offset generates the bias torque through the same bounded
        # actuators. qfrc_bias includes gravity and velocity-dependent forces.
        bias_offset = data.qfrc_bias[self.dof_indices] / self.position_gains
        data.ctrl[self.actuator_indices] = np.clip(
            nominal + bias_offset, self.lower, self.upper,
        )
        data.ctrl[self.gripper_actuator] = self.gripper_target
        if elapsed < self.duration_s:
            return
        position_error, rotation_error = self.pose_error(data, self.target)
        settled = (
            np.linalg.norm(position_error) <= self.settings.position_tolerance_m
            and np.linalg.norm(rotation_error) <= self.settings.orientation_tolerance_rad
            and np.max(np.abs(data.qvel[self.dof_indices]))
            <= self.settings.joint_speed_tolerance_rad_s
        )
        if settled:
            if self.stable_since_s is None:
                self.stable_since_s = float(data.time)
            if data.time - self.stable_since_s >= self.settings.stable_window_s:
                self.status = "succeeded"
                self.reason = "pose_stable"
        else:
            self.stable_since_s = None
        if (self.status == "moving"
                and elapsed > self.duration_s + self.settings.settling_timeout_s):
            self.stop(data, "tracking_timeout", "timed_out")
