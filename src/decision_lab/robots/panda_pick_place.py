"""Compose the verified grasp skill with carrying, release and placement."""

import mujoco
import numpy as np

from decision_lab.robots.panda import PoseTarget
from decision_lab.robots.panda_grasp import PandaGrasp


class PandaPickPlace:
    def __init__(self, arm, grasp_settings, placement_settings, scene, on_event):
        self.arm = arm
        self.grasp_settings = grasp_settings
        self.settings = placement_settings
        self.scene = scene
        self.on_event = on_event
        self.grasp = PandaGrasp(arm, grasp_settings, on_event)
        self.allowed_contacts = self.grasp.allowed_contacts
        self.floor_contacts = frozenset((frozenset((
            arm.model.geom("cube_geom").id, arm.model.geom("tray_floor").id,
        )),))
        self.status = "ready"
        self.stage = "ready"
        self.reason = "home"
        self.phase_started_s = 0.0
        self.contact_lost_since_s = None
        self.payload = None
        self.quaternion = (0.7071067811865476, 0.7071067811865476, 0, 0)
        self.carry_targets = []
        self.carry_index = 0
        self.carry_height_m = 0.0

    @property
    def phase(self):
        return self.grasp.phase if self.stage == "grasp" else self.stage

    def enter(self, data, phase):
        self.stage = phase
        self.phase_started_s = float(data.time)
        self.on_event("phase_entered", phase=phase)

    def start(self, data, object_position_m, offset_xy_m=(0, 0)):
        if self.status == "running":
            return "busy"
        self.status = "running"
        self.reason = "executing"
        self.stage = "grasp"
        self.grasp.start(data, object_position_m, offset_xy_m)
        return None

    def fail(self, data, reason):
        self.arm.stop(data, reason)
        self.status = "failed"
        self.reason = reason
        self.enter(data, "failed")

    def cancel(self, data):
        if self.status == "running":
            self.grasp.cancel(data)
            self.arm.stop(data, "operator_cancelled", "cancelled")
            self.status = "cancelled"
            self.reason = "operator_cancelled"
            self.enter(data, "cancelled")

    def move(self, data, phase, position, *, carrying=False, floor_contact=False):
        payload = None
        if carrying:
            payload = self.arm.capture_payload(data, self.arm.model.body("cube").id)
        target = PoseTarget(phase, tuple(position), self.quaternion)
        allowed_geometry = self.floor_contacts if floor_contact else frozenset()
        reason = self.arm.start(
            data, target, self.allowed_contacts, payload=payload,
            allowed_geom_contacts=allowed_geometry,
        )
        self.on_event(
            "loaded_path_checked", phase=phase, carrying=carrying,
            accepted=reason is None, reason=reason,
            target_position_m=list(position),
            start_joint_positions_rad=data.qpos[self.arm.qpos_indices].tolist(),
            goal_joint_positions_rad=(None if self.arm.last_ik_attempt is None
                                      else self.arm.last_ik_attempt.joint_positions.tolist()),
            payload_relative_position_m=(None if payload is None else payload.relative_position.tolist()),
            payload_relative_rotation=(None if payload is None else payload.relative_rotation.tolist()),
        )
        if reason:
            self.fail(data, f"{phase}:{reason}")
            return
        self.payload = payload
        self.enter(data, phase)

    def tcp_for_cube(self, data, cube_position):
        payload = self.arm.capture_payload(data, self.arm.model.body("cube").id)
        rotation = np.empty(9)
        wxyz = np.array([self.quaternion[3], *self.quaternion[:3]])
        mujoco.mju_quat2Mat(rotation, wxyz)
        return np.asarray(cube_position) - rotation.reshape(3, 3) @ payload.relative_position

    def carry_next(self, data):
        cube_goal = [*self.carry_targets[self.carry_index], self.carry_height_m]
        self.on_event("carry_waypoint_selected", index=self.carry_index, cube_position_m=cube_goal)
        self.move(data, "carry", self.tcp_for_cube(data, cube_goal), carrying=True)

    def check_holding(self, data, evidence):
        if evidence["bilateral_contact"]:
            self.contact_lost_since_s = None
        else:
            if self.contact_lost_since_s is None:
                self.contact_lost_since_s = float(data.time)
            if data.time - self.contact_lost_since_s >= self.grasp_settings.contact_loss_timeout_s:
                self.fail(data, "holding_contact_lost")
                return False
        drift = np.linalg.norm(np.asarray(evidence["relative_position_m"]) - self.payload.relative_position)
        if drift > self.settings.maximum_payload_drift_m:
            self.fail(data, "payload_slipped")
            return False
        return True

    def update(self, data, evidence, verifier):
        if self.status != "running":
            return
        if self.stage == "grasp":
            self.grasp.update(data, evidence, verifier)
            if self.grasp.status in ("failed", "cancelled"):
                self.fail(data, self.grasp.reason)
            elif self.grasp.status == "succeeded":
                self.carry_targets = [*self.settings.carry_via_xy_m, self.settings.target_center_xy_m]
                self.carry_index = 0
                self.carry_height_m = evidence["cube_height_m"]
                self.carry_next(data)
            return
        if self.arm.status in ("failed", "timed_out"):
            self.fail(data, self.arm.reason)
            return
        elapsed = float(data.time) - self.phase_started_s
        if self.stage in ("carry", "lower") and not self.check_holding(data, evidence):
            return
        if self.stage == "carry" and self.arm.status == "succeeded":
            if self.carry_index + 1 < len(self.carry_targets):
                self.carry_index += 1
                self.carry_next(data)
                return
            cube_rotation = data.xmat[self.arm.model.body("cube").id].reshape(3, 3)
            extent_z = self.scene.cube_size_m / 2 * np.sum(np.abs(cube_rotation[2, :]))
            cube_z = verifier.placement.tray_top_m + extent_z + self.settings.release_clearance_m
            cube_goal = [*self.settings.target_center_xy_m, cube_z]
            self.move(data, "lower", self.tcp_for_cube(data, cube_goal), carrying=True, floor_contact=True)
        elif self.stage == "lower" and self.arm.status == "succeeded":
            self.arm.set_gripper(data, 0.08)
            self.enter(data, "release")
        elif self.stage == "release":
            if evidence["gripper_released"]:
                retreat = data.site_xpos[self.arm.site_id] + [0, 0, self.settings.retract_height_m]
                self.move(data, "retract", retreat)
            elif elapsed >= self.settings.release_timeout_s:
                self.fail(data, "release_timeout")
        elif self.stage == "retract" and self.arm.status == "succeeded":
            verifier.placement.reset_window()
            self.enter(data, "verify_place")
        elif self.stage == "verify_place":
            if evidence["placement_confirmed"]:
                self.status = "succeeded"
                self.reason = "physical_placement_verified"
                self.enter(data, "completed")
            elif elapsed >= self.settings.verify_timeout_s:
                self.fail(data, "placement_verification_timeout")
