"""Interruptible grasp phases, composed from the existing Panda pose control."""

from dataclasses import dataclass

import numpy as np

from decision_lab.robots.panda import PoseTarget


@dataclass(frozen=True)
class GraspSettings:
    approach_height_m: float = 0.12
    lift_height_m: float = 0.12
    close_timeout_s: float = 1.5
    contact_window_s: float = 0.15
    contact_loss_timeout_s: float = 0.15
    verify_timeout_s: float = 2.0
    normal_gripper_force_n: float = 100.0
    weak_gripper_force_n: float = 0.3
    empty_offset_xy_m: tuple[float, float] = (0.0, 0.09)

    def __post_init__(self):
        for name, value in vars(self).items():
            if name == "empty_offset_xy_m":
                vector = np.asarray(value, dtype=float)
                if vector.shape != (2,) or not np.isfinite(vector).all():
                    raise ValueError("empty_offset_xy_m must contain two finite numbers.")
            elif not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite.")


class PandaGrasp:
    def __init__(self, arm, settings, on_event):
        self.arm = arm
        self.settings = settings
        self.on_event = on_event
        cube_id = arm.model.body("cube").id
        self.allowed_contacts = frozenset(
            frozenset((cube_id, arm.model.body(name).id))
            for name in ("left_finger", "right_finger")
        )
        self.status = "ready"
        self.phase = "ready"
        self.reason = "home"
        self.phase_started_s = 0.0
        self.contact_since_s = None
        self.contact_lost_since_s = None
        self.grasp_position = np.zeros(3)

    def enter_phase(self, data, phase):
        self.phase = phase
        self.phase_started_s = float(data.time)
        self.on_event("phase_entered", phase=phase)

    def move(self, data, phase, position):
        target = PoseTarget(phase, tuple(position), (0.7071067811865476, 0.7071067811865476, 0, 0))
        reason = self.arm.start(data, target, self.allowed_contacts)
        if reason:
            attempt = self.arm.last_ik_attempt
            self.on_event(
                "motion_rejected", phase=phase, reason=reason,
                ik_position_error_m=None if attempt is None else attempt.position_error_m,
                ik_orientation_error_rad=None if attempt is None else attempt.orientation_error_rad,
            )
            self.fail(data, f"{phase}:{reason}")
            return
        self.enter_phase(data, phase)
        self.on_event(
            "motion_started", phase=phase, target_position_m=list(position),
            ik_position_error_m=self.arm.ik_result.position_error_m,
            ik_orientation_error_rad=self.arm.ik_result.orientation_error_rad,
        )

    def start(self, data, object_position_m, offset_xy_m=(0.0, 0.0)):
        if self.status == "running":
            return "busy"
        self.grasp_position = np.asarray(object_position_m, dtype=float).copy()
        self.grasp_position[:2] += np.asarray(offset_xy_m)
        self.contact_since_s = None
        self.contact_lost_since_s = None
        self.status = "running"
        self.reason = "executing"
        self.arm.set_gripper(data, 0.08)
        approach = self.grasp_position + [0, 0, self.settings.approach_height_m]
        self.move(data, "approach", approach)
        return None

    def fail(self, data, reason):
        self.arm.stop(data, reason)
        self.status = "failed"
        self.reason = reason
        self.enter_phase(data, "failed")

    def cancel(self, data):
        if self.status == "running":
            self.arm.stop(data, "operator_cancelled", "cancelled")
            self.status = "cancelled"
            self.reason = "operator_cancelled"
            self.enter_phase(data, "cancelled")

    def update(self, data, evidence, verifier):
        if self.status != "running":
            return
        if self.arm.status in ("failed", "timed_out"):
            self.fail(data, self.arm.reason)
            return
        elapsed = float(data.time) - self.phase_started_s
        if self.phase == "approach" and self.arm.status == "succeeded":
            self.move(data, "descend", self.grasp_position)
        elif self.phase == "descend" and self.arm.status == "succeeded":
            self.arm.set_gripper(data, 0.0)
            self.enter_phase(data, "close")
        elif self.phase == "close":
            if evidence["bilateral_contact"]:
                if self.contact_since_s is None:
                    self.contact_since_s = float(data.time)
                if data.time - self.contact_since_s >= self.settings.contact_window_s:
                    lift = self.grasp_position + [0, 0, self.settings.lift_height_m]
                    self.move(data, "lift", lift)
            else:
                self.contact_since_s = None
            if self.phase == "close" and elapsed >= self.settings.close_timeout_s:
                self.fail(data, "no_bilateral_contact")
        elif self.phase in ("lift", "verify_hold"):
            if evidence["bilateral_contact"]:
                self.contact_lost_since_s = None
            else:
                if self.contact_lost_since_s is None:
                    self.contact_lost_since_s = float(data.time)
                if data.time - self.contact_lost_since_s >= self.settings.contact_loss_timeout_s:
                    self.fail(data, "holding_contact_lost")
                    return
            if self.phase == "lift" and self.arm.status == "succeeded":
                verifier.reset_window()
                self.enter_phase(data, "verify_hold")
            elif self.phase == "verify_hold":
                if evidence["holding_confirmed"]:
                    self.status = "succeeded"
                    self.reason = "physical_grasp_verified"
                    self.enter_phase(data, "held")
                elif elapsed >= self.settings.verify_timeout_s:
                    self.fail(data, "hold_verification_timeout")
