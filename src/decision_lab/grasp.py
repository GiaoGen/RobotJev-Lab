"""Physical grasp experiment using the same Panda motion controller as reach."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform

import mujoco
import numpy as np

from decision_lab import PROJECT_ROOT
from decision_lab.recording import RunRecorder
from decision_lab.robots.panda import ControlSettings, PandaArm
from decision_lab.robots.panda_grasp import GraspSettings, PandaGrasp
from decision_lab.simulation.tabletop import TabletopSettings, create_tabletop
from decision_lab.simulation.verification import GraspVerificationSettings, GraspVerifier


class GraspExperiment:
    SCENARIOS = ("normal", "empty", "weak")
    READY_MESSAGE = "Ready. 1 normal | 2 empty | 3 weak grip."

    def __init__(self, config_path: Path, output_root: Path, *, seed=None):
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        if seed is not None:
            self.config["scene"]["seed"] = seed
        self.scene_settings = TabletopSettings(**self.config["scene"])
        self.grasp_settings = GraspSettings(**self.config["grasp"])
        verification_settings = GraspVerificationSettings(**self.config["verification"])
        control_settings = ControlSettings(**self.config["controller"])
        scene_path = PROJECT_ROOT / self.config["model_scene"]
        self.model = self.create_model(scene_path)
        self.data = mujoco.MjData(self.model)
        self.arm = PandaArm(self.model, control_settings)
        self.arm.reset(self.data)
        self.verifier = self.create_verifier(verification_settings)
        self.skill = self.create_skill()
        self.evidence = self.verifier.measure(self.data)
        self.epoch = 0
        self.step = 0
        self.paused = True
        self.next_control_time_s = 0.0
        self.next_record_time_s = 0.0
        self.scenario = "normal"
        self.execution_id = 0
        self.active_execution = None
        self.results = []
        self.released = False
        self.holding_lost = False
        self.contact_lost_since_s = None
        self.last_message = self.READY_MESSAGE
        source = json.loads((scene_path.parent / "SOURCE.json").read_text(encoding="utf-8"))
        code_files = (
            "__main__.py", "runner.py", "watchdog.py", "grasp.py", "grasp_viewer.py", "native_viewer.py", "recording.py",
            "robots/panda.py", "robots/panda_grasp.py",
            "simulation/tabletop.py", "simulation/verification.py",
        )
        code_hashes = {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in code_files
        }
        extra_manifest, extra_fields = self.recording_extensions()
        self.recorder = RunRecorder(output_root, {
            "experiment": "panda_contact_grasp_v1",
            "python": platform.python_version(), "mujoco": mujoco.mj_versionString(),
            "numpy": np.__version__, "seed": self.scene_settings.seed,
            "config": self.config, "scene": asdict(self.scene_settings),
            "grasp": asdict(self.grasp_settings), "controller": asdict(control_settings),
            "verification": asdict(verification_settings),
            "model_commit": source["commit"], "model_files": source["files"],
            "code_sha256": code_hashes,
            "physics_timestep_s": float(self.model.opt.timestep),
            "initial_qpos": self.data.qpos.tolist(),
            "initial_qvel": self.data.qvel.tolist(),
            "initial_ctrl": self.data.ctrl.tolist(),
            "grasp_center_local_position_m": [0, 0, 0.1034],
            "object_pose_source": "simulated_state_sensor_v0_exact_pose",
            "verification_source": "simulation_ground_truth",
            "equality_constraints": [int(value) for value in self.model.eq_type],
            "attachment_constraints_added": 0,
            "cube_contact_priority": 1,
            "cube_contact_dimension": int(self.model.geom_condim[self.model.geom("cube_geom").id]),
            **extra_manifest,
        }, extra_fields=(
            "execution_id", "phase", "scenario", "cube_height_m", "bottom_clearance_m",
            "left_contact_force_n", "right_contact_force_n", "gripper_opening_m",
            "relative_position_drift_m", "relative_angle_drift_rad",
            "stable_elapsed_s", "holding_confirmed", *extra_fields,
        ))
        self.recorder.attach_model(self.model)
        self.record_event("run_started")

    def create_model(self, scene_path):
        return create_tabletop(scene_path, self.scene_settings)

    def create_verifier(self, verification_settings):
        return GraspVerifier(
            self.model, self.scene_settings.table_top_m,
            self.scene_settings.cube_size_m, verification_settings,
        )

    def create_skill(self):
        return PandaGrasp(self.arm, self.grasp_settings, self.record_event)

    def recording_extensions(self):
        return {}, ()

    def prepare_scenario(self, scenario):
        pass

    def restore_scene(self):
        pass

    def physical_settings(self):
        return {
            "body_pos": self.model.body_pos.tolist(),
            "actuator_forcerange": self.model.actuator_forcerange.tolist(),
            "geom_friction": self.model.geom_friction.tolist(),
        }

    def record_event(self, event_type, **payload):
        self.recorder.event(
            event_type, self.data.time, epoch=self.epoch, step=self.step,
            execution_id=self.active_execution, **payload,
        )
        if event_type == "phase_entered":
            self.last_message = f"{self.scenario}: {payload['phase']}"
            print(f"{self.data.time:.2f}s {self.last_message}", flush=True)
        if event_type in ("run_started", "reset", "task_started", "task_finished", "run_finished",
                          "phase_entered", "pause_changed", "release_requested", "grip_force_reduced"):
            self.recorder.state(self.model, self.data, self.snapshot())

    def snapshot(self):
        position_error = None
        orientation_error = None
        target_id = None
        if self.arm.target is not None:
            target_id = self.arm.target.id
            position, orientation = self.arm.pose_error(self.data, self.arm.target)
            position_error = float(np.linalg.norm(position))
            orientation_error = float(np.linalg.norm(orientation))
        return {
            "sim_time_s": float(self.data.time), "epoch": self.epoch, "step": self.step,
            "execution_id": self.execution_id,
            "status": self.skill.status, "phase": self.skill.phase,
            "reason": self.skill.reason, "scenario": self.scenario,
            "paused": self.paused, "released": self.released,
            "holding_lost": self.holding_lost,
            "target_id": target_id,
            "actual_position_m": self.data.site_xpos[self.arm.site_id].tolist(),
            "position_error_m": position_error, "orientation_error_rad": orientation_error,
            "max_joint_speed_rad_s": float(np.max(np.abs(self.data.qvel[self.arm.dof_indices]))),
            "gripper_force_limit_n": float(self.model.actuator_forcerange[self.arm.gripper_actuator, 1]),
            **self.evidence,
        }

    def finish_execution(self):
        if self.active_execution is None:
            return
        result = {"execution_id": self.active_execution, **self.snapshot()}
        self.results.append(result)
        self.record_event("task_finished", result=result)
        self.last_message = f"{self.scenario}: {self.skill.status} ({self.skill.reason})"
        print(self.last_message, flush=True)
        self.active_execution = None

    def cancel(self):
        self.skill.cancel(self.data)
        self.finish_execution()

    def reset(self):
        self.cancel()
        self.scenario = "normal"
        self.restore_scene()
        force_limit = self.grasp_settings.normal_gripper_force_n
        self.model.actuator_forcerange[self.arm.gripper_actuator] = [-force_limit, force_limit]
        self.arm.reset(self.data)
        self.verifier.reset()
        self.skill = self.create_skill()
        self.evidence = self.verifier.measure(self.data)
        self.epoch += 1
        self.step = 0
        self.next_control_time_s = 0.0
        self.next_record_time_s = 0.0
        self.released = False
        self.holding_lost = False
        self.contact_lost_since_s = None
        self.paused = True
        self.last_message = self.READY_MESSAGE
        self.record_event("reset", physical_settings=self.physical_settings())

    def start(self, scenario="normal"):
        if scenario not in self.SCENARIOS:
            raise ValueError(f"Unknown grasp scenario: {scenario}")
        self.reset()
        self.scenario = scenario
        self.execution_id += 1
        self.active_execution = self.execution_id
        force_limit = self.grasp_settings.normal_gripper_force_n
        offset = (0.0, 0.0)
        if scenario == "weak":
            force_limit = self.grasp_settings.weak_gripper_force_n
        elif scenario == "empty":
            offset = self.grasp_settings.empty_offset_xy_m
        self.model.actuator_forcerange[self.arm.gripper_actuator] = [-force_limit, force_limit]
        self.prepare_scenario(scenario)
        mujoco.mj_forward(self.model, self.data)
        self.evidence = self.verifier.measure(self.data)
        self.record_event(
            "task_started", scenario=scenario, gripper_force_limit_n=force_limit,
            object_pose_source="simulated_state_sensor_v0_exact_pose",
            object_position_m=self.evidence["cube_position_m"], grasp_offset_xy_m=list(offset),
            physical_settings=self.physical_settings(),
        )
        self.skill.start(self.data, self.evidence["cube_position_m"], offset)
        self.paused = False
        if self.skill.status != "running":
            self.finish_execution()

    def toggle_pause(self):
        self.paused = not self.paused
        self.record_event("pause_changed", paused=self.paused)

    def release(self):
        self.cancel()
        self.arm.set_gripper(self.data, 0.08)
        self.verifier.reset_window()
        self.released = True
        self.paused = False
        self.last_message = "Release requested. Observe contacts and cube falling."
        self.record_event("release_requested", related_execution_id=self.execution_id)

    def advance(self):
        if self.paused:
            return
        if self.data.time + 1e-9 >= self.next_control_time_s:
            self.skill.update(self.data, self.evidence, self.verifier)
            self.arm.update(self.data)
            self.next_control_time_s = float(self.data.time) + self.arm.settings.control_period_s
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.step += 1
        if not (np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()):
            raise RuntimeError("Non-finite physical state. Inspect the grasp run.")
        self.evidence = self.verifier.measure(self.data)
        if (self.skill.status == "succeeded" and not self.released
                and self.arm.gripper_target < 255):
            supported = (
                self.evidence["bilateral_contact"]
                and self.evidence["bottom_clearance_m"] >= self.verifier.settings.minimum_clearance_m
            )
            if supported:
                self.contact_lost_since_s = None
            else:
                if self.contact_lost_since_s is None:
                    self.contact_lost_since_s = float(self.data.time)
                if (not self.holding_lost and self.data.time - self.contact_lost_since_s
                        >= self.grasp_settings.contact_loss_timeout_s):
                    self.holding_lost = True
                    self.last_message = "Holding lost after grasp verification. Inspect evidence."
                    self.record_event(
                        "holding_lost_after_completion", evidence=self.evidence,
                        related_execution_id=self.execution_id,
                    )
        if self.active_execution is not None:
            reason = self.evidence.get("constraint_violation_reason") or self.arm.execution_collision(self.data)
            joints = self.data.qpos[self.arm.qpos_indices]
            if np.any(joints < self.arm.lower - 0.02) or np.any(joints > self.arm.upper + 0.02):
                reason = "joint_limit_violation"
            if reason:
                self.skill.fail(self.data, reason)
                self.paused = True
        if self.active_execution is not None and self.skill.status != "running":
            self.finish_execution()
        if self.data.time + 1e-9 >= self.next_record_time_s:
            self.recorder.sample(self.snapshot())
            self.recorder.state(self.model, self.data, self.snapshot())
            self.next_record_time_s = float(self.data.time) + self.recorder.manifest["snapshot_period_s"]

    def close(self, reason):
        if self.recorder.closed:
            return
        self.cancel()
        self.record_event("run_finished", reason=reason)
        self.recorder.close({
            "reason": reason, "results": self.results, "last_state": self.snapshot(),
        })


def run_headless(experiment, scenario="normal", on_progress=None):
    experiment.start(scenario)
    while experiment.skill.status == "running":
        experiment.advance()
        if on_progress is not None:
            on_progress()
    return experiment.skill.status == "succeeded"
