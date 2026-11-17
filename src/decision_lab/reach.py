"""One simulation owner, with identical control for headless and native runs."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform

import mujoco
import numpy as np

from decision_lab.recording import RunRecorder
from decision_lab import PROJECT_ROOT
from decision_lab.robots.panda import ControlSettings, PandaArm, PoseTarget, load_panda


class ReachExperiment:
    def __init__(self, config_path: Path, output_root: Path):
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.targets = [PoseTarget(**entry) for entry in self.config["targets"]]
        if len(self.targets) != 3 or len({target.id for target in self.targets}) != 3:
            raise ValueError("This experiment requires three uniquely identified targets.")
        settings = ControlSettings(**self.config["controller"])
        scene_path = PROJECT_ROOT / self.config["model_scene"]
        self.model = load_panda(scene_path, self.targets)
        self.data = mujoco.MjData(self.model)
        self.arm = PandaArm(self.model, settings)
        self.arm.reset(self.data)
        self.step = 0
        self.epoch = 0
        self.paused = True
        self.next_control_time_s = 0.0
        self.next_record_time_s = 0.0
        self.last_message = "Ready. Select target 1, 2 or 3."
        self.execution_id = 0
        self.active_execution = None
        self.results = []
        source = json.loads((scene_path.parent / "SOURCE.json").read_text(encoding="utf-8"))
        code_hashes = {}
        for name in ["__main__.py", "runner.py", "watchdog.py", "reach.py", "native_viewer.py", "recording.py", "robots/panda.py"]:
            path = Path(__file__).parent / name
            code_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.recorder = RunRecorder(output_root, {
            "experiment": "panda_reach_v1",
            "python": platform.python_version(),
            "mujoco": mujoco.mj_versionString(),
            "numpy": np.__version__,
            "config": self.config,
            "controller": asdict(settings),
            "model_commit": source["commit"],
            "model_files": source["files"],
            "code_sha256": code_hashes,
            "grasp_center_local_position_m": [0, 0, 0.1034],
            "physics_timestep_s": float(self.model.opt.timestep),
            "initial_qpos": self.data.qpos.tolist(),
            "initial_qvel": self.data.qvel.tolist(),
            "initial_ctrl": self.data.ctrl.tolist(),
            "pose_frame": "world",
            "quaternion_order": "xyzw",
        })
        self.recorder.attach_model(self.model)
        self.record_event("run_started")

    def record_event(self, event_type, **payload):
        self.recorder.event(
            event_type, self.data.time, epoch=self.epoch, step=self.step,
            execution_id=payload.pop("execution_id", self.active_execution), **payload,
        )
        self.recorder.state(self.model, self.data, self.snapshot())

    def snapshot(self):
        quaternion_wxyz = np.empty(4)
        mujoco.mju_mat2Quat(quaternion_wxyz, self.data.site_xmat[self.arm.site_id])
        position_error = None
        orientation_error = None
        target_id = None
        if self.arm.target is not None:
            target_id = self.arm.target.id
            position, orientation = self.arm.pose_error(self.data, self.arm.target)
            position_error = float(np.linalg.norm(position))
            orientation_error = float(np.linalg.norm(orientation))
        return {
            "sim_time_s": float(self.data.time),
            "step": self.step,
            "epoch": self.epoch,
            "status": self.arm.status,
            "reason": self.arm.reason,
            "paused": self.paused,
            "target_id": target_id,
            "actual_position_m": self.data.site_xpos[self.arm.site_id].tolist(),
            "actual_quaternion_xyzw": quaternion_wxyz[[1, 2, 3, 0]].tolist(),
            "position_error_m": position_error,
            "orientation_error_rad": orientation_error,
            "max_joint_speed_rad_s": float(np.max(np.abs(
                self.data.qvel[self.arm.dof_indices],
            ))),
            "actual_joint_positions_rad": self.data.qpos[self.arm.qpos_indices].tolist(),
            "actuator_targets_rad": self.data.ctrl[self.arm.actuator_indices].tolist(),
        }

    def select_target(self, index):
        target = self.targets[index]
        reason = self.arm.start(self.data, target)
        if reason:
            self.last_message = f"Rejected {target.id}: {reason}"
            attempt = self.arm.last_ik_attempt
            residual = None
            if reason != "busy" and attempt is not None:
                residual = {
                    "iterations": attempt.iterations,
                    "position_error_m": attempt.position_error_m,
                    "orientation_error_rad": attempt.orientation_error_rad,
                }
            self.record_event(
                "command_rejected", target_id=target.id, reason=reason,
                ik_residual=residual,
            )
            return False
        self.execution_id += 1
        self.active_execution = self.execution_id
        self.paused = False
        self.last_message = f"Moving to {target.id}"
        solution = self.arm.ik_result
        self.record_event(
            "motion_started", execution_id=self.active_execution,
            target=asdict(target), ik_iterations=solution.iterations,
            ik_position_error_m=solution.position_error_m,
            ik_orientation_error_rad=solution.orientation_error_rad,
            goal_joint_positions_rad=solution.joint_positions.tolist(),
            duration_s=float(self.arm.duration_s),
        )
        return True

    def finish_execution(self):
        if self.active_execution is None:
            return
        result = {"execution_id": self.active_execution, **self.snapshot()}
        self.results.append(result)
        self.record_event("motion_finished", result=result)
        self.last_message = f"{self.arm.target.id}: {self.arm.status} ({self.arm.reason})"
        print(self.last_message, flush=True)
        self.active_execution = None

    def cancel(self):
        if self.arm.status == "moving":
            self.arm.stop(self.data, "operator_cancelled", "cancelled")
            self.finish_execution()
        self.last_message = "Cancelled. Holding current joint targets."

    def reset(self):
        self.cancel()
        self.arm.reset(self.data)
        self.step = 0
        self.epoch += 1
        self.next_control_time_s = 0.0
        self.next_record_time_s = 0.0
        self.paused = True
        self.last_message = "Reset to home. Select target 1, 2 or 3."
        self.record_event("reset")

    def toggle_pause(self):
        self.paused = not self.paused
        self.record_event("pause_changed", paused=self.paused)

    def advance(self):
        if self.paused:
            return
        if self.data.time + 1e-9 >= self.next_control_time_s:
            self.arm.update(self.data)
            self.next_control_time_s = float(self.data.time) + self.arm.settings.control_period_s
        mujoco.mj_step(self.model, self.data)
        # Refresh FK/contact state at the newly integrated qpos.
        mujoco.mj_forward(self.model, self.data)
        self.step += 1
        if not (np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()):
            raise RuntimeError("Non-finite physical state. Stop and inspect the run.")
        if self.active_execution is not None:
            collision = self.arm.collision(self.data)
            if collision:
                self.arm.stop(self.data, collision)
                self.paused = True
            joints = self.data.qpos[self.arm.qpos_indices]
            if np.any(joints < self.arm.lower - 0.02) or np.any(joints > self.arm.upper + 0.02):
                self.arm.stop(self.data, "joint_limit_violation")
                self.paused = True
        if self.active_execution is not None and self.arm.status != "moving":
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


def run_headless(experiment, on_progress=None):
    for index in range(len(experiment.targets)):
        if not experiment.select_target(index):
            return False
        while experiment.arm.status == "moving":
            experiment.advance()
            if on_progress is not None:
                on_progress()
        if experiment.arm.status != "succeeded":
            return False
    return True
