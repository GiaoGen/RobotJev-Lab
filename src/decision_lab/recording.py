"""Small, inspectable run records shared by CLI and native-viewer experiments."""

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

import mujoco
import numpy as np

from decision_lab import PROJECT_ROOT

class RunRecorder:
    def __init__(self, output_root: Path, manifest: dict, extra_fields=()):
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.directory = output_root / f"{run_id}_{uuid.uuid4().hex[:8]}"
        self.directory.mkdir(parents=True)
        self.sequence = 0
        self.closed = False
        self.extra_fields = tuple(extra_fields)
        self.manifest = {
            **manifest, "record_schema_version": 1, "run_id": self.directory.name,
            "snapshot_period_s": 0.05, "state_spec": "mjSTATE_INTEGRATION",
            "world_up": "z", "quaternion_order": "xyzw", "units": "metre-kilogram-second-radian",
            "dependency_lock_sha256": hashlib.sha256((PROJECT_ROOT / "uv.lock").read_bytes()).hexdigest(),
        }
        self.write_json("manifest.json", self.manifest)
        self.events = (self.directory / "events.jsonl").open("w", encoding="utf-8")
        self.states = (self.directory / "states.jsonl").open("w", encoding="utf-8")
        self.state_sequence = 0
        self.telemetry = (self.directory / "telemetry.csv").open(
            "w", encoding="utf-8", newline="",
        )
        self.writer = csv.writer(self.telemetry)
        self.writer.writerow([
            "sim_time_s", "epoch", "step", "status", "target_id", "actual_x_m", "actual_y_m",
            "actual_z_m", "position_error_m", "orientation_error_rad",
            "max_joint_speed_rad_s", *self.extra_fields,
        ])

    def write_json(self, name, value):
        path = self.directory / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def attach_model(self, model):
        """Keep the compiled scene so recorded states retain their interpretation."""
        path = self.directory / "model.mjb"
        mujoco.mj_saveModel(model, str(path), None)
        self.manifest.update({
            "compiled_model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "body_names": [model.body(i).name for i in range(model.nbody)],
            "nq": model.nq, "nv": model.nv, "nu": model.nu,
        })
        self.write_json("manifest.json", self.manifest)

    def state(self, model, data, snapshot):
        specification = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(model, specification))
        mujoco.mj_getState(model, data, state, specification)
        self.state_sequence += 1
        entry = {
            **snapshot, "run_id": self.directory.name,
            "state_seq": self.state_sequence, "event_seq": self.sequence,
            "integration_state": state.tolist(), "qpos": data.qpos.tolist(),
            "qvel": data.qvel.tolist(), "ctrl": data.ctrl.tolist(),
            "body_positions_m": data.xpos.tolist(),
            "body_quaternions_xyzw": data.xquat[:, [1, 2, 3, 0]].tolist(),
        }
        self.states.write(json.dumps(entry, allow_nan=False) + "\n")
        self.states.flush()
        return self.state_sequence

    def event(self, event_type, sim_time_s, **payload):
        self.sequence += 1
        event = {
            "run_id": self.directory.name,
            "event_seq": self.sequence,
            "wall_time_utc": datetime.now(timezone.utc).isoformat(),
            "sim_time_s": float(sim_time_s),
            "type": event_type,
            **payload,
        }
        self.events.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        self.events.flush()

    def sample(self, snapshot):
        self.writer.writerow([
            snapshot["sim_time_s"], snapshot["epoch"], snapshot["step"],
            snapshot["status"], snapshot["target_id"],
            *snapshot["actual_position_m"], snapshot["position_error_m"],
            snapshot["orientation_error_rad"], snapshot["max_joint_speed_rad_s"],
            *(snapshot[field] for field in self.extra_fields),
        ])
        self.telemetry.flush()

    def close(self, summary):
        if self.closed:
            return
        self.telemetry.flush()
        self.events.close()
        self.telemetry.close()
        self.states.close()
        self.write_json("summary.json", {
            **summary, "run_id": self.directory.name, "record_schema_version": 1,
            "event_count": self.sequence, "state_count": self.state_sequence,
            "recording_complete": True,
        })
        self.closed = True


def recover_run(directory, reason):
    """Finalize durable evidence only after the supervisor has stopped the worker."""
    def read_records(name):
        path = directory / name
        records = []
        lines = path.read_bytes().splitlines(keepends=True)
        for index, line in enumerate(lines):
            try:
                records.append(json.loads(line))
            except (ValueError, UnicodeDecodeError):
                path.with_suffix(path.suffix + ".incomplete").write_bytes(b"".join(lines[index:]))
                path.write_bytes(b"".join(lines[:index]))
                break
        if records and not path.read_bytes().endswith(b"\n"):
            with path.open("ab") as stream:
                stream.write(b"\n")
        return records

    events = read_records("events.jsonl")
    states = read_records("states.jsonl")
    last_state = states[-1] if states else None
    results = [event["result"] for event in events
               if event["type"] in ("task_finished", "motion_finished")]
    finished = {result["execution_id"] for result in results}
    starts = [event for event in events if event["type"] in ("task_started", "motion_started")]
    pending = [event for event in starts if event.get("execution_id") not in finished]
    sequence = max((event["event_seq"] for event in events), default=0)

    def append_event(event_type, **payload):
        nonlocal sequence
        sequence += 1
        event = {
            "run_id": directory.name,
            "event_seq": sequence, "wall_time_utc": datetime.now(timezone.utc).isoformat(),
            "sim_time_s": None if last_state is None else last_state["sim_time_s"],
            "epoch": None if last_state is None else last_state["epoch"],
            "step": None if last_state is None else last_state["step"],
            "type": event_type, "source": "supervisor_recovery", **payload,
        }
        with (directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")

    # Grasp skills emit several motion_started events within one execution.
    pending_ids = sorted({event["execution_id"] for event in pending
                          if event.get("execution_id") is not None})
    for execution_id in pending_ids:
        result = {
            "execution_id": execution_id, "status": "aborted", "reason": reason,
            "physical_outcome": "unknown", "last_durable_state_seq": (
                None if last_state is None else last_state["state_seq"]
            ),
        }
        results.append(result)
        terminal = "task_finished" if any(event["type"] == "task_started" for event in starts) else "motion_finished"
        append_event(terminal, execution_id=execution_id, result=result)
    if not any(event["type"] == "run_finished" for event in events):
        append_event("run_finished", reason=reason)
    summary = {
        "run_id": directory.name, "record_schema_version": 1, "reason": reason,
        "recording_complete": False, "recovered_by_supervisor": True,
        "results": results, "last_state": last_state,
        "last_state_source": "last_durable_snapshot", "event_count": sequence,
        "state_count": len(states),
    }
    temporary = directory / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(directory / "summary.json")
