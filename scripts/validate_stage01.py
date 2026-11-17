"""Run the stage-01 physical acceptance checks and preserve every case's evidence."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import time

import mujoco
import numpy as np

from decision_lab import PROJECT_ROOT
from decision_lab.grasp import GraspExperiment, run_headless
from decision_lab.pick_place import PickPlaceExperiment
from decision_lab.watchdog import WorkerProgress, run_guarded


def stalled_simulation(output, connection, stop_requested):
    progress = WorkerProgress(connection, stop_requested)
    progress()
    experiment = PickPlaceExperiment(PROJECT_ROOT / "configs/tasks/panda_pick_place.json", Path(output))
    progress.directory = experiment.recorder.directory
    experiment.start()
    for _ in range(100):
        experiment.advance()
    experiment.record_event("validation_fault_injected", fault="worker_stall")
    progress(force=True)
    time.sleep(30)


class Stage01Acceptance:
    def __init__(self, directory):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=False)
        self.checks = []
        self.runs = []

    def check(self, group, name, passed, **evidence):
        self.checks.append({"group": group, "name": name, "passed": bool(passed), **evidence})
        print(f"{'PASS' if passed else 'FAIL'} {group} {name}", flush=True)

    def config(self, name, task="pick_place", **sections):
        config = json.loads((PROJECT_ROOT / f"configs/tasks/panda_{task}.json").read_text(encoding="utf-8"))
        for section, values in sections.items():
            config[section].update(values)
        path = self.directory / f"{name}.json"
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        return path

    def run_case(self, name, *, task="pick_place", scenario="normal", config=None, seed=None):
        cls = PickPlaceExperiment if task == "pick_place" else GraspExperiment
        path = config or PROJECT_ROOT / f"configs/tasks/panda_{task}.json"
        experiment = cls(path, self.directory / name, seed=seed)
        try:
            run_headless(experiment, scenario)
            result = experiment.snapshot()
        finally:
            experiment.close("acceptance_finished")
            self.runs.append(experiment.recorder.directory)
        return result

    def normal_and_resets(self):
        experiment = PickPlaceExperiment(
            PROJECT_ROOT / "configs/tasks/panda_pick_place.json", self.directory / "normal_repeated",
        )
        initial = experiment.data.qpos.copy()
        initial_ctrl = experiment.data.ctrl.copy()
        initial_body_pos = experiment.model.body_pos.copy()
        initial_forces = experiment.model.actuator_forcerange.copy()
        specification = mujoco.mjtState.mjSTATE_INTEGRATION
        initial_state = np.empty(mujoco.mj_stateSize(experiment.model, specification))
        mujoco.mj_getState(experiment.model, experiment.data, initial_state, specification)
        current_state = np.empty_like(initial_state)
        reset_equal = []
        try:
            for _ in range(10):
                experiment.start()
                mujoco.mj_getState(experiment.model, experiment.data, current_state, specification)
                reset_equal.append(
                    np.array_equal(initial, experiment.data.qpos)
                    and np.array_equal(initial_ctrl, experiment.data.ctrl)
                    and np.array_equal(experiment.data.qvel, np.zeros(experiment.model.nv))
                    and experiment.data.time == 0
                    and np.array_equal(initial_state, current_state)
                )
                while experiment.skill.status == "running":
                    experiment.advance()
            successes = sum(result["status"] == "succeeded" for result in experiment.results)
            self.check("A02", "normal_pick_place_10_trials", successes == 10,
                       trials=10, successes=successes, run=str(experiment.recorder.directory))
            self.check("A03", "same_configuration_reset_10_times", all(reset_equal), comparisons=reset_equal)
            for scenario in ("blocked", "drop"):
                run_headless(experiment, scenario)
                experiment.reset()
                mujoco.mj_getState(experiment.model, experiment.data, current_state, specification)
                restored = (
                    np.array_equal(experiment.data.qpos, initial)
                    and np.array_equal(experiment.data.qvel, np.zeros(experiment.model.nv))
                    and np.array_equal(experiment.data.ctrl, initial_ctrl)
                    and np.array_equal(experiment.model.body_pos, initial_body_pos)
                    and np.array_equal(experiment.model.actuator_forcerange, initial_forces)
                    and experiment.arm.payload is None
                    and not experiment.verifier.constraint_violated
                    and experiment.scenario == "normal"
                    and np.array_equal(initial_state, current_state)
                )
                self.check("A03", f"restore_after_{scenario}", restored)
        finally:
            experiment.close("acceptance_finished")
            self.runs.append(experiment.recorder.directory)

    def physical_parameters_and_failures(self):
        mass_probes = []
        for mass in (0.08, 0.16):
            config = self.config(f"mass_{mass}", scene={"cube_mass_kg": mass})
            experiment = PickPlaceExperiment(config, self.directory / f"mass_{mass}")
            try:
                experiment.paused = False
                for _ in range(500):
                    experiment.advance()
                cube = experiment.model.geom("cube_geom").id
                table = experiment.model.geom("tabletop").id
                support_force = 0.0
                force = np.zeros(6)
                for index, contact in enumerate(experiment.data.contact):
                    if {contact.geom1, contact.geom2} == {cube, table}:
                        mujoco.mj_contactForce(experiment.model, experiment.data, index, force)
                        support_force += max(0.0, float(force[0]))
                experiment.record_event("parameter_probe", cube_mass_kg=mass, table_support_force_n=support_force)
                run_headless(experiment)
                mass_probes.append({"mass_kg": mass, "table_support_force_n": support_force,
                                    "task_result": experiment.snapshot()})
            finally:
                experiment.close("acceptance_finished")
                self.runs.append(experiment.recorder.directory)
        self.check("A01", "mass_changes_measured_support", all(
            abs(probe["table_support_force_n"] - probe["mass_kg"] * 9.81) < 0.01
            for probe in mass_probes
        ) and abs(mass_probes[1]["table_support_force_n"] / mass_probes[0]["table_support_force_n"] - 2) < 0.02,
                   probes=mass_probes,
                   criterion="Settled tabletop support matches mass times gravity for both masses.")
        friction_config = self.config("low_friction", task="grasp", scene={"cube_friction": 0.05})
        friction = self.run_case("low_friction", task="grasp", config=friction_config)
        self.check("A01", "low_friction_prevents_verified_hold", friction["status"] == "failed"
                   and not friction["holding_confirmed"], result=friction)
        for name, task, scenario, expected in (
            ("empty", "pick_place", "empty", "no_bilateral_contact"),
            ("weak", "grasp", "weak", "holding_contact_lost"),
            ("blocked", "pick_place", "blocked", "carry:collision:cube/path_obstacle"),
            ("drop", "pick_place", "drop", "payload_slipped"),
        ):
            result = self.run_case(name, task=task, scenario=scenario)
            self.check("A02", name, result["status"] == "failed" and result["reason"] == expected, result=result)
        detour = self.run_case("detour", scenario="blocked",
                               config=PROJECT_ROOT / "configs/tasks/panda_pick_place_detour.json")
        self.check("A02", "configured_detour", detour["status"] == "succeeded", result=detour)
        timeout_config = self.config("tracking_timeout", controller={
            "position_tolerance_m": 1e-12, "settling_timeout_s": 0.1,
        })
        timeout = self.run_case("tracking_timeout", config=timeout_config)
        self.check("A02", "simulation_tracking_timeout", timeout["status"] == "failed"
                   and timeout["reason"] == "tracking_timeout", result=timeout)

    def forbidden_contact(self):
        experiment = PickPlaceExperiment(PROJECT_ROOT / "configs/tasks/panda_pick_place.json",
                                         self.directory / "forbidden_contact")
        try:
            experiment.start()
            address = experiment.model.joint("cube_free").qposadr[0]
            experiment.data.qpos[address:address + 3] = experiment.model.body_pos[
                experiment.model.body("forbidden").id
            ]
            mujoco.mj_forward(experiment.model, experiment.data)
            experiment.evidence = experiment.verifier.measure(experiment.data)
            experiment.record_event("validation_fault_injected", fault="cube_overlaps_forbidden")
            experiment.recorder.state(experiment.model, experiment.data, experiment.snapshot())
            experiment.advance()
            result = experiment.snapshot()
            self.check("A02", "forbidden_contact_stops_execution", result["status"] == "failed"
                       and result["reason"] == "forbidden_contact" and result["constraint_violated"], result=result)
        finally:
            experiment.close("acceptance_finished")
            self.runs.append(experiment.recorder.directory)

    def seeds(self):
        config = self.config("seeded", scene={"cube_position_jitter_xy_m": [0.005, 0.005]})
        initial = []
        outcomes = []
        for index, seed in enumerate((41, 41, 42)):
            result = self.run_case(f"seed_{index}_{seed}", config=config, seed=seed)
            manifest = json.loads((self.runs[-1] / "manifest.json").read_text())
            initial.append(manifest["initial_qpos"])
            outcomes.append(result["status"])
        self.check("A03", "seeded_initial_state_reproducibility", initial[0] == initial[1]
                   and initial[0] != initial[2] and all(value == "succeeded" for value in outcomes),
                   seeds=[41, 41, 42], initial_qpos=initial, outcomes=outcomes,
                   jitter_half_range_m=[0.005, 0.005])

    def cli_and_watchdog(self):
        for command, scenario, expected in (("reach", "normal", 0), ("pick_place", "normal", 0),
                                            ("pick_place", "drop", 1), ("pick_place", "empty", 1)):
            output = self.directory / f"cli_{command}_{scenario}"
            result = subprocess.run([
                sys.executable, "-m", "decision_lab", command, "--headless", "--scenario", scenario,
                "--output", str(output),
            ], cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
            (self.directory / f"cli_{command}_{scenario}.log").write_text(
                result.stdout + result.stderr, encoding="utf-8",
            )
            self.check("A02", f"cli_{command}_{scenario}", result.returncode == expected,
                       exit_code=result.returncode, expected_exit_code=expected)
            if output.exists():
                self.runs.extend(path for path in output.iterdir() if path.is_dir())
        output = self.directory / "stalled_worker"
        code = run_guarded(stalled_simulation, (str(output),), stall_timeout_s=0.5, startup_timeout_s=15)
        directory = next(output.iterdir())
        self.runs.append(directory)
        summary = json.loads((directory / "summary.json").read_text())
        self.check("A02", "wall_clock_watchdog", code == 124 and summary["reason"] == "wall_clock_timeout"
                   and not summary["recording_complete"] and summary["results"][0]["status"] == "aborted",
                   exit_code=code, run=str(directory), result=summary["results"][0])

    def records(self):
        maximum_pose_error = 0.0
        total_states = 0
        for directory in self.runs:
            manifest = json.loads((directory / "manifest.json").read_text())
            summary = json.loads((directory / "summary.json").read_text())
            events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
            states = [json.loads(line) for line in (directory / "states.jsonl").read_text().splitlines()]
            model = mujoco.MjModel.from_binary_path(str(directory / "model.mjb"))
            data = mujoco.MjData(model)
            event_index = 0
            pose_error = 0.0
            maximum_gap_s = 0.0
            for index, state in enumerate(states):
                while event_index < len(events) and events[event_index]["event_seq"] <= state["event_seq"]:
                    for name, value in events[event_index].get("physical_settings", {}).items():
                        getattr(model, name)[:] = value
                    event_index += 1
                mujoco.mj_setState(model, data, np.asarray(state["integration_state"]), mujoco.mjtState.mjSTATE_INTEGRATION)
                mujoco.mj_forward(model, data)
                pose_error = max(pose_error, float(np.max(np.abs(data.xpos - state["body_positions_m"]))))
                if index and state["epoch"] == states[index - 1]["epoch"]:
                    maximum_gap_s = max(maximum_gap_s, state["sim_time_s"] - states[index - 1]["sim_time_s"])
            event_sequence_ok = [event["event_seq"] for event in events] == list(range(1, len(events) + 1))
            state_sequence_ok = [state["state_seq"] for state in states] == list(range(1, len(states) + 1))
            terminals = [event for event in events if event["type"] in ("task_finished", "motion_finished")]
            terminal_ids = [event.get("execution_id", event["result"]["execution_id"]) for event in terminals]
            unique_terminals = len(set(terminal_ids)) == len(terminal_ids)
            compiled_hash = hashlib.sha256((directory / "model.mjb").read_bytes()).hexdigest()
            self.check("A03", f"record_integrity_{directory.parent.name}", event_sequence_ok and state_sequence_ok
                       and unique_terminals and summary["state_count"] == len(states)
                       and compiled_hash == manifest["compiled_model_sha256"]
                       and pose_error < 1e-12 and maximum_gap_s <= 0.052,
                       run=str(directory), state_count=len(states), maximum_pose_error_m=pose_error,
                       maximum_snapshot_gap_s=maximum_gap_s, recording_complete=summary["recording_complete"])
            maximum_pose_error = max(maximum_pose_error, pose_error)
            total_states += len(states)
        self.record_statistics = {"runs": len(self.runs), "states": total_states,
                                  "maximum_reconstructed_position_error_m": maximum_pose_error}

    def run(self):
        source_path = PROJECT_ROOT / "assets/robots/franka_emika_panda/SOURCE.json"
        source = json.loads(source_path.read_text())
        mismatches = [entry["path"] for entry in source["files"] if hashlib.sha256(
            (source_path.parent / entry["path"]).read_bytes()
        ).hexdigest() != entry["sha256"]]
        self.check("A01", "official_model_integrity", not mismatches,
                   verified_files=len(source["files"]), mismatches=mismatches, commit=source["commit"])
        self.normal_and_resets()
        self.physical_parameters_and_failures()
        self.forbidden_contact()
        self.seeds()
        self.cli_and_watchdog()
        self.records()
        with (self.directory / "regression_tests.log").open("w", encoding="utf-8") as stream:
            result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                                    cwd=PROJECT_ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=180)
        self.check("regression", "all_existing_and_reliability_tests", result.returncode == 0,
                   log=str(self.directory / "regression_tests.log"))
        report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "stage_01_physical_foundation_A01_A03",
            "passed": all(check["passed"] for check in self.checks),
            "groups": {group: all(check["passed"] for check in self.checks if check["group"] == group)
                       for group in ("A01", "A02", "A03", "regression")},
            "A04": {"status": "deferred_by_user", "deliverable": "browser_playground"},
            "environment": {"python_executable": sys.executable, "python": sys.version,
                            "packages": {name: version(name) for name in ("decision-lab", "mujoco", "numpy")},
                            "lock_sha256": hashlib.sha256((PROJECT_ROOT / "uv.lock").read_bytes()).hexdigest()},
            "validation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "assumptions": {"object_pose_source": "simulated_state_sensor_v0_exact_pose",
                            "routes": "explicit_configured_waypoints",
                            "seed_test_condition": "cube XY uniform jitter within 5 mm of nominal position"},
            "record_statistics": self.record_statistics,
            "checks": self.checks,
        }
        path = self.directory / "acceptance.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Acceptance report: {path}", flush=True)
        return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    directory = args.output or PROJECT_ROOT / "runs/stage01_acceptance" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    return Stage01Acceptance(directory.resolve()).run()


if __name__ == "__main__":
    raise SystemExit(main())
