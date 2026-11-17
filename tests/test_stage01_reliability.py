"""Regression coverage for reset, durable states and Windows process supervision."""

import json
import os
from pathlib import Path
import tempfile
import time
import unittest

import mujoco
import numpy as np

from decision_lab import PROJECT_ROOT
from decision_lab.grasp import GraspExperiment, run_headless
from decision_lab.pick_place import PickPlaceExperiment
from decision_lab.recording import recover_run
from decision_lab.watchdog import WorkerProgress, run_guarded


def fault_worker(output, fault, connection, stop_requested):
    progress = WorkerProgress(connection, stop_requested)
    progress()
    experiment = GraspExperiment(PROJECT_ROOT / "configs/tasks/panda_grasp.json", Path(output))
    progress.directory = experiment.recorder.directory
    experiment.start()
    for _ in range(100):
        experiment.advance()
    progress(force=True)
    if fault == "crash":
        os._exit(3)
    if fault == "stall":
        experiment.recorder.events.write('{"event_seq":')
        experiment.recorder.events.flush()
        experiment.recorder.states.write('{"state_seq":')
        experiment.recorder.states.flush()
        time.sleep(30)
        return
    experiment.toggle_pause()
    until = time.monotonic() + 0.8
    while time.monotonic() < until:
        experiment.advance()
        progress()
        time.sleep(0.02)
    experiment.close("test_finished")
    connection.send({"kind": "finished", "exit_code": 0})
    connection.close()


class Stage01ReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_seeded_initialization_and_full_reset(self):
        config = json.loads((PROJECT_ROOT / "configs/tasks/panda_pick_place.json").read_text())
        config["scene"] = {"cube_position_jitter_xy_m": [0.005, 0.005]}
        path = self.root / "seeded.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        experiments = [PickPlaceExperiment(path, self.root / str(i), seed=seed)
                       for i, seed in enumerate((41, 41, 42))]
        try:
            first, same, different = experiments
            np.testing.assert_array_equal(first.data.qpos, same.data.qpos)
            self.assertFalse(np.array_equal(first.data.qpos, different.data.qpos))
            initial = first.data.qpos.copy()
            initial_ctrl = first.data.ctrl.copy()
            body_pos = first.model.body_pos.copy()
            force_range = first.model.actuator_forcerange.copy()
            specification = mujoco.mjtState.mjSTATE_INTEGRATION
            initial_state = np.empty(mujoco.mj_stateSize(first.model, specification))
            mujoco.mj_getState(first.model, first.data, initial_state, specification)
            first.start("blocked")
            np.testing.assert_array_equal(first.data.ctrl, initial_ctrl)
            for _ in range(100):
                first.advance()
            first.reset()
            reset_state = np.empty_like(initial_state)
            mujoco.mj_getState(first.model, first.data, reset_state, specification)
            np.testing.assert_array_equal(reset_state, initial_state)
            np.testing.assert_array_equal(first.data.qpos, initial)
            np.testing.assert_array_equal(first.data.qvel, 0)
            np.testing.assert_array_equal(first.data.ctrl, initial_ctrl)
            np.testing.assert_array_equal(first.model.body_pos, body_pos)
            np.testing.assert_array_equal(first.model.actuator_forcerange, force_range)
            self.assertEqual(first.scenario, "normal")
            self.assertIsNone(first.arm.payload)
            self.assertFalse(first.verifier.constraint_violated)
        finally:
            for experiment in experiments:
                experiment.close("test_finished")

    def test_weak_case_reset_restores_normal_force(self):
        experiment = GraspExperiment(PROJECT_ROOT / "configs/tasks/panda_grasp.json", self.root)
        try:
            experiment.start("weak")
            self.assertEqual(experiment.snapshot()["gripper_force_limit_n"], 0.3)
            experiment.reset()
            self.assertEqual(experiment.snapshot()["gripper_force_limit_n"], 100)
            self.assertEqual(experiment.scenario, "normal")
        finally:
            experiment.close("test_finished")

    def test_saved_states_reconstruct_body_poses_and_terminal_frame(self):
        experiment = PickPlaceExperiment(PROJECT_ROOT / "configs/tasks/panda_pick_place.json", self.root)
        self.assertFalse(run_headless(experiment, "drop"))
        experiment.close("test_finished")
        directory = experiment.recorder.directory
        model = mujoco.MjModel.from_binary_path(str(directory / "model.mjb"))
        data = mujoco.MjData(model)
        events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
        states = [json.loads(line) for line in (directory / "states.jsonl").read_text().splitlines()]
        self.assertGreater(len(states), 150)
        self.assertEqual(states[-1]["status"], "failed")
        self.assertEqual(states[-1]["reason"], "payload_slipped")
        event_index = 0
        for index, state in enumerate(states, 1):
            self.assertEqual(state["state_seq"], index)
            self.assertEqual(state["run_id"], directory.name)
            while event_index < len(events) and events[event_index]["event_seq"] <= state["event_seq"]:
                physical = events[event_index].get("physical_settings", {})
                for field, value in physical.items():
                    getattr(model, field)[:] = value
                event_index += 1
            mujoco.mj_setState(model, data, np.asarray(state["integration_state"]), mujoco.mjtState.mjSTATE_INTEGRATION)
            mujoco.mj_forward(model, data)
            np.testing.assert_allclose(data.xpos, state["body_positions_m"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(data.xquat[:, [1, 2, 3, 0]], state["body_quaternions_xyzw"], atol=1e-12, rtol=0)
            np.testing.assert_array_equal(data.qpos, state["qpos"])
        summary = json.loads((directory / "summary.json").read_text())
        self.assertTrue(summary["recording_complete"])
        self.assertEqual(summary["state_count"], len(states))
        self.assertEqual(model.actuator_forcerange[experiment.arm.gripper_actuator, 1], 0.1)

    def test_stalled_worker_is_stopped_and_torn_logs_are_recovered_once(self):
        code = run_guarded(fault_worker, (str(self.root), "stall"), stall_timeout_s=0.4, startup_timeout_s=15)
        self.assertEqual(code, 124)
        directory = next(self.root.iterdir())
        recover_run(directory, "wall_clock_timeout")
        summary = json.loads((directory / "summary.json").read_text())
        self.assertFalse(summary["recording_complete"])
        self.assertTrue(summary["recovered_by_supervisor"])
        self.assertEqual(len(summary["results"]), 1)
        self.assertEqual(summary["results"][0]["status"], "aborted")
        self.assertEqual(summary["results"][0]["physical_outcome"], "unknown")
        events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
        self.assertEqual(sum(event["type"] == "task_finished" for event in events), 1)
        self.assertEqual(sum(event["type"] == "run_finished" for event in events), 1)
        self.assertEqual([event["event_seq"] for event in events], list(range(1, len(events) + 1)))
        self.assertTrue((directory / "states.jsonl.incomplete").exists())
        self.assertTrue((directory / "events.jsonl.incomplete").exists())

    def test_worker_crash_is_reported_with_durable_evidence(self):
        code = run_guarded(fault_worker, (str(self.root), "crash"), startup_timeout_s=15)
        self.assertEqual(code, 1)
        summary = json.loads((next(self.root.iterdir()) / "summary.json").read_text())
        self.assertEqual(summary["reason"], "worker_process_exited")
        self.assertEqual(summary["results"][0]["status"], "aborted")

    def test_live_paused_worker_keeps_heartbeats(self):
        code = run_guarded(fault_worker, (str(self.root), "pause"), stall_timeout_s=0.5, startup_timeout_s=15)
        self.assertEqual(code, 0)
        summary = json.loads((next(self.root.iterdir()) / "summary.json").read_text())
        self.assertTrue(summary["recording_complete"])


if __name__ == "__main__":
    unittest.main()
