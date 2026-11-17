"""Regressions for state isolation, real pose verification and cancellation."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from decision_lab.reach import PROJECT_ROOT, ReachExperiment, run_headless
from decision_lab.robots.panda import ControlSettings, PandaArm, PoseTarget, load_panda


class PandaReachTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_path = PROJECT_ROOT / "configs/tasks/panda_reach.json"
        config = json.loads(cls.config_path.read_text(encoding="utf-8"))
        cls.targets = [PoseTarget(**entry) for entry in config["targets"]]
        cls.model = load_panda(PROJECT_ROOT / config["model_scene"], cls.targets)

    def setUp(self):
        self.data = mujoco.MjData(self.model)
        self.arm = PandaArm(self.model, ControlSettings())
        self.arm.reset(self.data)

    def test_ik_preserves_actual_state(self):
        before = (self.data.qpos.copy(), self.data.qvel.copy(), self.data.ctrl.copy())
        result = self.arm.solve_ik(self.data, self.targets[0])
        self.assertTrue(result.converged)
        self.assertLessEqual(result.position_error_m, 0.0002)
        self.assertIsNone(self.arm.check_path(self.data, result.joint_positions))
        for original, actual in zip(before, (self.data.qpos, self.data.qvel, self.data.ctrl)):
            np.testing.assert_array_equal(original, actual)
        self.assertEqual(self.data.time, 0)

    def test_quaternion_sign_represents_same_orientation(self):
        target = self.targets[0]
        equivalent = replace(target, quaternion_xyzw=tuple(-np.asarray(target.quaternion_xyzw)))
        first = self.arm.solve_ik(self.data, target)
        second = self.arm.solve_ik(self.data, equivalent)
        self.assertTrue(first.converged and second.converged)
        np.testing.assert_allclose(first.joint_positions, second.joint_positions, atol=1e-8)

    def test_unreachable_target_is_bounded_and_rejected(self):
        far_away = replace(self.targets[0], position_m=(3, 0, 0.3))
        reason = self.arm.start(self.data, far_away)
        self.assertEqual(reason, "ik_not_converged")
        self.assertEqual(self.arm.status, "ready")
        self.assertEqual(self.arm.last_ik_attempt.iterations, 250)
        self.assertEqual(self.data.time, 0)

    def test_collision_path_is_rejected(self):
        reason = self.arm.check_path(self.data, self.arm.lower)
        self.assertIsNotNone(reason)
        self.assertTrue(reason.startswith("collision:"))

    def test_invalid_target_and_settings(self):
        with self.assertRaises(ValueError):
            replace(self.targets[0], quaternion_xyzw=(0, 0, 0, 0))
        with self.assertRaises(ValueError):
            replace(self.targets[0], position_m=(np.nan, 0, 0))
        with self.assertRaises(ValueError):
            ControlSettings(damping=0)

    def test_three_real_physical_targets_and_records(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment = ReachExperiment(self.config_path, Path(directory))
            try:
                self.assertTrue(run_headless(experiment))
                self.assertEqual(len(experiment.results), 3)
                for result in experiment.results:
                    self.assertEqual(result["status"], "succeeded")
                    self.assertLessEqual(result["position_error_m"], 0.005)
                    self.assertLessEqual(result["orientation_error_rad"], 0.035)
            finally:
                experiment.close("test_finished")
            summary = json.loads((experiment.recorder.directory / "summary.json").read_text())
            self.assertEqual(len(summary["results"]), 3)
            events = [json.loads(line) for line in
                      (experiment.recorder.directory / "events.jsonl").read_text().splitlines()]
            self.assertEqual([event["event_seq"] for event in events], list(range(1, len(events) + 1)))

    def test_pause_busy_cancel_and_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment = ReachExperiment(self.config_path, Path(directory))
            try:
                self.assertTrue(experiment.select_target(0))
                self.assertFalse(experiment.select_target(1))
                for _ in range(100):
                    experiment.advance()
                experiment.toggle_pause()
                paused_position = experiment.data.qpos.copy()
                for _ in range(20):
                    experiment.advance()
                np.testing.assert_array_equal(paused_position, experiment.data.qpos)
                experiment.cancel()
                self.assertEqual(experiment.arm.status, "cancelled")
                experiment.reset()
                self.assertEqual(experiment.epoch, 1)
                self.assertEqual(experiment.data.time, 0)
                self.assertTrue(experiment.paused)
                np.testing.assert_array_equal(experiment.data.qvel, 0)
                self.assertTrue(experiment.select_target(2))
                while experiment.arm.status == "moving":
                    experiment.advance()
                self.assertEqual(experiment.arm.status, "succeeded")
            finally:
                experiment.close("test_finished")

    def test_tracking_timeout_has_terminal_state(self):
        settings = replace(ControlSettings(), position_tolerance_m=1e-12, settling_timeout_s=0.1)
        arm = PandaArm(self.model, settings)
        arm.reset(self.data)
        self.assertIsNone(arm.start(self.data, self.targets[0]))
        for step in range(4000):
            if step % 10 == 0:
                arm.update(self.data)
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            if arm.status != "moving":
                break
        self.assertEqual(arm.status, "timed_out")
        self.assertEqual(arm.reason, "tracking_timeout")


if __name__ == "__main__":
    unittest.main()
