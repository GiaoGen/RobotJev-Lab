"""Physical grasp, contact policy and repeatable-state regressions."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from decision_lab import PROJECT_ROOT
from decision_lab.grasp import GraspExperiment, run_headless
from decision_lab.simulation.tabletop import TabletopSettings


class PandaGraspTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.experiment = GraspExperiment(
            PROJECT_ROOT / "configs/tasks/panda_grasp.json", Path(self.directory.name),
        )

    def tearDown(self):
        self.experiment.close("test_finished")
        self.directory.cleanup()

    def test_contact_grasp_and_physical_release(self):
        experiment = self.experiment
        self.assertTrue(run_headless(experiment))
        evidence = experiment.evidence
        self.assertTrue(evidence["holding_confirmed"])
        self.assertGreaterEqual(evidence["bottom_clearance_m"], 0.05)
        self.assertGreaterEqual(evidence["stable_elapsed_s"], 0.5)
        self.assertGreater(evidence["gripper_opening_m"], 0.03)
        self.assertEqual(experiment.model.neq, 1)
        self.assertEqual(experiment.model.eq_type[0], mujoco.mjtEq.mjEQ_JOINT)
        self.assertEqual(experiment.model.joint("cube_free").type[0], mujoco.mjtJoint.mjJNT_FREE)
        held_height = evidence["cube_height_m"]
        experiment.release()
        for _ in range(750):
            experiment.advance()
        self.assertFalse(experiment.evidence["holding_confirmed"])
        self.assertLess(experiment.evidence["cube_height_m"], held_height - 0.08)
        self.assertAlmostEqual(experiment.evidence["cube_height_m"], 0.22, delta=0.005)
        self.assertEqual(len(experiment.results), 1)
        self.assertTrue(experiment.results[0]["holding_confirmed"])

    def test_empty_closure_cannot_succeed(self):
        experiment = self.experiment
        self.assertFalse(run_headless(experiment, "empty"))
        self.assertEqual(experiment.skill.reason, "no_bilateral_contact")
        self.assertLess(experiment.evidence["gripper_opening_m"], 0.002)
        self.assertFalse(experiment.evidence["holding_confirmed"])

    def test_weak_grip_cannot_be_reported_as_holding(self):
        experiment = self.experiment
        self.assertFalse(run_headless(experiment, "weak"))
        self.assertEqual(experiment.skill.reason, "holding_contact_lost")
        self.assertFalse(experiment.evidence["holding_confirmed"])
        self.assertEqual(experiment.snapshot()["gripper_force_limit_n"], 0.3)

    def test_cube_table_contact_and_expected_finger_contacts(self):
        experiment = self.experiment
        experiment.paused = False
        for _ in range(100):
            experiment.advance()
        self.assertGreater(experiment.data.ncon, 0)
        self.assertIsNone(experiment.arm.collision(experiment.data))
        self.assertTrue(run_headless(experiment))
        self.assertIsNotNone(experiment.arm.collision(experiment.data))
        self.assertIsNone(experiment.arm.collision(
            experiment.data, experiment.skill.allowed_contacts,
        ))

    def test_pause_cancel_reset_and_scenario_switch(self):
        experiment = self.experiment
        initial = experiment.data.qpos.copy()
        experiment.start("normal")
        for _ in range(100):
            experiment.advance()
        experiment.toggle_pause()
        paused = experiment.data.qpos.copy()
        for _ in range(100):
            experiment.advance()
        np.testing.assert_array_equal(paused, experiment.data.qpos)
        experiment.cancel()
        self.assertEqual(experiment.skill.status, "cancelled")
        experiment.reset()
        np.testing.assert_array_equal(initial, experiment.data.qpos)
        np.testing.assert_array_equal(experiment.data.qvel, 0)
        self.assertEqual(experiment.arm.gripper_target, 255)
        self.assertTrue(experiment.paused)
        self.assertFalse(run_headless(experiment, "weak"))
        self.assertTrue(run_headless(experiment, "normal"))
        self.assertEqual(experiment.snapshot()["gripper_force_limit_n"], 100)

    def test_holding_depends_on_contact_friction(self):
        config = json.loads((PROJECT_ROOT / "configs/tasks/panda_grasp.json").read_text())
        config["scene"]["cube_friction"] = 0.05
        config_path = Path(self.directory.name) / "low_friction.json"
        config_path.write_text(json.dumps(config))
        experiment = GraspExperiment(config_path, Path(self.directory.name) / "low_friction")
        try:
            self.assertFalse(run_headless(experiment))
            self.assertFalse(experiment.evidence["holding_confirmed"])
            self.assertLess(experiment.evidence["bottom_clearance_m"], 0.05)
        finally:
            experiment.close("test_finished")

    def test_frictionless_contact_remains_numerically_stable(self):
        config = json.loads((PROJECT_ROOT / "configs/tasks/panda_grasp.json").read_text())
        config["scene"]["cube_friction"] = 0
        config_path = Path(self.directory.name) / "zero_friction.json"
        config_path.write_text(json.dumps(config))
        experiment = GraspExperiment(config_path, Path(self.directory.name) / "zero_friction")
        try:
            self.assertFalse(run_headless(experiment))
            self.assertTrue(np.isfinite(experiment.data.qpos).all())
            self.assertLess(experiment.evidence["cube_height_m"], 0.3)
            self.assertEqual(experiment.model.geom_condim[experiment.model.geom("cube_geom").id], 1)
        finally:
            experiment.close("test_finished")

    def test_sustained_hold_and_unrequested_loss_are_observable(self):
        experiment = self.experiment
        self.assertTrue(run_headless(experiment))
        held_height = experiment.evidence["cube_height_m"]
        for _ in range(15000):
            experiment.advance()
        self.assertTrue(experiment.evidence["holding_confirmed"])
        self.assertAlmostEqual(experiment.evidence["cube_height_m"], held_height, delta=0.003)
        experiment.model.actuator_forcerange[experiment.arm.gripper_actuator] = [-0.1, 0.1]
        for _ in range(1000):
            experiment.advance()
        self.assertTrue(experiment.holding_lost)
        self.assertFalse(experiment.evidence["holding_confirmed"])
        self.assertEqual(len(experiment.results), 1)
        self.assertTrue(experiment.results[0]["holding_confirmed"])

    def test_configuration_and_invalid_case(self):
        with self.assertRaises(ValueError):
            replace(TabletopSettings(), cube_mass_kg=-1)
        self.assertEqual(replace(TabletopSettings(), cube_friction=0).cube_friction, 0)
        experiment = self.experiment
        experiment.start("normal")
        with self.assertRaises(ValueError):
            experiment.start("unknown")
        self.assertEqual(experiment.skill.status, "running")

    def test_records_include_evidence_and_single_terminal_event(self):
        experiment = self.experiment
        self.assertTrue(run_headless(experiment))
        experiment.close("test_finished")
        events = [json.loads(line) for line in
                  (experiment.recorder.directory / "events.jsonl").read_text().splitlines()]
        self.assertEqual(sum(event["type"] == "task_finished" for event in events), 1)
        result = next(event["result"] for event in events if event["type"] == "task_finished")
        self.assertTrue(result["holding_confirmed"])
        self.assertGreater(result["left_contact_force_n"], 0)
        telemetry = (experiment.recorder.directory / "telemetry.csv").read_text()
        self.assertIn("bottom_clearance_m", telemetry)
        self.assertIn("stable_elapsed_s", telemetry)
        self.assertIn("epoch", telemetry)
        self.assertIn("execution_id", telemetry)


if __name__ == "__main__":
    unittest.main()
