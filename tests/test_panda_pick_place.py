"""Loaded geometry, placement truth, terminal states and prior-skill reuse."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from decision_lab import PROJECT_ROOT
from decision_lab.grasp import run_headless
from decision_lab.pick_place import PickPlaceExperiment
from decision_lab.robots.panda import PoseTarget
from decision_lab.simulation.placement import PlacementSettings


class PandaPickPlaceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.experiment = PickPlaceExperiment(
            PROJECT_ROOT / "configs/tasks/panda_pick_place.json", Path(self.directory.name),
        )

    def tearDown(self):
        self.experiment.close("test_finished")
        self.directory.cleanup()

    def set_cube(self, position):
        experiment = self.experiment
        address = experiment.model.joint("cube_free").qposadr[0]
        experiment.data.qpos[address:address + 7] = [*position, 1, 0, 0, 0]
        experiment.data.qvel[:] = 0
        experiment.data.qacc_warmstart[:] = 0
        mujoco.mj_forward(experiment.model, experiment.data)
        return experiment.verifier.measure(experiment.data)

    def test_complete_task_requires_physical_placement(self):
        experiment = self.experiment
        self.assertTrue(run_headless(experiment))
        state = experiment.snapshot()
        for field in ("inside_target_region", "tray_supported", "gripper_released",
                      "gripper_retracted", "placement_confirmed"):
            self.assertTrue(state[field], field)
        self.assertFalse(state["holding_confirmed"])
        self.assertFalse(state["holding_lost"])
        self.assertGreaterEqual(state["placement_stable_s"], 0.5)
        self.assertLessEqual(state["cube_linear_speed_m_s"], 0.02)
        self.assertAlmostEqual(state["tray_support_force_n"], 0.08 * 9.81, delta=0.01)
        self.assertEqual(len(experiment.results), 1)
        self.assertEqual(experiment.skill.reason, "physical_placement_verified")
        self.assertEqual(experiment.model.neq, 1)

    def test_empty_grasp_terminates_the_complete_pick_place_task(self):
        experiment = self.experiment
        self.assertFalse(run_headless(experiment, "empty"))
        self.assertEqual(experiment.skill.status, "failed")
        self.assertEqual(experiment.skill.reason, "no_bilateral_contact")
        self.assertFalse(experiment.evidence["placement_confirmed"])
        self.assertEqual(len(experiment.results), 1)

    def test_payload_changes_collision_result_without_changing_live_world(self):
        experiment = self.experiment
        self.assertFalse(run_headless(experiment, "blocked"))
        self.assertEqual(experiment.skill.reason, "carry:collision:cube/path_obstacle")
        qpos = experiment.data.qpos.copy()
        qvel = experiment.data.qvel.copy()
        goal = experiment.arm.last_ik_attempt.joint_positions
        self.assertIsNone(experiment.arm.check_path(experiment.data, goal, experiment.skill.allowed_contacts))
        payload = experiment.arm.capture_payload(experiment.data, experiment.model.body("cube").id)
        self.assertEqual(experiment.arm.check_path(
            experiment.data, goal, experiment.skill.allowed_contacts, payload=payload,
        ), "collision:cube/path_obstacle")
        np.testing.assert_array_equal(qpos, experiment.data.qpos)
        np.testing.assert_array_equal(qvel, experiment.data.qvel)

    def test_drop_is_a_failure_and_keeps_its_physical_consequence(self):
        experiment = self.experiment
        self.assertFalse(run_headless(experiment, "drop"))
        self.assertTrue(experiment.drop_triggered)
        self.assertEqual(experiment.skill.reason, "payload_slipped")
        for _ in range(1000):
            experiment.advance()
        self.assertFalse(experiment.evidence["placement_confirmed"])
        self.assertAlmostEqual(experiment.evidence["cube_height_m"], 0.22, delta=0.005)
        self.assertEqual(len(experiment.results), 1)
        self.assertEqual(experiment.results[0]["status"], "failed")
        experiment.reset()
        self.assertFalse(experiment.drop_triggered)
        self.assertEqual(experiment.snapshot()["gripper_force_limit_n"], 100)

    def test_configured_alternate_route_uses_the_same_controller(self):
        config = json.loads((PROJECT_ROOT / "configs/tasks/panda_pick_place.json").read_text())
        config["placement"]["carry_via_xy_m"] = [[0.58, -0.05], [0.58, 0.18]]
        path = Path(self.directory.name) / "alternate.json"
        path.write_text(json.dumps(config))
        experiment = PickPlaceExperiment(path, Path(self.directory.name) / "alternate")
        try:
            self.assertTrue(run_headless(experiment, "blocked"))
            self.assertTrue(experiment.evidence["placement_confirmed"])
        finally:
            experiment.close("test_finished")

    def test_pause_cancel_reset_and_repeated_execution(self):
        experiment = self.experiment
        initial = experiment.data.qpos.copy()
        experiment.start()
        while experiment.skill.stage != "carry":
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
        self.assertIsNone(experiment.arm.payload)
        self.assertFalse(experiment.verifier.constraint_violated)
        self.assertTrue(run_headless(experiment))
        self.assertEqual(len(experiment.results), 2)

    def test_center_inside_with_footprint_outside_is_rejected(self):
        settings = self.experiment.placement_settings
        state = self.set_cube((settings.target_center_xy_m[0] + 0.065, 0.18, 0.229892))
        self.assertFalse(state["inside_target_region"])
        self.assertFalse(state["placement_confirmed"])

    def test_airborne_object_inside_region_is_rejected(self):
        state = self.set_cube((0.48, 0.18, 0.4))
        self.assertTrue(state["inside_target_region"])
        self.assertFalse(state["tray_supported"])
        self.assertFalse(state["placement_confirmed"])

    def test_closed_gripper_cannot_complete_placement(self):
        experiment = self.experiment
        state = self.set_cube((0.48, 0.18, 0.229892))
        self.assertTrue(state["tray_supported"])
        indices = experiment.verifier.holding.finger_joints
        experiment.data.qpos[indices] = 0
        mujoco.mj_forward(experiment.model, experiment.data)
        state = experiment.verifier.measure(experiment.data)
        self.assertFalse(state["gripper_released"])
        self.assertFalse(state["placement_confirmed"])

    def test_gripper_must_withdraw(self):
        experiment = self.experiment
        self.set_cube((0.48, 0.18, 0.229892))
        target = PoseTarget("near_cube", (0.48, 0.18, 0.235), (0.7071067811865476, 0.7071067811865476, 0, 0))
        solution = experiment.arm.solve_ik(experiment.data, target)
        self.assertTrue(solution.converged)
        experiment.data.qpos[experiment.arm.qpos_indices] = solution.joint_positions
        mujoco.mj_forward(experiment.model, experiment.data)
        state = experiment.verifier.measure(experiment.data)
        self.assertFalse(state["gripper_retracted"])
        self.assertFalse(state["placement_confirmed"])

    def test_moving_cube_cannot_complete_placement(self):
        experiment = self.experiment
        self.set_cube((0.48, 0.18, 0.229892))
        velocity_address = experiment.model.joint("cube_free").dofadr[0]
        experiment.data.qvel[velocity_address] = 1
        mujoco.mj_forward(experiment.model, experiment.data)
        state = experiment.verifier.measure(experiment.data)
        self.assertGreater(state["cube_linear_speed_m_s"], 0.02)
        self.assertFalse(state["placement_confirmed"])

    def test_forbidden_contact_is_latched_until_full_reset(self):
        experiment = self.experiment
        forbidden = experiment.model.body_pos[experiment.model.body("forbidden").id]
        state = self.set_cube(forbidden)
        self.assertTrue(state["constraint_violated"])
        experiment.verifier.reset_window()
        state = self.set_cube((0.48, 0.18, 0.229892))
        self.assertTrue(state["constraint_violated"])
        self.assertFalse(state["placement_confirmed"])
        experiment.verifier.reset()
        state = experiment.verifier.measure(experiment.data)
        self.assertFalse(state["constraint_violated"])

    def test_placement_configuration_and_record_identity(self):
        with self.assertRaises(ValueError):
            replace(PlacementSettings(), carry_via_xy_m=((float("nan"), 0),))
        with self.assertRaises(ValueError):
            replace(PlacementSettings(), released_opening_m=0.1)
        experiment = self.experiment
        self.assertTrue(run_headless(experiment))
        experiment.close("test_finished")
        events = [json.loads(line) for line in
                  (experiment.recorder.directory / "events.jsonl").read_text().splitlines()]
        self.assertEqual(sum(event["type"] == "task_finished" for event in events), 1)
        phases = [event["phase"] for event in events if event["type"] == "phase_entered"]
        self.assertEqual(phases[-6:], ["carry", "lower", "release", "retract", "verify_place", "completed"])
        result = next(event["result"] for event in events if event["type"] == "task_finished")
        self.assertTrue(result["placement_confirmed"])


if __name__ == "__main__":
    unittest.main()
