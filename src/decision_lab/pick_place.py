"""Pick/place task assembly sharing the grasp experiment's simulation owner."""

from dataclasses import asdict
import hashlib
from pathlib import Path

import mujoco

from decision_lab.grasp import GraspExperiment
from decision_lab.robots.panda_pick_place import PandaPickPlace
from decision_lab.simulation.placement import PlacementSettings, PickPlaceVerifier, create_pick_place_scene


class PickPlaceExperiment(GraspExperiment):
    SCENARIOS = ("normal", "blocked", "drop", "empty")
    READY_MESSAGE = "Ready. 1 pick/place | 2 blocked path | 3 induced grip loss."

    def create_model(self, scene_path):
        self.placement_settings = PlacementSettings(**self.config["placement"])
        self.drop_triggered = False
        return create_pick_place_scene(scene_path, self.scene_settings, self.placement_settings)

    def create_verifier(self, verification_settings):
        return PickPlaceVerifier(
            self.model, self.scene_settings, verification_settings, self.placement_settings,
        )

    def create_skill(self):
        return PandaPickPlace(
            self.arm, self.grasp_settings, self.placement_settings,
            self.scene_settings, self.record_event,
        )

    def recording_extensions(self):
        paths = ("pick_place.py", "pick_place_viewer.py", "robots/panda_pick_place.py", "simulation/placement.py")
        hashes = {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in paths
        }
        return {
            "experiment": "panda_pick_place_v1",
            "placement": asdict(self.placement_settings), "task_code_sha256": hashes,
        }, (
            "inside_target_region", "tray_supported", "tray_support_force_n",
            "cube_linear_speed_m_s", "cube_angular_speed_rad_s",
            "gripper_released", "gripper_retracted", "placement_stable_s", "placement_confirmed",
            "constraint_violated",
        )

    def prepare_scenario(self, scenario):
        self.drop_triggered = False
        xy = self.placement_settings.obstacle_parked_xy_m
        if scenario == "blocked":
            xy = self.placement_settings.obstacle_blocked_xy_m
        self.model.body_pos[self.model.body("path_obstacle").id, :2] = xy
        mujoco.mj_forward(self.model, self.data)
        self.record_event("obstacle_configured", position_xy_m=list(xy), scenario=scenario)

    def restore_scene(self):
        self.model.body_pos[self.model.body("path_obstacle").id, :2] = self.placement_settings.obstacle_parked_xy_m
        self.drop_triggered = False

    def advance(self):
        if (not self.paused and self.scenario == "drop" and not self.drop_triggered
                and self.skill.stage == "carry" and self.skill.status == "running"
                and self.data.time - self.skill.phase_started_s >= self.placement_settings.drop_delay_s):
            force = self.placement_settings.drop_gripper_force_n
            previous = float(self.model.actuator_forcerange[self.arm.gripper_actuator, 1])
            self.model.actuator_forcerange[self.arm.gripper_actuator] = [-force, force]
            self.drop_triggered = True
            self.record_event(
                "grip_force_reduced", previous_force_limit_n=previous,
                force_limit_n=force, phase="carry", physical_settings=self.physical_settings(),
            )
        super().advance()

    def reset(self):
        super().reset()
        self.drop_triggered = False
