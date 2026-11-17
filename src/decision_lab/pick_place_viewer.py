"""Pick/place controls and physical placement evidence."""

from decision_lab.grasp_viewer import grasp_key
from decision_lab.native_viewer import run_viewer


def overlay(experiment, state):
    return [
        "PANDA - PICK, CARRY AND PLACE",
        "1 normal | 2 blocked path | 3 induced grip loss | O manual release",
        "SPACE pause | R reset | C cancel | F focus | V overview | Q quit",
        experiment.last_message,
        f"Phase: {state['phase']} | Result: {state['status']} | Sim: {state['sim_time_s']:.2f}s",
        f"Finger L/R: {state['left_contact_force_n']:.2f} / {state['right_contact_force_n']:.2f} N",
        f"Cube bottom clearance: {1000 * state['bottom_clearance_m']:.1f} mm",
        f"Whole cube inside target: {state['inside_target_region']}",
        f"Tray support: {state['tray_supported']} ({state['tray_support_force_n']:.2f} N)",
        f"Cube speed: {1000 * state['cube_linear_speed_m_s']:.2f} mm/s",
        f"Gripper released: {state['gripper_released']} | Retracted: {state['gripper_retracted']}",
        f"Placement stable: {state['placement_stable_s']:.2f} / {experiment.placement_settings.stable_window_s:.2f}s",
        f"Placement confirmed: {state['placement_confirmed']}",
        f"Forbidden contact recorded: {state['constraint_violated']}",
    ]


def run_native(experiment, on_progress=None):
    run_viewer(experiment, grasp_key, overlay, on_progress)
