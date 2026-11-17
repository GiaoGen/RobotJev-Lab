"""Grasp keys and evidence display for the shared native-viewer adapter."""

from decision_lab.native_viewer import run_viewer


def grasp_key(experiment, key):
    if key in (49, 50, 51):
        experiment.start(experiment.SCENARIOS[key - 49])
    elif key == 32:
        experiment.toggle_pause()
    elif key == 82:
        experiment.reset()
    elif key == 67:
        experiment.cancel()
    elif key == 79:
        experiment.release()
    return key != 81


def grasp_overlay(experiment, state):
    return [
        "PANDA - PHYSICAL CONTACT GRASP",
        "1 normal | 2 empty | 3 weak grip | O open/release",
        "SPACE pause | R reset | C cancel | F focus | V overview | Q quit",
        experiment.last_message,
        f"Phase: {state['phase']} | Result: {state['status']} | Paused: {state['paused']}",
        f"Sim time: {state['sim_time_s']:.2f} s | Force limit: {state['gripper_force_limit_n']:.1f} N",
        f"Cube bottom clearance: {1000 * state['bottom_clearance_m']:.1f} mm "
        f"(required >= {1000 * experiment.verifier.settings.minimum_clearance_m:.0f})",
        f"Finger contact L/R: {state['left_contact_force_n']:.2f} / {state['right_contact_force_n']:.2f} N",
        f"Bilateral contact: {state['bilateral_contact']} | Finger opening: {1000 * state['gripper_opening_m']:.1f} mm",
        f"Relative drift: {1000 * state['relative_position_drift_m']:.2f} mm",
        f"Stable window: {state['stable_elapsed_s']:.2f} / {experiment.verifier.settings.stable_window_s:.2f} s",
        f"Holding now confirmed: {state['holding_confirmed']} | Release requested: {state['released']}",
        f"Unrequested holding loss: {state['holding_lost']}",
    ]


def run_native(experiment, on_progress=None):
    run_viewer(experiment, grasp_key, grasp_overlay, on_progress)
