"""Keyboard/overlay adapter; simulation and control live in ReachExperiment."""

from queue import SimpleQueue
import time

import mujoco
import mujoco.viewer
import numpy as np


def add_frame(scene, position, rotation, length):
    colors = ([1, 0.2, 0.2, 1], [0.2, 1, 0.2, 1], [0.2, 0.5, 1, 1])
    for axis, color in enumerate(colors):
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
            np.eye(3).ravel(), np.asarray(color, dtype=np.float32),
        )
        mujoco.mjv_connector(
            geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.003,
            position, position + length * rotation[:, axis],
        )
        scene.ngeom += 1


def run_viewer(experiment, handle_key, overlay_lines, on_progress=None):
    commands = SimpleQueue()
    # The render thread only queues keys. The calling thread owns all physics.
    with mujoco.viewer.launch_passive(
        experiment.model, experiment.data, key_callback=commands.put,
        show_left_ui=False, show_right_ui=False,
    ) as viewer:
        focus_tcp = False
        with viewer.lock():
            viewer.cam.lookat[:] = [0.3, 0, 0.35]
            viewer.cam.distance = 1.6
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -25
        while viewer.is_running():
            started = time.perf_counter()
            while not commands.empty():
                key = commands.get()
                if key == 70:
                    focus_tcp = True
                elif key == 86:
                    focus_tcp = False
                    with viewer.lock():
                        viewer.cam.lookat[:] = [0.3, 0, 0.35]
                        viewer.cam.distance = 1.6
                if not handle_key(experiment, key):
                    return
            # Five 2 ms physics steps per display update (~100 Hz).
            for _ in range(5):
                experiment.advance()
            state = experiment.snapshot()
            actual = np.asarray(state["actual_position_m"])
            actual_rotation = experiment.data.site_xmat[experiment.arm.site_id].reshape(3, 3)
            target = experiment.arm.target
            with viewer.lock():
                if focus_tcp:
                    viewer.cam.lookat[:] = actual
                    viewer.cam.distance = 0.45
                viewer.user_scn.ngeom = 0
                add_frame(viewer.user_scn, np.zeros(3), np.eye(3), 0.12)
                add_frame(viewer.user_scn, actual, actual_rotation, 0.07)
                if target is not None:
                    rotation = np.empty(9)
                    mujoco.mju_quat2Mat(rotation, target.quaternion_wxyz)
                    add_frame(
                        viewer.user_scn, np.asarray(target.position_m),
                        rotation.reshape(3, 3), 0.10,
                    )
            viewer.set_texts((
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_TOPLEFT, "\n".join(overlay_lines(experiment, state)), "",
            ))
            viewer.sync()
            if on_progress is not None:
                on_progress()
            time.sleep(max(0.0, 5 * experiment.model.opt.timestep - (time.perf_counter() - started)))


def reach_key(experiment, key):
    if key in (49, 50, 51):
        experiment.select_target(key - 49)
    elif key == 32:
        experiment.toggle_pause()
    elif key == 82:
        experiment.reset()
    elif key == 67:
        experiment.cancel()
    return key != 81


def reach_overlay(experiment, state):
    actual = state["actual_position_m"]
    lines = [
        "PANDA - CARTESIAN REACH",
        "1/2/3 select target | SPACE pause | R reset | C cancel | Q quit",
        experiment.last_message,
        f"State: {state['status']} | Paused: {state['paused']} | Sim: {state['sim_time_s']:.2f} s",
        f"Actual XYZ (m): {actual[0]:.3f}, {actual[1]:.3f}, {actual[2]:.3f}",
    ]
    target = experiment.arm.target
    if target is not None:
        lines += [
            f"Target {target.id} XYZ (m): " + ", ".join(f"{value:.3f}" for value in target.position_m),
            f"Physical error: {1000 * state['position_error_m']:.2f} mm / "
            f"{np.degrees(state['orientation_error_rad']):.2f} deg",
            f"IK residual: {1000 * experiment.arm.ik_result.position_error_m:.3f} mm / "
            f"{np.degrees(experiment.arm.ik_result.orientation_error_rad):.3f} deg",
        ]
    return lines


def run_native(experiment, on_progress=None):
    run_viewer(experiment, reach_key, reach_overlay, on_progress)
