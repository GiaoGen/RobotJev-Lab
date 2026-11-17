"""CLI worker. One process owns the model, data, controller and recorder."""

import signal
import traceback

from decision_lab.watchdog import WorkerProgress


def run_experiment(args, connection, stop_requested):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    progress = WorkerProgress(connection, stop_requested)
    experiment = None
    reason = "exception"
    code = 1
    try:
        progress()
        if args.command == "reach":
            from decision_lab.reach import ReachExperiment, run_headless
            from decision_lab.native_viewer import run_native
            experiment = ReachExperiment(args.config, args.output)
        else:
            from decision_lab.grasp import GraspExperiment, run_headless
            if args.command == "pick_place":
                from decision_lab.pick_place import PickPlaceExperiment
                from decision_lab.pick_place_viewer import run_native
                experiment = PickPlaceExperiment(args.config, args.output, seed=args.seed)
            else:
                from decision_lab.grasp_viewer import run_native
                experiment = GraspExperiment(args.config, args.output, seed=args.seed)
        progress.directory = experiment.recorder.directory
        progress(force=True)
        print(f"Run records: {progress.directory}", flush=True)
        if args.headless:
            if args.command == "reach":
                succeeded = run_headless(experiment, on_progress=progress)
            else:
                succeeded = run_headless(experiment, args.scenario, on_progress=progress)
            reason = "completed" if succeeded else "failed"
            code = 0 if succeeded else 1
        else:
            run_native(experiment, on_progress=progress)
            reason = "closed"
            code = 0
    except KeyboardInterrupt:
        reason = "interrupted"
        code = 130
    except Exception:
        traceback.print_exc()
    finally:
        if experiment is not None:
            experiment.close(reason)
    connection.send({"kind": "finished", "exit_code": code})
    connection.close()
