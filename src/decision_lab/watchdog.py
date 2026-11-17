"""Supervise the simulation process without sharing its mutable physics state."""

import multiprocessing
from pathlib import Path
import time

from decision_lab.recording import recover_run


class WorkerProgress:
    def __init__(self, connection, stop_requested):
        self.connection = connection
        self.stop_requested = stop_requested
        self.directory = None
        self.last_sent = float("-inf")

    def __call__(self, force=False):
        if self.stop_requested.is_set():
            raise KeyboardInterrupt
        now = time.monotonic()
        if force or now - self.last_sent >= 0.25:
            self.connection.send({
                "kind": "progress",
                "directory": None if self.directory is None else str(self.directory),
            })
            self.last_sent = now


def run_guarded(target, arguments, *, stall_timeout_s=30.0, startup_timeout_s=60.0):
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    stop_requested = context.Event()
    process = context.Process(target=target, args=(*arguments, sender, stop_requested))
    process.start()
    sender.close()
    last_progress = time.monotonic()
    directory = None
    exit_code = None
    reason = None
    interrupt_deadline = None
    pipe_open = True
    try:
        while True:
            try:
                while pipe_open:
                    try:
                        if not receiver.poll():
                            break
                        message = receiver.recv()
                    except (EOFError, OSError):
                        pipe_open = False
                        break
                    if message["kind"] == "progress":
                        last_progress = time.monotonic()
                        if message["directory"] is not None:
                            directory = Path(message["directory"])
                    elif message["kind"] == "finished":
                        exit_code = message["exit_code"]
                if not process.is_alive():
                    process.join()
                    if pipe_open:
                        continue
                    break
                now = time.monotonic()
                timeout = startup_timeout_s if directory is None else stall_timeout_s
                if interrupt_deadline is not None and now >= interrupt_deadline:
                    reason = "operator_interrupted"
                    break
                if interrupt_deadline is None and now - last_progress > timeout:
                    reason = "startup_timeout" if directory is None else "wall_clock_timeout"
                    break
                process.join(timeout=0.05)
            except KeyboardInterrupt:
                stop_requested.set()
                interrupt_deadline = time.monotonic() + 2.0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
            if process.is_alive():
                process.kill()
                process.join()
        else:
            process.join()
        receiver.close()
    if reason is None and (exit_code is None or process.exitcode != 0):
        reason = "worker_process_exited"
    if reason is not None:
        if directory is not None:
            recover_run(directory, reason)
        print(f"Simulation supervisor stopped the run: {reason}", flush=True)
        if reason == "operator_interrupted":
            return 130
        return 124 if reason.endswith("timeout") else 1
    return exit_code
