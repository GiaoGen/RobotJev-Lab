import argparse
import math
from pathlib import Path

from decision_lab import PROJECT_ROOT
from decision_lab.runner import run_experiment
from decision_lab.watchdog import run_guarded


def main():
    parser = argparse.ArgumentParser(description="Panda physical manipulation experiments")
    parser.add_argument("command", choices=["reach", "grasp", "pick_place"])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scenario", choices=["normal", "empty", "weak", "blocked", "drop"], default="normal")
    parser.add_argument("--seed", type=int, help="Override the tabletop initialization seed")
    parser.add_argument("--stall-timeout", type=float, default=30.0, help="Maximum seconds without a worker heartbeat")
    parser.add_argument("--startup-timeout", type=float, default=60.0, help="Maximum seconds without progress during startup")
    args = parser.parse_args()
    if any(not math.isfinite(value) or value <= 0 for value in (args.stall_timeout, args.startup_timeout)):
        parser.error("Timeouts must be positive and finite.")
    if args.seed is not None and (args.seed < 0 or args.command == "reach"):
        parser.error("--seed must be nonnegative and applies to grasp and pick_place.")
    scenarios = {"reach": ("normal",), "grasp": ("normal", "empty", "weak"),
                 "pick_place": ("normal", "blocked", "drop", "empty")}
    if args.scenario not in scenarios[args.command]:
        parser.error(f"{args.command} supports scenarios {scenarios[args.command]}.")
    args.config = (args.config or PROJECT_ROOT / f"configs/tasks/panda_{args.command}.json").resolve()
    args.output = (args.output or PROJECT_ROOT / f"runs/panda_{args.command}").resolve()
    return run_guarded(
        run_experiment, (args,), stall_timeout_s=args.stall_timeout,
        startup_timeout_s=args.startup_timeout,
    )


if __name__ == "__main__":
    raise SystemExit(main())
