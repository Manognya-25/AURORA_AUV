"""Child-process launcher for the real PyBullet simulation (one process per run).

PyBullet lives in its own process so a crash, a missing PyBullet install or a stop request can never
take Flask down, and so "STOP" can be verified by checking that the OS process really ended.

IPC lives in this run's private directory (--runtime); see src/simulation/run_protocol.py.
The 3D PyBullet window is controlled by `dashboard.headless` (false => GUI window), taken from the
config AFTER the dashboard's next-start overrides are merged in.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from src.simulation.run_protocol import ChildRun


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--runtime", required=True)
    ap.add_argument("--override", default=None, help="JSON file with a nested config override")
    ap.add_argument("--run-id", default="standalone")
    ap.add_argument("--parent-pid", type=int, default=None)
    args = ap.parse_args(argv)

    child = ChildRun(args.runtime, args.run_id, args.parent_pid)
    child.write_status("STARTING")
    stop_event = child.start_watchers()
    try:
        import yaml
        from src.simulation.commands import deep_merge
        override = json.loads(Path(args.override).read_text(encoding="utf-8")) if args.override else None
        cfg = deep_merge(yaml.safe_load(Path(args.config).read_text(encoding="utf-8")), override)
        headless = bool((cfg.get("dashboard", {}) or {}).get("headless", True))
        # Importing run_simulation imports pybullet: a missing install surfaces here as a real error.
        from src.simulation.run_simulation import run
        child.write_status("RUNNING")
        result = run(config_path=str(Path(args.config).resolve()), mode_override="DIRECT" if headless else "GUI",
                     stop_event=stop_event, cfg_override=override, live_dir=str(child.dir),
                     command_path=str(child.dir / "commands.jsonl"))
        child.write_status(result.get("final_status", "COMPLETED"), result=result)
        return 0
    except BaseException as exc:                       # noqa: BLE001 - reported, then non-zero exit
        child.write_status("ERROR", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1
    finally:
        child.finish()
        sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main())
