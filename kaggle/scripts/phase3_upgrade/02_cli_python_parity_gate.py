#!/usr/bin/env python3
"""Phase3 upgrade gate: CLI/Python API parity smoke on single GPU."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/kaggle/work_here/yolov13")
OUT = Path("/kaggle/working/phase3_upgrade/cli_python_parity_gate.json")
PROJECT = Path("/kaggle/working/phase3_upgrade/parity_runs")


def run_step(name: str, cmd: list[str], env: dict[str, str], cwd: Path) -> dict:
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True)
    return {
        "name": name,
        "cmd": cmd,
        "returncode": p.returncode,
        "elapsed_s": round(time.time() - t0, 3),
        "stdout_tail": "\n".join(p.stdout.splitlines()[-80:]),
        "stderr_tail": "\n".join(p.stderr.splitlines()[-80:]),
    }


def run_python_api_step(env: dict[str, str]) -> dict:
    t0 = time.time()
    status = "ok"
    err = ""
    try:
        from ultralytics import YOLO

        workers = max(1, os.cpu_count() or 1)
        model = YOLO("ultralytics/cfg/models/v13/yolov13n.yaml")
        model.train(
            data="coco8.yaml",
            task="detect",
            epochs=1,
            imgsz=64,
            batch=8,
            workers=workers,
            device="0",
            project=str(PROJECT),
            name="python_train_smoke",
            cache="ram",
            amp=True,
        )
    except Exception as e:
        status = "fail"
        err = str(e)

    best = PROJECT / "python_train_smoke" / "weights" / "best.pt"
    return {
        "name": "python_train_smoke",
        "cmd": ["python_api", "YOLO(...).train(...)"],
        "returncode": 0 if status == "ok" else 1,
        "elapsed_s": round(time.time() - t0, 3),
        "stdout_tail": "",
        "stderr_tail": err,
        "best_exists": best.exists(),
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    PROJECT.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = str(ROOT)
    env["Y13_DISABLE_FLASH"] = "0"
    env["Y13_USE_TURING_FLASH"] = "1"
    env["Y13_PREFER_FLASH4"] = "0"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    workers = max(1, os.cpu_count() or 1)
    steps = []

    steps.append(
        run_step(
            "cli_train_smoke",
            [
                sys.executable,
                "scripts/train.py",
                "--model",
                "ultralytics/cfg/models/v13/yolov13n.yaml",
                "--data",
                "coco8.yaml",
                "--task",
                "detect",
                "--epochs",
                "1",
                "--imgsz",
                "64",
                "--workers",
                str(workers),
                "--device",
                "0",
                "--project",
                str(PROJECT),
                "--name",
                "cli_train_smoke",
                "--arg",
                "batch=8",
                "--arg",
                "cache=ram",
                "--arg",
                "amp=true",
                "--flash-mode",
                "turing",
            ],
            env,
            ROOT,
        )
    )

    os.environ.update(env)
    os.chdir(ROOT)
    steps.append(run_python_api_step(env))

    best_cli = PROJECT / "cli_train_smoke" / "weights" / "best.pt"
    best_py = PROJECT / "python_train_smoke" / "weights" / "best.pt"

    status = "ok"
    if any(s["returncode"] != 0 for s in steps):
        status = "fail"
    if not best_cli.exists() or not best_py.exists():
        status = "fail"

    report = {
        "status": status,
        "workers": workers,
        "steps": steps,
        "artifacts": {
            "best_cli": str(best_cli),
            "best_python": str(best_py),
            "best_cli_exists": best_cli.exists(),
            "best_python_exists": best_py.exists(),
        },
    }

    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved={OUT}")
    if status != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
