#!/usr/bin/env python3
"""Phase3 upgrade stress gate: max workers, high prefetch, single GPU."""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

from ultralytics import YOLO


OUT = Path("/kaggle/working/phase3_upgrade/stress_gate.json")
PROJECT = Path("/kaggle/working/phase3_upgrade/stress_runs")



def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    PROJECT.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("Y13_DISABLE_FLASH", "0")
    os.environ.setdefault("Y13_USE_TURING_FLASH", "1")

    workers = max(1, os.cpu_count() or 1)
    prefetch = 8
    batches = [8, 6, 4]

    report = {
        "status": "fail",
        "workers": workers,
        "prefetch_factor": prefetch,
        "attempts": [],
        "max_stable_batch": None,
    }

    for b in batches:
        run_name = f"stress_b{b}"
        entry = {"batch": b, "name": run_name, "status": "fail", "error": None}
        try:
            model = YOLO("ultralytics/cfg/models/v13/yolov13l.pt")
            model.train(
                data="coco8.yaml",
                task="detect",
                epochs=1,
                imgsz=640,
                batch=b,
                workers=workers,
                prefetch_factor=prefetch,
                persistent_workers=True,
                device="0",
                project=str(PROJECT),
                name=run_name,
                cache="ram",
                amp=True,
                classes=[0, 9],
                optimizer="SGD",
                deterministic=True,
            )
            best = PROJECT / run_name / "weights" / "best.pt"
            entry["best"] = str(best)
            entry["best_exists"] = best.exists()
            entry["status"] = "ok" if best.exists() else "fail"
            if entry["status"] == "ok":
                report["max_stable_batch"] = b
                report["attempts"].append(entry)
                report["status"] = "ok"
                break
        except Exception as e:
            entry["error"] = str(e)
            entry["traceback_tail"] = "\n".join(traceback.format_exc().splitlines()[-40:])
        report["attempts"].append(entry)

    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved={OUT}")
    if report["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
