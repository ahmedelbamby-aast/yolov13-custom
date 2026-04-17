#!/usr/bin/env python3
"""Phase3 upgrade stress gate: max workers, high prefetch, single GPU."""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

from ultralytics import YOLO

from common_artifacts import (
    HeartbeatTicker,
    append_progress_heartbeat,
    load_release_evidence,
    save_gate_json,
    save_host_runtime_profile,
    save_release_evidence,
    utc_now_iso,
)


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
        "started_at": utc_now_iso(),
        "workers": workers,
        "prefetch_factor": prefetch,
        "attempts": [],
        "max_stable_batch": None,
    }

    with HeartbeatTicker("stress_gate", interval_s=int(os.environ.get("Y13_PROGRESS_INTERVAL_S", "300"))):
        profile_path = save_host_runtime_profile("stress_gate_host_runtime_profile.json")
        report["host_runtime_profile"] = str(profile_path)

        for b in batches:
            run_name = f"stress_b{b}"
            append_progress_heartbeat("stress_gate", "attempt_start", {"batch": b, "run_name": run_name})
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
                    append_progress_heartbeat("stress_gate", "attempt_ok", {"batch": b})
                    break
            except Exception as e:
                entry["error"] = str(e)
                entry["traceback_tail"] = "\n".join(traceback.format_exc().splitlines()[-40:])
                append_progress_heartbeat("stress_gate", "attempt_fail", {"batch": b, "error": str(e)[:240]})
            report["attempts"].append(entry)

    report["ended_at"] = utc_now_iso()
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    evidence = load_release_evidence()
    gates = evidence.setdefault("gates", {})
    stress_gate = gates.setdefault("compatibility", {})
    stress_gate["status"] = "pass" if report["status"] == "ok" else "fail"
    refs = stress_gate.setdefault("evidence_refs", [])
    refs.append("stress_gate.json")
    refs.append("stress_gate_host_runtime_profile.json")
    refs.append("heartbeats/stress_gate.jsonl")
    if report["status"] != "ok":
        evidence["status"] = "blocked"
        blocking = evidence.setdefault("blocking_reasons", [])
        if "failed_stress_gate" not in blocking:
            blocking.append("failed_stress_gate")
    save_release_evidence(evidence)

    save_gate_json("stress_gate.json", report)
    print(json.dumps(report, indent=2))
    print(f"saved={OUT}")
    if report["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
