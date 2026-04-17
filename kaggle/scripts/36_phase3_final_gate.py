#!/usr/bin/env python3
"""Phase3 final integration gate across core workflows."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from phase3_upgrade.common_artifacts import (
    HeartbeatTicker,
    append_progress_heartbeat,
    load_release_evidence,
    normalized_step_result,
    save_gate_json,
    save_host_runtime_profile,
    save_release_evidence,
    utc_now_iso,
)


def step(result: dict, name: str, fn):
    started = utc_now_iso()
    t0 = time.time()
    append_progress_heartbeat("phase3_final_gate", "step_start", {"step": name})
    try:
        out = fn()
        result["steps"][name] = normalized_step_result(
            name=name,
            status="ok",
            started_at=started,
            ended_at=utc_now_iso(),
            details={"elapsed_s": round(time.time() - t0, 3), "output": out},
        )
        append_progress_heartbeat("phase3_final_gate", "step_ok", {"step": name})
    except Exception as e:
        result["steps"][name] = normalized_step_result(
            name=name,
            status="fail",
            started_at=started,
            ended_at=utc_now_iso(),
            details={"elapsed_s": round(time.time() - t0, 3)},
            error=str(e)[:1200],
        )
        result["status"] = "fail"
        append_progress_heartbeat("phase3_final_gate", "step_fail", {"step": name, "error": str(e)[:240]})


def main() -> None:
    os.environ.setdefault("Y13_DISABLE_FLASH", "0")
    os.environ.setdefault("Y13_USE_TURING_FLASH", "0")

    from ultralytics import YOLO

    result = {
        "phase": "phase3",
        "gate": "final_integration",
        "status": "ok",
        "started_at": utc_now_iso(),
        "steps": {},
    }

    profile_path = save_host_runtime_profile("phase3_final_gate_host_runtime_profile.json")
    result["host_runtime_profile"] = str(profile_path)

    project = Path("/kaggle/working/phase3_final_gate")
    run_name = "train_detect_1e"
    best = project / run_name / "weights" / "best.pt"

    def train_step():
        YOLO("ultralytics/cfg/models/v13/yolov13n.yaml").train(
            data="coco8.yaml",
            task="detect",
            epochs=1,
            imgsz=64,
            batch=8,
            device="0",
            workers=2,
            project=str(project),
            name=run_name,
            exist_ok=True,
            optimizer="MuSGD",
        )
        return {"best": str(best)}

    def val_step():
        m = YOLO(str(best))
        r = m.val(data="coco8.yaml", task="detect", imgsz=64, batch=8, device="0", split="val")
        return {"metric_keys": sorted((getattr(r, "results_dict", {}) or {}).keys())}

    def predict_step():
        m = YOLO(str(best))
        src = "/kaggle/work_here/datasets/coco8/images/val"
        r = m.predict(source=src, imgsz=64, device="0", conf=0.25, save=True, project=str(project), name="predict")
        return {"results_len": len(r)}

    def export_step():
        m = YOLO(str(best))
        out = m.export(format="onnx", imgsz=64, batch=1, device="0", dynamic=True)
        return {"artifact": str(out)}

    def benchmark_step():
        out_json = Path("/kaggle/working/phase3_final_gate_benchmark.json")
        cmd = [
            "/kaggle/work_here/yolov13/.venv/bin/python",
            "scripts/benchmark.py",
            "--model",
            str(best),
            "--data",
            "coco8.yaml",
            "--imgsz",
            "64",
            "--device",
            "0",
            "--flash-mode",
            "turing",
            "--format",
            "onnx",
            "--format",
            "engine",
            "--out-json",
            str(out_json),
        ]
        subprocess.run(cmd, check=True)
        data = json.loads(out_json.read_text(encoding="utf-8"))
        results = data.get("results", [])
        failed = [r for r in results if r.get("status") != "ok"]
        if failed:
            raise RuntimeError(f"Benchmark failures: {failed}")
        return {
            "flash_backend": data.get("flash_backend"),
            "device": data.get("device"),
            "formats": [r.get("format") for r in results],
        }

    with HeartbeatTicker("phase3_final_gate", interval_s=int(os.environ.get("Y13_PROGRESS_INTERVAL_S", "300"))):
        step(result, "train", train_step)
        if result["status"] == "ok":
            step(result, "val", val_step)
        if result["status"] == "ok":
            step(result, "predict", predict_step)
        if result["status"] == "ok":
            step(result, "export", export_step)
        if result["status"] == "ok":
            step(result, "benchmark", benchmark_step)

    result["ok_steps"] = sum(1 for s in result["steps"].values() if s["status"] == "ok")
    result["fail_steps"] = sum(1 for s in result["steps"].values() if s["status"] == "fail")
    result["ended_at"] = utc_now_iso()

    evidence = load_release_evidence()
    gates = evidence.setdefault("gates", {})
    release_gate = gates.setdefault("release_summary", {})
    release_gate["status"] = "pass" if result["status"] == "ok" else "fail"
    refs = release_gate.setdefault("evidence_refs", [])
    refs.append("phase3_final_gate.json")
    refs.append("phase3_final_gate_host_runtime_profile.json")
    refs.append("heartbeats/phase3_final_gate.jsonl")
    evidence["status"] = "approved" if result["status"] == "ok" else "blocked"
    if result["status"] != "ok":
        blocking = evidence.setdefault("blocking_reasons", [])
        if "failed_release_summary_gate" not in blocking:
            blocking.append("failed_release_summary_gate")
    save_release_evidence(evidence)

    out_path = save_gate_json("phase3_final_gate.json", result)
    print(json.dumps(result, indent=2))
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
