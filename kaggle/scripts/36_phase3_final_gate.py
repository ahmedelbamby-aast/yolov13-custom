#!/usr/bin/env python3
"""Phase3 final integration gate across core workflows."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path


def step(result: dict, name: str, fn):
    t0 = time.time()
    try:
        out = fn()
        result["steps"][name] = {"status": "ok", "elapsed_s": time.time() - t0, "output": out}
    except Exception as e:
        result["steps"][name] = {"status": "fail", "elapsed_s": time.time() - t0, "error": str(e)[:1200]}
        result["status"] = "fail"


def main() -> None:
    os.environ.setdefault("Y13_DISABLE_FLASH", "0")
    os.environ.setdefault("Y13_USE_TURING_FLASH", "0")

    from ultralytics import YOLO

    result = {
        "phase": "phase3",
        "gate": "final_integration",
        "status": "ok",
        "steps": {},
    }

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
    out_path = Path("/kaggle/working/phase3_final_gate.json")
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
