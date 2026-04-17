# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import sys
from unittest import mock

from tests import MODEL
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.engine.exporter import Exporter
from ultralytics.models.yolo import classify, detect, segment
from ultralytics.utils import ASSETS, DEFAULT_CFG, WEIGHTS_DIR


def test_func(*args):  # noqa
    """Test function callback for evaluating YOLO model performance metrics."""
    print("callback test passed")


def test_flash_backend_toggle_env_contract():
    """Validate flash backend env toggles are deterministic."""
    import os

    from scripts._common import apply_flash_mode

    apply_flash_mode("fallback")
    assert os.environ.get("Y13_DISABLE_FLASH") == "1"
    assert os.environ.get("Y13_USE_TURING_FLASH") == "0"
    assert os.environ.get("Y13_PREFER_FLASH4") == "0"

    apply_flash_mode("turing")
    assert os.environ.get("Y13_DISABLE_FLASH") == "0"
    assert os.environ.get("Y13_USE_TURING_FLASH") == "1"
    assert os.environ.get("Y13_PREFER_FLASH4") == "0"

    apply_flash_mode("flash4")
    assert os.environ.get("Y13_DISABLE_FLASH") == "0"
    assert os.environ.get("Y13_USE_TURING_FLASH") == "0"
    assert os.environ.get("Y13_PREFER_FLASH4") == "1"


def test_release_blocking_logic_reference_present():
    """Ensure release-blocking aggregator script exists for governance path."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    target = root / "kaggle" / "scripts" / "phase3_upgrade" / "03_stress_gate.py"
    assert target.exists()


def test_host_runtime_profile_detection_shape():
    """Ensure runtime host profile includes required portability keys."""
    from scripts._common import detect_host_runtime_profile

    profile = detect_host_runtime_profile()
    required = {
        "os_family",
        "headless",
        "accelerator_profile",
        "gpu_count",
        "gpu_names",
        "cuda_available",
        "flash_recommendation",
    }
    assert required.issubset(profile.keys())
    assert profile["accelerator_profile"] in {"cpu-only", "single-gpu", "multi-gpu"}


def test_release_blocking_path_flagged_when_final_gate_fails():
    """US3: release evidence must include final gate blocking reason when summary gate fails."""
    import yaml
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    p = root / "specs" / "001-align-upstream-custom" / "artifacts" / "release-evidence.yaml"
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    blocking = set(data.get("blocking_reasons", []) or [])
    release_summary = ((data.get("gates") or {}).get("release_summary") or {}).get("status")

    if release_summary == "fail":
        assert "failed_release_summary_gate" in blocking


def test_export():
    """Tests the model exporting function by adding a callback and asserting its execution."""
    exporter = Exporter()
    exporter.add_callback("on_export_start", test_func)
    assert test_func in exporter.callbacks["on_export_start"], "callback test failed"
    f = exporter(model=YOLO("yolo11n.yaml").model)
    YOLO(f)(ASSETS)  # exported model inference


def test_detect():
    """Test YOLO object detection training, validation, and prediction functionality."""
    overrides = {"data": "coco8.yaml", "model": "yolo11n.yaml", "imgsz": 32, "epochs": 1, "save": False}
    cfg = get_cfg(DEFAULT_CFG)
    cfg.data = "coco8.yaml"
    cfg.imgsz = 32

    # Trainer
    trainer = detect.DetectionTrainer(overrides=overrides)
    trainer.add_callback("on_train_start", test_func)
    assert test_func in trainer.callbacks["on_train_start"], "callback test failed"
    trainer.train()

    # Validator
    val = detect.DetectionValidator(args=cfg)
    val.add_callback("on_val_start", test_func)
    assert test_func in val.callbacks["on_val_start"], "callback test failed"
    val(model=trainer.best)  # validate best.pt

    # Predictor
    pred = detect.DetectionPredictor(overrides={"imgsz": [64, 64]})
    pred.add_callback("on_predict_start", test_func)
    assert test_func in pred.callbacks["on_predict_start"], "callback test failed"
    # Confirm there is no issue with sys.argv being empty.
    with mock.patch.object(sys, "argv", []):
        result = pred(source=ASSETS, model=MODEL)
        assert list(result), "predictor test failed"

    overrides["resume"] = trainer.last
    trainer = detect.DetectionTrainer(overrides=overrides)
    try:
        trainer.train()
    except Exception as e:
        print(f"Expected exception caught: {e}")
        return

    Exception("Resume test failed!")


def test_segment():
    """Tests image segmentation training, validation, and prediction pipelines using YOLO models."""
    overrides = {"data": "coco8-seg.yaml", "model": "yolo11n-seg.yaml", "imgsz": 32, "epochs": 1, "save": False}
    cfg = get_cfg(DEFAULT_CFG)
    cfg.data = "coco8-seg.yaml"
    cfg.imgsz = 32
    # YOLO(CFG_SEG).train(**overrides)  # works

    # Trainer
    trainer = segment.SegmentationTrainer(overrides=overrides)
    trainer.add_callback("on_train_start", test_func)
    assert test_func in trainer.callbacks["on_train_start"], "callback test failed"
    trainer.train()

    # Validator
    val = segment.SegmentationValidator(args=cfg)
    val.add_callback("on_val_start", test_func)
    assert test_func in val.callbacks["on_val_start"], "callback test failed"
    val(model=trainer.best)  # validate best.pt

    # Predictor
    pred = segment.SegmentationPredictor(overrides={"imgsz": [64, 64]})
    pred.add_callback("on_predict_start", test_func)
    assert test_func in pred.callbacks["on_predict_start"], "callback test failed"
    result = pred(source=ASSETS, model=WEIGHTS_DIR / "yolo11n-seg.pt")
    assert list(result), "predictor test failed"

    # Test resume
    overrides["resume"] = trainer.last
    trainer = segment.SegmentationTrainer(overrides=overrides)
    try:
        trainer.train()
    except Exception as e:
        print(f"Expected exception caught: {e}")
        return

    Exception("Resume test failed!")


def test_classify():
    """Test image classification including training, validation, and prediction phases."""
    overrides = {"data": "imagenet10", "model": "yolo11n-cls.yaml", "imgsz": 32, "epochs": 1, "save": False}
    cfg = get_cfg(DEFAULT_CFG)
    cfg.data = "imagenet10"
    cfg.imgsz = 32
    # YOLO(CFG_SEG).train(**overrides)  # works

    # Trainer
    trainer = classify.ClassificationTrainer(overrides=overrides)
    trainer.add_callback("on_train_start", test_func)
    assert test_func in trainer.callbacks["on_train_start"], "callback test failed"
    trainer.train()

    # Validator
    val = classify.ClassificationValidator(args=cfg)
    val.add_callback("on_val_start", test_func)
    assert test_func in val.callbacks["on_val_start"], "callback test failed"
    val(model=trainer.best)

    # Predictor
    pred = classify.ClassificationPredictor(overrides={"imgsz": [64, 64]})
    pred.add_callback("on_predict_start", test_func)
    assert test_func in pred.callbacks["on_predict_start"], "callback test failed"
    result = pred(source=ASSETS, model=trainer.best)
    assert list(result), "predictor test failed"
