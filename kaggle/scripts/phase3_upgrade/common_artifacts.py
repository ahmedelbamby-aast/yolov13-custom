#!/usr/bin/env python3
"""Shared helpers for alignment artifacts and gate reporting."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml


EXCEPTION_REQUIRED_FIELDS = (
    "exception_id",
    "workflow_id",
    "rationale",
    "risk_level",
    "owner",
    "rollback_or_mitigation",
    "remediation_date",
    "approval_state",
)


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def repo_root() -> Path:
    """Resolve repository root from this file location."""
    return Path(__file__).resolve().parents[3]


def feature_artifacts_dir() -> Path:
    """Resolve feature artifact directory with optional override."""
    override = os.environ.get("Y13_FEATURE_ARTIFACTS_DIR", "").strip()
    if override:
        return Path(override)
    return repo_root() / "specs" / "001-align-upstream-custom" / "artifacts"


def _read_yaml(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return default if data is None else data


def _write_yaml(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_baseline() -> Dict[str, Any]:
    """Load upstream baseline artifact."""
    return _read_yaml(feature_artifacts_dir() / "upstream-baseline.json", {})


def load_parity_inventory() -> Dict[str, Any]:
    """Load parity inventory artifact."""
    return _read_yaml(feature_artifacts_dir() / "parity-inventory.yaml", {"workflows": []})


def load_custom_feature_registry() -> Dict[str, Any]:
    """Load custom feature registry artifact."""
    return _read_yaml(feature_artifacts_dir() / "custom-feature-registry.yaml", {"features": []})


def load_parity_exceptions() -> Dict[str, Any]:
    """Load parity exception artifact."""
    return _read_yaml(feature_artifacts_dir() / "parity-exceptions.yaml", {"exceptions": []})


def save_parity_exceptions(data: Dict[str, Any]) -> None:
    """Persist parity exception artifact."""
    _write_yaml(feature_artifacts_dir() / "parity-exceptions.yaml", data)


def load_release_evidence() -> Dict[str, Any]:
    """Load release evidence artifact."""
    return _read_yaml(feature_artifacts_dir() / "release-evidence.yaml", {})


def save_release_evidence(data: Dict[str, Any]) -> None:
    """Persist release evidence artifact."""
    _write_yaml(feature_artifacts_dir() / "release-evidence.yaml", data)


def save_gate_json(report_name: str, data: Dict[str, Any]) -> Path:
    """Save a JSON gate report under feature artifacts."""
    path = feature_artifacts_dir() / report_name
    _write_json(path, data)
    return path


def detect_host_runtime_profile() -> Dict[str, Any]:
    """Detect host portability/runtime profile for gate evidence."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    headless = not any(os.environ.get(k) for k in ("DISPLAY", "WAYLAND_DISPLAY", "MIR_SOCKET"))

    gpu_names: List[str] = []
    gpu_count = 0
    cuda_available = False

    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            gpu_count = int(torch.cuda.device_count())
            gpu_names = [str(torch.cuda.get_device_name(i)) for i in range(gpu_count)]
    except Exception:
        pass

    if gpu_count <= 0:
        try:
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                check=False,
                capture_output=True,
                text=True,
            )
            if proc.returncode == 0:
                gpu_names = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
                gpu_count = len(gpu_names)
                cuda_available = gpu_count > 0
        except Exception:
            pass

    if gpu_count <= 0:
        accelerator_profile = "cpu-only"
    elif gpu_count == 1:
        accelerator_profile = "single-gpu"
    else:
        accelerator_profile = "multi-gpu"

    return {
        "detected_at": utc_now_iso(),
        "os_family": system,
        "machine": machine,
        "headless": headless,
        "accelerator_profile": accelerator_profile,
        "gpu_count": gpu_count,
        "gpu_names": gpu_names,
        "cuda_available": cuda_available,
    }


def save_host_runtime_profile(name: str = "host_runtime_profile.json") -> Path:
    """Save host runtime profile under feature artifacts."""
    path = feature_artifacts_dir() / name
    _write_json(path, detect_host_runtime_profile())
    return path


def append_progress_heartbeat(job_name: str, status: str, details: Dict[str, Any] | None = None) -> Path:
    """Append one heartbeat record for long-running jobs."""
    hb_dir = feature_artifacts_dir() / "heartbeats"
    hb_dir.mkdir(parents=True, exist_ok=True)
    path = hb_dir / f"{job_name}.jsonl"
    payload = {
        "job": job_name,
        "status": status,
        "ts": utc_now_iso(),
        "details": details or {},
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    return path


class HeartbeatTicker:
    """Background heartbeat writer with fixed interval."""

    def __init__(self, job_name: str, interval_s: int = 300):
        self.job_name = job_name
        self.interval_s = max(60, int(interval_s))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            append_progress_heartbeat(self.job_name, "running", {"interval_s": self.interval_s})

    def start(self) -> None:
        append_progress_heartbeat(self.job_name, "started", {"interval_s": self.interval_s})
        self._thread = threading.Thread(target=self._run, name=f"hb-{self.job_name}", daemon=True)
        self._thread.start()

    def stop(self, status: str = "completed", details: Dict[str, Any] | None = None) -> None:
        self._stop.set()
        append_progress_heartbeat(self.job_name, status, details)

    def __enter__(self) -> "HeartbeatTicker":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop("failed" if exc else "completed")


def validate_exception_records(exceptions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate required fields for exception records."""
    missing: List[Dict[str, Any]] = []
    for item in exceptions:
        missing_fields = [k for k in EXCEPTION_REQUIRED_FIELDS if not item.get(k)]
        if missing_fields:
            missing.append(
                {
                    "exception_id": item.get("exception_id", "<missing>"),
                    "missing_fields": missing_fields,
                }
            )
    return {
        "total": len(exceptions),
        "complete": len(exceptions) - len(missing),
        "incomplete": len(missing),
        "missing_details": missing,
    }


def normalized_step_result(
    name: str,
    status: str,
    started_at: str,
    ended_at: str,
    details: Dict[str, Any] | None = None,
    error: str | None = None,
) -> Dict[str, Any]:
    """Create normalized gate step payload."""
    payload: Dict[str, Any] = {
        "name": name,
        "status": status,
        "started_at": started_at,
        "ended_at": ended_at,
        "details": details or {},
    }
    if error:
        payload["error"] = error
    return payload
