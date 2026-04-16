#!/usr/bin/env python3
"""Shared helpers for alignment artifacts and gate reporting."""

from __future__ import annotations

import json
import os
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
