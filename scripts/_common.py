#!/usr/bin/env python3
"""Shared helpers for developer-facing YOLOv13 scripts."""

from __future__ import annotations

import argparse
import ast
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FLASH_MODE_CHOICES = ("auto", "fallback", "turing", "flash4")
DESTRUCTIVE_SERVER_ACTIONS = {
    "reboot",
    "shutdown",
    "delete_server",
    "delete_system_files",
    "format_disk",
}

# Shared artifact directory used by alignment gates.
FEATURE_ARTIFACTS_DIR = Path(os.environ.get("Y13_FEATURE_ARTIFACTS_DIR", "specs/001-align-upstream-custom/artifacts"))


def add_flash_mode_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--flash-mode",
        default="auto",
        choices=FLASH_MODE_CHOICES,
        help="Flash backend mode: auto, fallback, turing, or flash4 (CuTe on Blackwell/Hopper).",
    )


def add_extra_overrides_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra Ultralytics override. Repeat for multiple values.",
    )


def apply_flash_mode(mode: str) -> None:
    """Set env flags before importing ultralytics modules."""
    resolved = mode
    if mode == "auto":
        env_mode = os.environ.get("Y13_FLASH_MODE", "").strip().lower()
        if env_mode in FLASH_MODE_CHOICES:
            resolved = env_mode

    if resolved == "fallback":
        os.environ["Y13_DISABLE_FLASH"] = "1"
        os.environ["Y13_USE_TURING_FLASH"] = "0"
        os.environ["Y13_PREFER_FLASH4"] = "0"
    elif resolved == "turing":
        os.environ["Y13_DISABLE_FLASH"] = "0"
        os.environ["Y13_USE_TURING_FLASH"] = "1"
        os.environ["Y13_PREFER_FLASH4"] = "0"
    elif resolved == "flash4":
        os.environ["Y13_DISABLE_FLASH"] = "0"
        os.environ["Y13_USE_TURING_FLASH"] = "0"
        os.environ["Y13_PREFER_FLASH4"] = "1"
    else:  # auto
        os.environ["Y13_DISABLE_FLASH"] = "0"
        os.environ["Y13_USE_TURING_FLASH"] = "0"
        os.environ["Y13_PREFER_FLASH4"] = "0"

    # Runtime stability defaults for long runs on Blackwell/SM120.
    os.environ.setdefault("Y13_FLASH_QUARANTINE", "1")
    os.environ.setdefault("Y13_FLASH_FAIL_THRESHOLD", "64")
    os.environ.setdefault("Y13_FLASH4_DTYPE", "auto")


def auto_configure_flash_environment() -> dict[str, Any]:
    """Set flash env defaults from detected hardware profile."""
    profile = detect_host_runtime_profile()
    recommendation = profile.get("flash_recommendation", "fallback")
    os.environ.setdefault("Y13_HOST_ACCELERATOR_PROFILE", str(profile.get("accelerator_profile", "unknown")))
    os.environ.setdefault("Y13_HOST_OS_FAMILY", str(profile.get("os_family", "unknown")))

    if recommendation == "turing":
        os.environ.setdefault("Y13_USE_TURING_FLASH", "1")
        os.environ.setdefault("Y13_DISABLE_FLASH", "0")
    elif recommendation == "auto":
        os.environ.setdefault("Y13_USE_TURING_FLASH", "0")
        os.environ.setdefault("Y13_DISABLE_FLASH", "0")
    else:
        os.environ.setdefault("Y13_USE_TURING_FLASH", "0")
        os.environ.setdefault("Y13_DISABLE_FLASH", "1")

    return profile


def enforce_server_safety(action: str, approved: bool | None = None) -> None:
    """Block destructive server operations unless explicit approval exists."""
    normalized = (action or "").strip().lower()
    if normalized not in DESTRUCTIVE_SERVER_ACTIONS:
        return

    if approved is None:
        approved = os.environ.get("Y13_DEVELOPER_APPROVED_DESTRUCTIVE", "0") == "1"

    if not approved:
        raise PermissionError(
            "Destructive server action blocked. Inform developer and set "
            "Y13_DEVELOPER_APPROVED_DESTRUCTIVE=1 only with explicit approval."
        )


def detect_host_runtime_profile() -> dict[str, Any]:
    """Detect host OS/headless/accelerator profile for portability-aware flows."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    is_headless = not any(os.environ.get(k) for k in ("DISPLAY", "WAYLAND_DISPLAY", "MIR_SOCKET"))

    gpu_names: list[str] = []
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

    if gpu_count == 0:
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

    t4_compatible = any("t4" in name.lower() for name in gpu_names)
    flash_recommendation = "fallback"
    if accelerator_profile != "cpu-only":
        flash_recommendation = "turing" if t4_compatible else "auto"

    return {
        "detected_at": datetime.now(timezone.utc).isoformat(),
        "os_family": system,
        "machine": machine,
        "headless": is_headless,
        "accelerator_profile": accelerator_profile,
        "gpu_count": gpu_count,
        "gpu_names": gpu_names,
        "cuda_available": cuda_available,
        "t4_compatible": t4_compatible,
        "flash_recommendation": flash_recommendation,
    }


def write_host_runtime_profile(path: str | Path) -> Path:
    """Write host profile JSON artifact and return path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(detect_host_runtime_profile(), indent=2), encoding="utf-8")
    return out


def artifact_path(name: str) -> Path:
    """Return resolved feature artifact path for a filename."""
    return FEATURE_ARTIFACTS_DIR / name


def _parse_scalar(value: str) -> Any:
    text = value.strip()
    low = text.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"none", "null"}:
        return None

    try:
        return ast.literal_eval(text)
    except Exception:
        return text


def parse_extra_overrides(raw_items: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in raw_items:
        if "=" not in item:
            raise ValueError(f"Invalid --arg '{item}'. Expected KEY=VALUE.")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --arg '{item}'. Empty KEY is not allowed.")
        overrides[key] = _parse_scalar(raw_value)
    return overrides


def parse_unknown_cli_overrides(tokens: list[str]) -> dict[str, Any]:
    """Parse unknown --key value/--key=value tokens into Ultralytics kwargs."""
    overrides: dict[str, Any] = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token.startswith("--"):
            raise ValueError(f"Unexpected token '{token}'. Use '--key value' or '--key=value'.")

        key_token = token[2:]
        if not key_token:
            raise ValueError("Invalid token '--'.")

        if "=" in key_token:
            key, raw_value = key_token.split("=", 1)
            overrides[key] = _parse_scalar(raw_value)
            i += 1
            continue

        key = key_token
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            overrides[key] = _parse_scalar(tokens[i + 1])
            i += 2
        else:
            overrides[key] = True
            i += 1

    return overrides


def merge_kwarg_sources(*sources: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for src in sources:
        for k, v in src.items():
            if v is not None:
                merged[k] = v
    return merged


def load_yolo_and_flash_backend() -> tuple[Any, str]:
    """Import YOLO after flash env is set, then force backend re-eval."""
    from ultralytics import YOLO
    from ultralytics.nn.modules import block

    if hasattr(block, "reset_flash_telemetry"):
        block.reset_flash_telemetry()

    if hasattr(block, "configure_flash_backend"):
        block.configure_flash_backend(
            disable_flash=os.environ.get("Y13_DISABLE_FLASH", "0") == "1",
            use_turing_flash=os.environ.get("Y13_USE_TURING_FLASH", "0") == "1",
        )

    return YOLO, getattr(block, "FLASH_BACKEND", "unknown")


def print_runtime_header(action: str, flash_backend: str, kwargs: dict[str, Any]) -> None:
    print(f"[y13] action={action}")
    print(
        f"[y13] flash_mode_env: Y13_DISABLE_FLASH={os.environ.get('Y13_DISABLE_FLASH', '')} "
        f"Y13_USE_TURING_FLASH={os.environ.get('Y13_USE_TURING_FLASH', '')} "
        f"Y13_PREFER_FLASH4={os.environ.get('Y13_PREFER_FLASH4', '')} "
        f"Y13_FLASH_QUARANTINE={os.environ.get('Y13_FLASH_QUARANTINE', '')} "
        f"Y13_FLASH_FAIL_THRESHOLD={os.environ.get('Y13_FLASH_FAIL_THRESHOLD', '')} "
        f"Y13_FLASH4_DTYPE={os.environ.get('Y13_FLASH4_DTYPE', '')}"
    )
    print(f"[y13] resolved_flash_backend={flash_backend}")
    print(f"[y13] kwargs={kwargs}")
