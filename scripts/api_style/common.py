#!/usr/bin/env python3
"""Shared helpers for API-style Ultralytics scripts."""

from __future__ import annotations

import ast
import os
from typing import Any


FLASH_MODE_CHOICES = ("auto", "fallback", "turing", "flash4")


def apply_flash_mode(mode: str) -> None:
    mode = (mode or "auto").strip().lower()
    if mode == "auto":
        env_mode = os.environ.get("Y13_FLASH_MODE", "").strip().lower()
        if env_mode in FLASH_MODE_CHOICES:
            mode = env_mode

    if mode == "fallback":
        os.environ["Y13_DISABLE_FLASH"] = "1"
        os.environ["Y13_USE_TURING_FLASH"] = "0"
        os.environ["Y13_PREFER_FLASH4"] = "0"
    elif mode == "turing":
        os.environ["Y13_DISABLE_FLASH"] = "0"
        os.environ["Y13_USE_TURING_FLASH"] = "1"
        os.environ["Y13_PREFER_FLASH4"] = "0"
    elif mode == "flash4":
        os.environ["Y13_DISABLE_FLASH"] = "0"
        os.environ["Y13_USE_TURING_FLASH"] = "0"
        os.environ["Y13_PREFER_FLASH4"] = "1"
    else:
        os.environ["Y13_DISABLE_FLASH"] = "0"
        os.environ["Y13_USE_TURING_FLASH"] = "0"
        os.environ["Y13_PREFER_FLASH4"] = "0"


def resolve_flash_backend() -> str:
    from ultralytics.nn.modules import block

    if hasattr(block, "reset_flash_telemetry"):
        block.reset_flash_telemetry()

    if hasattr(block, "configure_flash_backend"):
        block.configure_flash_backend(
            disable_flash=os.environ.get("Y13_DISABLE_FLASH", "0") == "1",
            use_turing_flash=os.environ.get("Y13_USE_TURING_FLASH", "0") == "1",
        )
    return getattr(block, "FLASH_BACKEND", "unknown")


def print_runtime(action: str, flash_mode: str, backend: str, kwargs: dict) -> None:
    print(f"[api-style] action={action}")
    print(
        "[api-style] flash_mode_env: "
        f"Y13_DISABLE_FLASH={os.environ.get('Y13_DISABLE_FLASH', '')} "
        f"Y13_USE_TURING_FLASH={os.environ.get('Y13_USE_TURING_FLASH', '')} "
        f"Y13_PREFER_FLASH4={os.environ.get('Y13_PREFER_FLASH4', '')}"
    )
    print(f"[api-style] flash_mode={flash_mode}")
    print(f"[api-style] resolved_flash_backend={backend}")
    print(f"[api-style] kwargs={kwargs}")


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


def parse_kv_overrides(raw_items: list[str]) -> dict[str, Any]:
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
