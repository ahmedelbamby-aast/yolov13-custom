#!/usr/bin/env python3
"""Shared helpers for developer-facing YOLOv13 scripts."""

from __future__ import annotations

import argparse
import ast
import os
from typing import Any


def add_flash_mode_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--flash-mode",
        default="auto",
        choices=("auto", "fallback", "turing"),
        help="Flash backend mode: auto, force fallback, or force Turing flash.",
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
    if mode == "fallback":
        os.environ["Y13_DISABLE_FLASH"] = "1"
        os.environ["Y13_USE_TURING_FLASH"] = "0"
    elif mode == "turing":
        os.environ["Y13_DISABLE_FLASH"] = "0"
        os.environ["Y13_USE_TURING_FLASH"] = "1"
    else:  # auto
        os.environ["Y13_DISABLE_FLASH"] = "0"
        os.environ.setdefault("Y13_USE_TURING_FLASH", "0")


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
        f"Y13_USE_TURING_FLASH={os.environ.get('Y13_USE_TURING_FLASH', '')}"
    )
    print(f"[y13] resolved_flash_backend={flash_backend}")
    print(f"[y13] kwargs={kwargs}")
