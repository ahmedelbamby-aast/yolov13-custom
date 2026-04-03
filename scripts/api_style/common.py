#!/usr/bin/env python3
"""Shared helpers for API-style Ultralytics scripts."""

from __future__ import annotations

import os


def apply_flash_mode(mode: str) -> None:
    mode = (mode or "auto").strip().lower()
    if mode == "fallback":
        os.environ["Y13_DISABLE_FLASH"] = "1"
        os.environ["Y13_USE_TURING_FLASH"] = "0"
    elif mode == "turing":
        os.environ["Y13_DISABLE_FLASH"] = "0"
        os.environ["Y13_USE_TURING_FLASH"] = "1"
    else:
        os.environ["Y13_DISABLE_FLASH"] = "0"
        os.environ.setdefault("Y13_USE_TURING_FLASH", "0")


def resolve_flash_backend() -> str:
    from ultralytics.nn.modules import block

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
        f"Y13_USE_TURING_FLASH={os.environ.get('Y13_USE_TURING_FLASH', '')}"
    )
    print(f"[api-style] flash_mode={flash_mode}")
    print(f"[api-style] resolved_flash_backend={backend}")
    print(f"[api-style] kwargs={kwargs}")
