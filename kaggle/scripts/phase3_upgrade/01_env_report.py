#!/usr/bin/env python3
"""Phase3 upgrade gate: capture environment/runtime report."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torchvision


OUT = Path("/kaggle/working/phase3_upgrade/env_report.json")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "python": os.popen("python3 --version").read().strip(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_count": int(torch.cuda.device_count()),
        "gpu0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cpu_count": os.cpu_count(),
    }
    OUT.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(json.dumps(data, indent=2))
    print(f"saved={OUT}")


if __name__ == "__main__":
    main()
