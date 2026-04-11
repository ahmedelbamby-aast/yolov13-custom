#!/usr/bin/env python3
"""Record autobatch + identical-GPU DDP best-practice notes and probe."""

from __future__ import annotations

import json
from pathlib import Path

import torch


OUT = Path('/kaggle/working/phase3_upgrade/autobatch_ddp_best_practices.json')


def main() -> None:
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            gpus.append({'index': i, 'name': p.name, 'total_memory': p.total_memory})

    same_gpu = len({(g['name'], g['total_memory']) for g in gpus}) <= 1 if gpus else False

    data = {
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'gpus': gpus,
        'identical_gpus': same_gpu,
        'best_practices': [
            'Run autobatch on a single GPU first to get per-GPU stable batch.',
            'Use fixed per-GPU batch in DDP and set global batch = per_gpu_batch * world_size.',
            'Do not run independent autobatch on each DDP rank.',
            'Keep workers/prefetch_factor/persistent_workers consistent per rank.',
            'Set static image size and deterministic flags for reproducibility during gate tests.',
            'For identical GPUs/vendors, keep same CUDA/PyTorch build on all nodes.',
        ],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2), encoding='utf-8')
    print(json.dumps(data, indent=2))
    print(f'saved={OUT}')


if __name__ == '__main__':
    main()
