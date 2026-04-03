# Phase 1 Core Runtime Parity Map

This map tracks high-priority runtime files for upstream parity and custom delta retention.

Reference baseline: `upstream/main`

## Critical files

| File | Upstream diff status | Approx drift size (added/deleted) | Notes |
|---|---|---:|---|
| `ultralytics/__init__.py` | M | 2 / 3 | Mostly aligned to upstream 8.4 lazy import pattern. |
| `ultralytics/models/__init__.py` | M | 2 / 2 | Minor export-surface differences. |
| `ultralytics/engine/trainer.py` | M | 258 / 435 | High-risk file; contains custom + version drift. |
| `ultralytics/engine/validator.py` | M | 93 / 148 | Includes task-aware dataset check path. |
| `ultralytics/data/utils.py` | M | 332 / 286 | Contains task preflight custom logic. |
| `ultralytics/utils/metrics.py` | M | 604 / 839 | Includes OBB metric compatibility path. |
| `ultralytics/nn/modules/block.py` | M | 903 / 1030 | High-risk; includes flash backend controls. |
| `ultralytics/utils/dist.py` | M | 49 / 82 | DDP env propagation customizations. |
| `ultralytics/optim/muon.py` | M | 38 / 222 | Present but not yet byte-identical to upstream. |
| `ultralytics/optim/__init__.py` | M | 2 / 2 | Minor export differences only. |

## Risk grouping

- High risk (needs careful merge):
  - `ultralytics/engine/trainer.py`
  - `ultralytics/nn/modules/block.py`
  - `ultralytics/utils/metrics.py`
  - `ultralytics/data/utils.py`

- Medium risk:
  - `ultralytics/engine/validator.py`
  - `ultralytics/utils/dist.py`
  - `ultralytics/optim/muon.py`

- Low risk:
  - `ultralytics/__init__.py`
  - `ultralytics/models/__init__.py`
  - `ultralytics/optim/__init__.py`

## Execution order for next commits

1. Finish low-risk and medium-risk parity refinements.
2. Rebase/port high-risk files one by one with explicit custom-delta reapplication.
3. Run smoke checks after each high-risk file update.
