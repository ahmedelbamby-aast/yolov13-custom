# Phase 1 Diff Baseline vs Upstream Main

Reference:

- Local branch: `phase3-upgrade-ultralytics-and-deps`
- Upstream reference: `upstream/main` (`ultralytics/ultralytics`)

Snapshot method:

- `git diff --name-status upstream/main --`

## High-level counts

- Total changed paths: `1363`
- Change types:
  - `D`: 596
  - `M`: 282
  - `A`: 485

## Top-level distribution

- `ultralytics`: 327
- `kaggle`: 413
- `docs`: 460
- `examples`: 71
- `roadmap`: 23
- `.github`: 16
- `docker`: 14
- `tests`: 9
- `scripts`: 8
- `notebooks`: 7
- `assets`: 4
- misc root files: remaining

## Baseline interpretation

1. Most divergence is outside core runtime (`docs`, `kaggle`, artifacts and custom project structure).
2. Core runtime divergence remains significant (`ultralytics`: 327 files), requiring phased reconciliation.
3. Immediate parity focus should stay on core runtime modules and package surfaces before secondary content.

## Priority buckets for next execution

1. Core package/runtime parity
   - `ultralytics/__init__.py`
   - `ultralytics/models/__init__.py`
   - `ultralytics/engine/trainer.py`
   - `ultralytics/optim/*`
2. Custom-delta retention replay
   - v13 configs
   - task preflight validation
   - OBB metric compatibility
   - flash backend controls
3. Dependency/runtime alignment and regression validation

## Phase 1 completion checklist status

- [x] Upstream remote attached and fetched
- [x] Baseline version surface aligned to `8.4.33`
- [x] MuSGD support wired and smoke-tested on Kaggle
- [x] Diff baseline captured and documented
- [ ] Complete core runtime file-level parity map (selected critical modules only)
