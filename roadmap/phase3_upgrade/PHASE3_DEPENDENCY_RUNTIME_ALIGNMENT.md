# Phase 3: Dependency and Runtime Alignment

## Objective

Align runtime dependencies and environment setup with latest upstream-compatible stacks while preserving Kaggle reproducibility.

## Scope

- Runtime dependency pins and optional extras.
- Kaggle install/setup scripts compatibility.
- Flash/Turing dependency toggles.

## Tasks

1. Dependency matrix
   - Align `pyproject.toml` deps and extras to upstream baseline.
   - Maintain a fork-specific `requirements.txt` for pinned runtime reproducibility.

2. Kaggle setup scripts
   - Validate `10_setup_uv.sh`, `20_install_deps.sh`, GPU checks, and flash install helpers.
   - Ensure scripts handle version-sensitive packages safely.

3. Compatibility probes
   - Verify train/val/export/benchmark script entrypoints after dependency refresh.

## Acceptance Criteria

- Clean environment bootstrap on Kaggle succeeds.
- Core scripts execute with refreshed dependency set.
- No regressions in flash backend selection from dependency changes.
