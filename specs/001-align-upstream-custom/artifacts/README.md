# Alignment Artifacts

This directory contains machine-readable artifacts for the
`001-align-upstream-custom` release gate workflow.

- `upstream-baseline.json`: pinned upstream reference and scope.
- `parity-inventory.yaml`: per-workflow `WorkflowParityItem` records.
- `custom-feature-registry.yaml`: release-blocking custom feature inventory.
- `parity-exceptions.yaml`: intentional parity divergence records.
- `release-evidence.yaml`: consolidated release decision evidence.

All release-blocking decisions must reference artifacts in this directory.
