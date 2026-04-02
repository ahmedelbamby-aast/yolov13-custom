# SPEC-11: Data Contracts for Segment, Pose, OBB

## Purpose

Define strict dataset schema and validation rules so task training fails early with actionable diagnostics.

## Segment Contract

- Dataset YAML must contain `path`, `train`, `val`, `names`, `nc`.
- Labels must include polygon/mask data in expected Ultralytics segmentation format.
- Empty labels are allowed but must not break batch collation.

## Pose Contract

- Dataset YAML must define `kpt_shape: [num_kpts, dims]` where dims in `{2, 3}`.
- Multi-instance-per-image keypoints are supported and must be validated.
- Missing/occluded visibility values follow Ultralytics conventions.

## OBB Contract

- Labels must be valid rotated boxes in the expected format for OBB pipeline.
- Angle convention must be documented and consistently applied.
- Dataset conversion (e.g., DOTA-style) must preserve orientation semantics.

## Preflight Validator Requirements

- New preflight checks run before dataloader build.
- Errors include file path, row index, and expected format example.
- Optional strict mode to stop on first malformed sample.

## Acceptance

- Invalid datasets produce clear, deterministic error messages.
- Valid reference datasets pass without warnings:
  - `coco8-seg`
  - `coco8-pose`
  - `dota8`
