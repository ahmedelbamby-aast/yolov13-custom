# Phase 6 - Turing Flash Optimization and Final Run Playbook

## Context

This phase documents the final custom training workflow and the optimization gap analysis for T4/Turing flash attention.

Primary goals:

- keep YOLOv13-L detect training stable on 2xT4 DDP
- force and verify Turing flash backend (`flash_attn_turing`)
- integrate full-image feature-map projection artifacts into the same run
- identify missing optimization points compared to:
  - https://github.com/ssiu/flash-attention-turing
  - https://github.com/Dao-AILab/flash-attention

## Final Run Requirements (Implemented)

- task: `detect`
- model: `yolov13l`
- optimizer: `MuSGD`
- time budget: `time=2` (hours)
- image size: `640`
- batch size: `16`
- workers: `8`
- cache: `ram`
- fraction: `1`
- DDP: `device=0,1` on 2xT4
- output root: `/kaggle/working/final_run`
- flash mode: forced Turing (`Y13_USE_TURING_FLASH=1`, `Y13_DISABLE_FLASH=0`, `--flash-mode turing`)

Dataset requirement implemented for this run:

- `/kaggle/work_here/datasets/roboflow_custom_detect/data.yaml`
  - `nc: 2`
  - `names: [student, teacher]`

## What Was Added for This Phase

### 1) Run launcher

- file: `kaggle/scripts/run_custom_time2_tmp.sh`
- responsibilities:
  - kill previous train/projection jobs
  - configure env for Turing flash
  - launch training with the required parameters
  - launch feature-map projection side process that waits for `best.pt`
  - write logs to:
    - `/kaggle/working/final_run/train.log`
    - `/kaggle/working/final_run/feature_projection.log`

### 2) Feature-map projection pipeline

- file: `kaggle/scripts/37_feature_map_projection.py`
- behavior:
  - waits for model artifact (`best.pt`)
  - captures per-layer feature maps through forward hooks
  - projects each map to full-image resolution and overlays on source image
  - writes report and metadata

Artifacts:

- `/kaggle/working/final_run/feature_projection/*.jpg`
- `/kaggle/working/final_run/feature_projection/feature_projection_meta.json`
- `/kaggle/working/final_run/ff_maps.md`

Enlargement/projection is explicitly enabled via:

- `cv2.resize(..., interpolation=cv2.INTER_CUBIC)`
- metadata fields:
  - `projection.mode = full_image_overlay`
  - `projection.enlargement_enabled = true`

## Runtime Verification Snapshot

From run logs, the critical checks were satisfied:

- flash env:
  - `Y13_DISABLE_FLASH=0`
  - `Y13_USE_TURING_FLASH=1`
- backend resolution:
  - `resolved_flash_backend=flash_attn_turing`
- DDP path:
  - launcher process uses `device=0,1`
  - spawned with `torch.distributed.run --nproc_per_node 2`

## Gap Analysis vs Flash Repositories

### Current strengths

- backend switching is explicit and stable in `ultralytics/nn/modules/block.py`
- Turing path is integrated and selectable by env/CLI
- training confirmed to run with `flash_attn_turing`

### Missing or suboptimal items

1. **Fallback attention path is manual matmul-softmax**
   - current fallback in `AAttn.forward` computes attention manually
   - impact: when flash is not used, slower and more memory pressure on T4

2. **No per-layer flash hit telemetry**
   - only global backend is logged
   - impact: difficult to detect partial fallback at layer granularity

3. **Turing wrapper exposes only base API**
   - current wrapper (`ultralytics/utils/flash_turing_interface.py`) exports only `flash_attn_func`
   - `flash-attention-turing` also includes packed/varlen APIs
   - impact: misses potential packing/dispatch optimizations

4. **Hot-path dtype/layout overhead not minimized**
   - repeated cast/contiguous handling in attention path
   - impact: small overhead repeated every forward

5. **Environment may not be peak-tested stack for Turing repo**
   - functional now, but not necessarily highest throughput configuration
   - impact: possible additional performance left on table

## Proposed Patch Set (Prioritized)

### Patch A - SDPA fallback replacement (highest ROI)

- replace manual fallback attention with `torch.nn.functional.scaled_dot_product_attention`
- expected effect:
  - if all layers already flash: small gain
  - if any layer falls back: moderate to high gain and lower memory pressure

### Patch B - Layer-level flash usage telemetry

- add counters/reason tags in `AAttn.forward` (hit/miss + reason)
- emit summary at epoch/end
- expected effect:
  - no direct speedup, but high debugging value and faster optimization cycles

### Patch C - Minimize q/k/v conversion overhead

- avoid unnecessary recast/recontiguous when tensors already compatible
- expected effect: small but consistent speed gain

### Patch D - Extend Turing Python wrapper APIs

- add packed/varlen interface parity where applicable
- expected effect: low-to-moderate gain depending on tensor layout usage

### Patch E - Controlled runtime stack benchmarking

- test torch/cuda combinations in a matrix with fixed benchmark protocol
- expected effect: variable; can produce meaningful gains if stable

## Estimated Performance Impact

These are practical expected ranges (T4, this model family, same dataset/task):

- Patch A (SDPA fallback): ~5-20% improvement on fallback-hit segments
- Patch B (telemetry): 0% direct, high operational effectiveness
- Patch C (dtype/layout): ~1-4%
- Patch D (API extension): ~0-5% in current YOLO detect path, potentially more for future variants
- Patch E (stack tuning): ~3-12% depending on compatibility and kernel behavior

## Validation Plan for Phase 6.1

After implementing patches, run the same fixed profile:

- model/task/dataset identical
- `time=2`, `batch=16`, `imgsz=640`, `workers=8`, `cache=ram`, `fraction=1`
- DDP on 2xT4
- compare:
  - epoch throughput
  - wall-clock per epoch
  - GPU memory trends
  - final metrics stability
  - flash hit-rate telemetry

Success criteria:

- no regression in correctness/metrics stability
- improved throughput or reduced time-per-epoch
- full run artifacts still produced under `/kaggle/working/final_run`

## Sync Status

Scripts and docs for this phase were synced to `origin/main` and validated on the Kaggle workspace.

Relevant commits:

- `167a96e20` - add final run launcher + feature-map projection scripts
- `673317e92` - add repeatable final-run procedure in `kaggle/QUICKSTART.md`
