# turFlash on T4: Research, Gaps, and Optimization Plan

## Scope

This document focuses on Turing FlashAttention (called `turFlash` in this project) for YOLOv13 workloads on 2xT4.

Key questions answered:

- What does the turFlash repo currently support, and what is missing?
- How do these gaps impact YOLOv13 across n/s/l/x variants?
- What does YOLOv13 need to run faster in the flash-attention area?
- Which CUDA version is most optimal for 2xT4 and why?

## Source baseline

Primary upstreams used for this analysis:

- `ssiu/flash-attention-turing` (turFlash)
- `Dao-AILab/flash-attention` (FlashAttention mainline: FA2/FA3/FA4 families)

## Version families and positioning

- turFlash is a Turing-focused implementation with FA2-style API shape and a subset of features.
- It is not FA3 or FA4.
- FA3 and FA4 are Hopper/Blackwell oriented and are not the primary path for T4.

Practical mapping:

- T4 (sm75): use turFlash.
- Ampere/Ada/Hopper: mainline FlashAttention-2/3/4 paths depending on GPU generation.

## turFlash current support and limits

From turFlash README and setup:

- Supports:
  - forward + backward
  - head dimensions: 64 and 128
  - causal mask
  - GQA
  - varlen APIs
- Does not support:
  - dropout
  - local/sliding window mask
  - KV cache
- Build target:
  - explicit `sm_75` kernels (T4 class)

turFlash tested matrix in upstream README:

- CUDA 12.4
- PyTorch 2.5.1 and 2.8.0

## Critical YOLOv13 finding: head dimension mismatch

Current YOLOv13 attention path in this repo (`AAttn`) only enables flash kernel call if `head_dim in {64, 128}`.

However, YOLOv13 model construction (`A2C2f` + `ABlock` + `AAttn`) sets:

- `num_heads = c_ // 32`
- `head_dim = c_ // num_heads`

This drives `head_dim` to 32 for common scaled variants (n/s/l/x under current config constraints).

Impact:

- Backend may resolve as `flash_attn_turing`, but many (often all) AAttn layers still take fallback path because `head_dim == 32` is rejected by the flash gate.
- This is the single highest-impact gap for YOLOv13 speed on T4.

## Missing items and expected impact on YOLOv13

### 1) No `head_dim=32` turFlash kernel path

- Status: missing in turFlash and currently blocked in YOLO flash gate.
- Impact: high for all YOLOv13 scales, because AAttn likely remains fallback.
- Expected gain if implemented and wired: moderate to high wall-clock improvement on attention-heavy sections.

### 2) Incomplete wrapper usage in this repo

- Current wrapper (`ultralytics/utils/flash_turing_interface.py`) only exposes `flash_attn_func(q,k,v)`.
- turFlash upstream also exposes packed/varlen interfaces.
- Impact on YOLO detect train: low-to-medium now (fixed-shape batches), but useful for future variable-length paths and cleaner integrations.

### 3) Fallback attention implementation is manual matmul-softmax

- Current fallback in `AAttn` uses explicit matmul + exp normalization.
- More efficient fallback candidate: `torch.nn.functional.scaled_dot_product_attention`.
- Impact: medium, especially when flash gate misses.

### 4) No layer-level flash hit telemetry

- Backend selection is logged globally, but per-layer hit/miss is not summarized.
- Impact: indirect but high operational value (finds hidden fallback hotspots quickly).

### 5) turFlash feature limits (dropout/local mask/kv cache)

- For current YOLOv13 detect/seg/pose/obb training, these are usually not primary bottlenecks.
- Impact: low for present tasks.

## YOLOv13 needs for faster flash-attention execution

Prioritized technical needs:

1. Add `head_dim=32` kernel support and enable it in gate.
2. Add per-layer counters for flash-hit vs fallback-reason.
3. Replace manual fallback attention with SDPA path.
4. Reduce avoidable dtype/layout conversions in hot path.
5. Extend wrapper to include packed/varlen APIs (future-proofing).

## CUDA version recommendation for 2xT4

Recommended target for turFlash-focused runs:

- CUDA toolkit 12.4 (for build path), with a compatible PyTorch runtime.

Why:

- It is explicitly tested upstream for turFlash.
- It minimizes compiler/runtime surprises for `sm_75` kernels.
- It reduces risk compared to untested combinations where toolkit and torch CUDA build differ significantly.

Notes:

- Minor mismatch can still run (observed), but increases integration risk and makes performance/debug behavior less predictable.
- For production-like repeatability, align toolkit/torch/turFlash to a tested tuple rather than chasing newest toolkit by default.

## Proposed implementation track (project side)

Phase A (highest ROI, low ambiguity):

- Add flash telemetry in `AAttn.forward`.
- Add SDPA fallback.
- Add benchmark script output for flash-hit rate and fallback reasons.

Phase B (core speed unlock):

- Implement and integrate `head_dim=32` turFlash kernels.
- Extend gate to allow `{32, 64, 128}` when backend is `flash_attn_turing`.

Phase C (hardening):

- Add wrapper parity for packed/varlen APIs.
- Run fixed-profile benchmark matrix on 2xT4:
  - n/s/l/x
  - detect task
  - same dataset and fixed `imgsz`, `batch`, `workers`, `cache`, `fraction`

## Decision summary

- turFlash is the correct focus for T4.
- Main blocker for YOLOv13 speed is head-dim mismatch (`32` not supported by current flash kernel gate).
- Best next optimization work should target flash coverage first, not only backend selection logs.

## `not_cuda` telemetry investigation (root cause and actions)

Observed in telemetry:

- baseline run: `fallback_reasons = {'not_cuda': 24, 'unsupported_head_dim_32': ...}`
- head32-enabled run: `fallback_reasons = {'not_cuda': 24}`

What `not_cuda` means in this code path:

- In `AAttn.forward`, fallback reason `not_cuda` is recorded whenever input tensor `x` for that attention call is not on a CUDA device.
- This is expected during non-training-GPU execution segments.

Primary root cause in current benchmark context:

- Validation/inference paths execute some model forwards on CPU-side tensors (or CPU execution contexts) where flash kernels are unavailable by design.
- These calls are a small fixed count (24 in this benchmark), independent of the head-dim gate.

Why it persists after enabling head32:

- Enabling head32 only resolves `unsupported_head_dim_32` on CUDA paths.
- It does not change CPU-side forwards, so `not_cuda` remains.

Resolution strategy:

1. Keep `not_cuda` as expected noise floor; do not treat it as a regression.
2. Track CUDA-only flash hit-rate separately (exclude `not_cuda`) for cleaner KPI.
3. Ensure training/validation device config remains GPU (`--device 0` or `0,1`) to avoid accidental CPU runs.
4. Optionally add telemetry split fields:
   - `cuda_total`, `cuda_flash_hits`, `cuda_fallbacks`
   so comparisons focus on GPU-executable attention calls.
