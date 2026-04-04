# YOLOv13 Training Speed Tasks

- [x] Fix DDP `world_size` propagation so EMA buffer broadcast uses real distributed state.
- [x] Add DDP runtime toggles for `find_unused_parameters` and `gradient_as_bucket_view` with speed-first defaults.
- [x] Add DataLoader runtime toggles for `persistent_workers` and `prefetch_factor`.
- [x] Reorder bootstrap pipeline to ensure GPU driver/runtime readiness before optional turFlash build.
- [x] Make turFlash installer expose `TORCH_CUDA_ARCH_LIST` and configurable compile jobs.
- [x] Add Torch/TorchVision install overrides in setup scripts so upgraded stacks can be tested reproducibly.
- [x] Keep benchmark and QUICKSTART docs aligned with new runtime/env knobs.
- [ ] Run controlled A/B throughput test on target server (before/after) and capture epoch-time deltas.
