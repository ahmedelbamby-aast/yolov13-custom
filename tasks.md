# turFlash Optimization Tasks

- [x] Add flash dispatch telemetry in `AAttn` (hit/miss, reasons, head-dim counts).
- [x] Add dirty-data smoke benchmarks with generated report/plots.
- [ ] Make head-dim 32 support mandatory for Turing backend (with opt-out env).
- [x] Replace manual fallback attention with SDPA.
- [x] Extend Turing wrapper APIs with varlen support (mandatory hardening).
- [x] Add CUDA-only telemetry counters and report fields (`not_cuda` isolation).
- [x] Update dirty benchmark to run baseline/head32 in parallel on GPU1/GPU0 with workers=2 each and cache=ram.
- [x] Re-run dirty real-dataset benchmark with new parallel policy and sync fresh plots/artifacts.
- [ ] Verify feature-map projection integration in both script styles remains toggleable and post-train-only (best.pt only).
- [ ] Update QUICKSTART and research docs for each change and keep both branches in sync.
