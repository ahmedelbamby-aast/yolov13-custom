# Spec 01 - Runtime and Environment Reliability

## Goal
Make CUDA visibility and Python environment setup deterministic for Kaggle GPU sessions.

## Background
In this environment, NVIDIA devices can be present while `torch.cuda.is_available()` returns false if `libcuda.so.1` is not visible in the runtime loader path.

## Implementation Plan
1. Add a common environment bootstrap script for all tasks.
2. Export GPU runtime paths in every script:
   - `/usr/local/nvidia/lib64`
   - `/usr/local/cuda/lib64`
3. Standardize workspace and output directories:
   - `/kaggle/work_here` for development
   - `/kaggle/working` for outputs
4. Create a uv virtualenv with system site packages to reuse Kaggle CUDA PyTorch build.
5. Add a strict GPU sanity check script.

## Dependency Diagram
```text
common.sh
  -> 10_setup_uv.sh
  -> 20_install_deps.sh
  -> 30_gpu_check.sh
  -> 40_ddp_smoke.sh
  -> 50_package_zip.sh
```

## Detailed Task List
- [x] Define required environment variables for workspace/output.
- [x] Force CUDA runtime library path setup in scripts.
- [x] Add uv setup script.
- [x] Add runtime GPU validation.
- [x] Ensure outputs are written to `/kaggle/working`.

## Edge Cases
- `/dev/nvidia*` exists but CUDA not discoverable by torch.
- Session shell path state differs between logins.
- Missing virtual environment.
