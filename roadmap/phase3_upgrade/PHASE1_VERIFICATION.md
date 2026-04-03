# Phase 1 Verification Snapshot

Branch: `phase3-upgrade-ultralytics-and-deps`

## Runtime smoke checks (Kaggle)

1. MuSGD training probe
   - Command family: `python scripts/train.py ... --optimizer MuSGD --epochs 1`
   - Result: pass
   - Verified after trainer, data-utils, metrics, and block updates.

2. Validation probe
   - Command family: `python scripts/val.py ... --task detect`
   - Result: pass
   - `scripts/val.py` updated to print `metrics.results_dict` for compatibility.

3. Task preflight smoke
   - Result: `6/6` expected outcomes passed.
   - Artifact: `roadmap/artifacts/phase3_task_preflight_smoke.json`

4. YOLOv13 model load matrix
   - Result: `16/16` configs loaded successfully.
   - Artifact: `roadmap/artifacts/phase3_model_load_matrix.json`

5. Task validation smoke matrix (fallback + turing)
   - Scope: `detect`, `segment`, `pose`, `obb` under both backend modes.
   - Result: `8/8` runs passed.
   - Turing path confirmed active (`FLASH_BACKEND=flash_attn_turing`).
   - Artifact: `roadmap/artifacts/phase3_task_val_smoke.json`

6. Custom delta audit
   - Scope: verify key custom features survived upstream-alignment updates.
   - Result: `11/11` checks passed.
   - Artifact: `roadmap/artifacts/phase3_custom_delta_audit.json`

7. DDP smoke gate
   - Scope: 2-GPU DDP train smoke on Kaggle (`device=0,1`, 1 epoch).
   - Result: pass.
   - Artifact: `roadmap/artifacts/phase3_ddp_smoke.json`

8. DDP regression re-check after trainer validation sync
   - Scope: validate upstream-style `validate()` DDP EMA sync integration.
   - Initial issue: missing `world_size` attribute caused DDP failure.
   - Fix: guard with `getattr(self, "world_size", 1)`.
   - Result after fix: pass.
   - Artifact: `roadmap/artifacts/phase3_ddp_smoke_after_validate_sync_fix.json`

## Notes

- Core runtime is now on `ultralytics.__version__ == 8.4.33` surface.
- Phase 1 remains focused on parity-safe updates while preserving custom feature behavior.
