# Phase 2 Model Load Matrix

Environment: Kaggle Linux (2x Tesla T4), branch `new_arch`

## Result

- Loaded successfully: 16 / 16

## Matrix

| Task | n | s | l | x |
|---|---|---|---|---|
| detect | ✅ | ✅ | ✅ | ✅ |
| segment | ✅ | ✅ | ✅ | ✅ |
| pose | ✅ | ✅ | ✅ | ✅ |
| obb | ✅ | ✅ | ✅ | ✅ |

## Notes

- `m` and `xl` are intentionally deferred for now due to channel-shape incompatibilities in current v13 graph scaling.
- Flash backend during this load check was fallback due to missing `flash_attn_turing` module in that machine state, but model-head integration remains backend-agnostic and compatible with Turing selection logic in `block.py`.
