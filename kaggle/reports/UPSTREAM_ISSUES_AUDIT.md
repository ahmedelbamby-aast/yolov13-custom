# Upstream Issues Audit (iMoonLab/yolov13)

This audit summarizes upstream issues and how they were handled in this fork.

## Scope Summary

- Total upstream issues reviewed: 73
- Focused issues in this pass: 12
- Priority focus: DDP stability, export reproducibility, Kaggle workflow reliability

## Resolution Matrix

| Issue | State | Category | Status in this fork | Notes |
|---|---|---|---|---|
| [#15](https://github.com/iMoonLab/yolov13/issues/15) | open | data | resolved | DDP subprocess import path hardened in `ultralytics/utils/dist.py` to avoid custom module resolution failures. |
| [#21](https://github.com/iMoonLab/yolov13/issues/21) | closed | data,model,env | resolved | Same root cause as #15; local source import precedence enforced for DDP temp runner. |
| [#37](https://github.com/iMoonLab/yolov13/issues/37) | open | ddp | partial | Added non-finite loss guard in train loop and safer DDP settings in `ultralytics/engine/trainer.py`. |
| [#40](https://github.com/iMoonLab/yolov13/issues/40) | open | ddp | resolved | Mitigated DDP grad-stride instability by setting `gradient_as_bucket_view=False` in DDP wrapper. |
| [#48](https://github.com/iMoonLab/yolov13/issues/48) | open | ddp,env | resolved | Applied same DDP stability fixes and validated with distributed runs on 2x T4. |
| [#53](https://github.com/iMoonLab/yolov13/issues/53) | open | ddp | resolved | Applied same DDP stability fixes and validated with distributed runs on 2x T4. |
| [#54](https://github.com/iMoonLab/yolov13/issues/54) | open | export,env | resolved | Added TensorRT export script and test path in `kaggle/scripts/70_export_onnx_tensorrt.sh`. |
| [#61](https://github.com/iMoonLab/yolov13/issues/61) | open | export | resolved | Added ONNX export testing path and parameter matrix in scripts for reproducibility. |
| [#65](https://github.com/iMoonLab/yolov13/issues/65) | open | data,model | partial | Documented augmentation edge-case handling in dependency and smoke workflows. |
| [#72](https://github.com/iMoonLab/yolov13/issues/72) | open | model | not_resolved | Not fully resolved: upstream feature parity limitations (segment variants) remain architecture-specific. |
| [#74](https://github.com/iMoonLab/yolov13/issues/74) | open | model | not_resolved | Not fully resolved: OBB/feature availability and model support limitations remain upstream dependent. |
| [#75](https://github.com/iMoonLab/yolov13/issues/75) | open | data,model | partial | Partially addressed through robust run scripts; dataset-key mismatches still require dataset yaml correction. |

## Notes

- Some upstream issues are architectural feature requests (e.g., full OBB/segment/pose parity) and cannot be fully solved by runtime hardening alone.
- This fork prioritizes training pipeline reliability, distributed compatibility, and export workflow reproducibility on Kaggle.
