# Spec 02 - DDP Hardening

## Goal
Improve distributed training reliability for this custom YOLOv13 fork.

## Problems Addressed
- DDP subprocess may import wrong package path and fail on custom modules.
- Save directory deletion can fail on missing directories.
- Argument serialization can fail for non-primitive fields.
- Some torch backends are fragile with bool broadcast tensors.

## Implementation Plan
1. Patch `ultralytics/utils/dist.py`:
   - Force repo path insertion in generated DDP temp entrypoint.
   - Serialize augmentation overrides defensively.
   - Make save-dir cleanup safer.
2. Patch `ultralytics/engine/trainer.py` AMP DDP broadcast:
   - cast `self.amp` to int before broadcast.
3. Validate via a 2-GPU smoke train.

## Dependency Diagram
```text
train(device='0,1')
  -> generate_ddp_command
  -> generated temp python file
  -> imports local repo ultralytics
  -> per-rank train startup
```

## Detailed Task List
- [x] Update DDP temp runner generation.
- [x] Add safer override serialization.
- [x] Harden save-dir cleanup.
- [x] Patch AMP broadcast type in trainer.
- [x] Confirm DDP smoke run success.
- [x] Confirm DDP 5-epoch run success.
- [x] Add non-finite loss guard for distributed stability.
- [x] Apply DDP bucket-view mitigation for stride mismatch warnings.

## Edge Cases
- Editable install absent but source tree must be imported.
- Augmentation objects inside args.
- Temporary file cleanup after failures.
