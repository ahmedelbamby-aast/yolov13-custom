# Tasks: Upstream Alignment with Custom Feature Preservation

**Input**: Design documents from `C:\Users\Ahmed\yolov13_custom\specs\001-align-upstream-custom\`
**Prerequisites**: `plan.md` (required), `spec.md` (required), `research.md`, `data-model.md`, `contracts/`, `quickstart.md`

**Tests**: Included because the feature specification requires mandatory compatibility and custom-regression gates.

**Organization**: Tasks are grouped by user story so each story is independently implementable and testable.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependency on incomplete tasks)
- **[Story]**: User story label (`[US1]`, `[US2]`, `[US3]`)
- All task descriptions include exact file paths

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Initialize alignment artifacts and baseline references used by all stories.

- [X] T001 Create alignment artifact directory index in `C:\Users\Ahmed\yolov13_custom\specs\001-align-upstream-custom\artifacts\README.md`
- [X] T002 Pin upstream baseline metadata in `C:\Users\Ahmed\yolov13_custom\specs\001-align-upstream-custom\artifacts\upstream-baseline.json`
- [X] T003 [P] Seed parity inventory template in `C:\Users\Ahmed\yolov13_custom\specs\001-align-upstream-custom\artifacts\parity-inventory.yaml`
- [X] T004 [P] Seed custom feature registry template in `C:\Users\Ahmed\yolov13_custom\specs\001-align-upstream-custom\artifacts\custom-feature-registry.yaml`
- [X] T005 [P] Seed parity exception register template in `C:\Users\Ahmed\yolov13_custom\specs\001-align-upstream-custom\artifacts\parity-exceptions.yaml`
- [X] T006 Configure release evidence template in `C:\Users\Ahmed\yolov13_custom\specs\001-align-upstream-custom\artifacts\release-evidence.yaml`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Build shared parity/custom gate infrastructure required before story work.

**CRITICAL**: No user story implementation begins until this phase is complete.

- [X] T007 Implement shared artifact IO helpers in `C:\Users\Ahmed\yolov13_custom\kaggle\scripts\phase3_upgrade\common_artifacts.py`
- [X] T008 [P] Implement alignment schema validation entrypoint in `C:\Users\Ahmed\yolov13_custom\kaggle\scripts\phase3_upgrade\00_alignment_schema_check.py`
- [X] T009 [P] Implement baseline and exception data loaders in `C:\Users\Ahmed\yolov13_custom\kaggle\scripts\34_phase3_custom_delta_audit.py`
- [X] T010 [P] Implement normalized gate-result serializer in `C:\Users\Ahmed\yolov13_custom\kaggle\scripts\36_phase3_final_gate.py`
- [X] T011 Add release-blocking decision aggregator in `C:\Users\Ahmed\yolov13_custom\kaggle\scripts\phase3_upgrade\03_stress_gate.py`
- [X] T012 Add shared artifact path constants for scripts in `C:\Users\Ahmed\yolov13_custom\scripts\_common.py`
- [X] T013 Add operator run order for parity/custom gates in `C:\Users\Ahmed\yolov13_custom\specs\001-align-upstream-custom\quickstart.md`

**Checkpoint**: Foundation complete; user stories can proceed.

---

## Phase 3: User Story 1 - Use Fork Like Upstream (Priority: P1) 🎯 MVP

**Goal**: Deliver upstream-equivalent API and CLI behavior for core workflows.

**Independent Test**: Upstream-style Python and CLI workflows execute on the fork without fork-specific call changes.

### Tests for User Story 1

- [ ] T014 [P] [US1] Add import/constructor parity smoke coverage in `C:\Users\Ahmed\yolov13_custom\tests\test_python.py`
- [ ] T015 [P] [US1] Add CLI mode parity smoke coverage in `C:\Users\Ahmed\yolov13_custom\tests\test_cli.py`
- [ ] T016 [P] [US1] Add API-style wrapper parity matrix test in `C:\Users\Ahmed\yolov13_custom\tests\test_integrations.py`

### Implementation for User Story 1

- [ ] T017 [US1] Align top-level lazy exports and model availability semantics in `C:\Users\Ahmed\yolov13_custom\ultralytics\__init__.py`
- [ ] T018 [US1] Align CLI entry argument normalization with upstream behavior in `C:\Users\Ahmed\yolov13_custom\ultralytics\cfg\__init__.py`
- [ ] T019 [US1] Align model/task mode dispatch defaults in `C:\Users\Ahmed\yolov13_custom\ultralytics\engine\model.py`
- [ ] T020 [US1] Align core developer script passthrough semantics in `C:\Users\Ahmed\yolov13_custom\scripts\train.py`, `C:\Users\Ahmed\yolov13_custom\scripts\val.py`, `C:\Users\Ahmed\yolov13_custom\scripts\test.py`, `C:\Users\Ahmed\yolov13_custom\scripts\predict.py`, `C:\Users\Ahmed\yolov13_custom\scripts\export.py`, `C:\Users\Ahmed\yolov13_custom\scripts\benchmark.py`
- [ ] T021 [US1] Align API-style wrappers to upstream call patterns in `C:\Users\Ahmed\yolov13_custom\scripts\api_style\common.py`, `C:\Users\Ahmed\yolov13_custom\scripts\api_style\train_api.py`, `C:\Users\Ahmed\yolov13_custom\scripts\api_style\val_api.py`, `C:\Users\Ahmed\yolov13_custom\scripts\api_style\test_api.py`, `C:\Users\Ahmed\yolov13_custom\scripts\api_style\predict_api.py`, `C:\Users\Ahmed\yolov13_custom\scripts\api_style\export_api.py`, `C:\Users\Ahmed\yolov13_custom\scripts\api_style\benchmark_api.py`
- [ ] T022 [US1] Record US1 parity status and evidence references in `C:\Users\Ahmed\yolov13_custom\specs\001-align-upstream-custom\artifacts\parity-inventory.yaml`
- [X] T043 [US1] Implement SC-001 metric computation using WorkflowParityItem ratio in `C:\Users\Ahmed\yolov13_custom\kaggle\scripts\phase3_upgrade\02_cli_python_parity_gate.py`

**Checkpoint**: User Story 1 is independently functional and testable.

---

## Phase 4: User Story 2 - Preserve Custom Fork Value (Priority: P1)

**Goal**: Preserve all release-blocking custom features while alignment changes are applied.

**Independent Test**: Custom feature registry workflows pass regression gates with no critical degradation.

### Tests for User Story 2

- [ ] T023 [P] [US2] Add flash-backend toggle regression coverage in `C:\Users\Ahmed\yolov13_custom\tests\test_engine.py`
- [ ] T024 [P] [US2] Add v13 multi-task config regression coverage in `C:\Users\Ahmed\yolov13_custom\tests\test_exports.py`
- [ ] T025 [P] [US2] Add DDP and task-preflight custom gate assertions in `C:\Users\Ahmed\yolov13_custom\kaggle\scripts\35_phase_ddp_gate.py` and `C:\Users\Ahmed\yolov13_custom\kaggle\scripts\33_phase1_task_preflight_smoke.py`

### Implementation for User Story 2

- [X] T026 [US2] Normalize flash-mode vocabulary and env precedence in `C:\Users\Ahmed\yolov13_custom\scripts\_common.py` and `C:\Users\Ahmed\yolov13_custom\kaggle\scripts\common.sh`
- [ ] T027 [US2] Preserve additive namespaced core v13 detect configs in `C:\Users\Ahmed\yolov13_custom\ultralytics\cfg\models\v13\yolov13.yaml`, `C:\Users\Ahmed\yolov13_custom\ultralytics\cfg\models\v13\yolov13s.yaml`, `C:\Users\Ahmed\yolov13_custom\ultralytics\cfg\models\v13\yolov13l.yaml`, `C:\Users\Ahmed\yolov13_custom\ultralytics\cfg\models\v13\yolov13x.yaml`
- [ ] T044 [US2] Preserve additive namespaced v13 task configs for segment and pose in `C:\Users\Ahmed\yolov13_custom\ultralytics\cfg\models\v13\yolov13-seg.yaml`, `C:\Users\Ahmed\yolov13_custom\ultralytics\cfg\models\v13\yolov13s-seg.yaml`, `C:\Users\Ahmed\yolov13_custom\ultralytics\cfg\models\v13\yolov13l-seg.yaml`, `C:\Users\Ahmed\yolov13_custom\ultralytics\cfg\models\v13\yolov13x-seg.yaml`, `C:\Users\Ahmed\yolov13_custom\ultralytics\cfg\models\v13\yolov13-pose.yaml`, `C:\Users\Ahmed\yolov13_custom\ultralytics\cfg\models\v13\yolov13s-pose.yaml`, `C:\Users\Ahmed\yolov13_custom\ultralytics\cfg\models\v13\yolov13l-pose.yaml`, `C:\Users\Ahmed\yolov13_custom\ultralytics\cfg\models\v13\yolov13x-pose.yaml`
- [ ] T045 [US2] Preserve additive namespaced v13 task configs for obb and variant families in `C:\Users\Ahmed\yolov13_custom\ultralytics\cfg\models\v13\yolov13-obb.yaml`, `C:\Users\Ahmed\yolov13_custom\ultralytics\cfg\models\v13\yolov13s-obb.yaml`, `C:\Users\Ahmed\yolov13_custom\ultralytics\cfg\models\v13\yolov13l-obb.yaml`, `C:\Users\Ahmed\yolov13_custom\ultralytics\cfg\models\v13\yolov13x-obb.yaml`, `C:\Users\Ahmed\yolov13_custom\ultralytics\cfg\models\v13\yolov13_2.yaml`, `C:\Users\Ahmed\yolov13_custom\ultralytics\cfg\models\v13\yolov13s_2.yaml`, `C:\Users\Ahmed\yolov13_custom\ultralytics\cfg\models\v13\yolov13l_2.yaml`, `C:\Users\Ahmed\yolov13_custom\ultralytics\cfg\models\v13\yolov13x_2.yaml`, `C:\Users\Ahmed\yolov13_custom\ultralytics\cfg\models\v13\yolov13l_2_p2.yaml`, `C:\Users\Ahmed\yolov13_custom\ultralytics\cfg\models\v13\yolov13n-seg.yaml`, `C:\Users\Ahmed\yolov13_custom\ultralytics\cfg\models\v13\yolov13n-pose.yaml`, `C:\Users\Ahmed\yolov13_custom\ultralytics\cfg\models\v13\yolov13n-obb.yaml`
- [ ] T028 [US2] Align task-aware preflight behavior without removing custom checks in `C:\Users\Ahmed\yolov13_custom\ultralytics\data\utils.py` and `C:\Users\Ahmed\yolov13_custom\ultralytics\engine\trainer.py`
- [ ] T029 [US2] Update custom feature preservation statuses in `C:\Users\Ahmed\yolov13_custom\specs\001-align-upstream-custom\artifacts\custom-feature-registry.yaml`
- [X] T030 [US2] Publish preserved custom usage and migration notes in `C:\Users\Ahmed\yolov13_custom\README.md` and `C:\Users\Ahmed\yolov13_custom\scripts\README.md`

**Checkpoint**: User Story 2 is independently functional and testable.

---

## Phase 5: User Story 3 - Govern and Audit Parity Drift (Priority: P2)

**Goal**: Produce auditable parity, exception, and release evidence for maintainer decisions.

**Independent Test**: A release candidate produces complete parity inventory, exception records, and deterministic approve/block decision.

### Tests for User Story 3

- [ ] T031 [P] [US3] Add exception-completeness validation coverage in `C:\Users\Ahmed\yolov13_custom\tests\test_integrations.py`
- [ ] T032 [P] [US3] Add release-blocking decision-path coverage in `C:\Users\Ahmed\yolov13_custom\tests\test_engine.py`

### Implementation for User Story 3

- [X] T033 [US3] Implement parity exception record generation with owner/risk/remediation fields in `C:\Users\Ahmed\yolov13_custom\kaggle\scripts\34_phase3_custom_delta_audit.py`
- [X] T034 [US3] Implement release evidence package assembler and decision writer in `C:\Users\Ahmed\yolov13_custom\kaggle\scripts\36_phase3_final_gate.py`
- [X] T035 [US3] Emit consolidated machine-readable status from gate runner in `C:\Users\Ahmed\yolov13_custom\kaggle\scripts\phase3_upgrade\05_phase6_gate_runner.sh`
- [X] T036 [US3] Persist cycle release evidence artifacts in `C:\Users\Ahmed\yolov13_custom\specs\001-align-upstream-custom\artifacts\release-evidence.yaml`
- [X] T037 [US3] Update parity contract and operator governance flow in `C:\Users\Ahmed\yolov13_custom\specs\001-align-upstream-custom\contracts\parity-contract.md` and `C:\Users\Ahmed\yolov13_custom\specs\001-align-upstream-custom\quickstart.md`

**Checkpoint**: User Story 3 is independently functional and testable.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final validation, documentation consistency, and publication readiness.

- [ ] T038 [P] Run full parity stress and CLI/Python gate suite in `C:\Users\Ahmed\yolov13_custom\kaggle\scripts\phase3_upgrade\03_stress_gate.py` and `C:\Users\Ahmed\yolov13_custom\kaggle\scripts\phase3_upgrade\02_cli_python_parity_gate.py`
- [X] T039 [P] Refresh alignment tracking documentation in `C:\Users\Ahmed\yolov13_custom\roadmap\phase3_upgrade\README.md` and `C:\Users\Ahmed\yolov13_custom\README.md`
- [ ] T040 Execute quickstart end-to-end validation and sign-off updates in `C:\Users\Ahmed\yolov13_custom\specs\001-align-upstream-custom\quickstart.md` and `C:\Users\Ahmed\yolov13_custom\specs\001-align-upstream-custom\artifacts\release-evidence.yaml`
- [ ] T041 Record approved exceptions and remediation schedule in `C:\Users\Ahmed\yolov13_custom\specs\001-align-upstream-custom\artifacts\parity-exceptions.yaml`
- [X] T042 Record canonical publication readiness for `https://github.com/ahmedelbamby-aast/yolov13-custom` in `C:\Users\Ahmed\yolov13_custom\specs\001-align-upstream-custom\artifacts\release-evidence.yaml`
- [ ] T046 Execute canonical publication push to `https://github.com/ahmedelbamby-aast/yolov13-custom` and persist push evidence in `C:\Users\Ahmed\yolov13_custom\specs\001-align-upstream-custom\artifacts\release-evidence.yaml`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies.
- **Phase 2 (Foundational)**: Depends on Phase 1; blocks all user stories.
- **Phase 3 (US1)**: Depends on Phase 2.
- **Phase 4 (US2)**: Depends on Phase 2; can run in parallel with US1 after foundation is complete.
- **Phase 5 (US3)**: Depends on Phase 3 and Phase 4 evidence artifacts.
- **Phase 6 (Polish)**: Depends on all user stories being complete.

### User Story Dependencies

- **US1**: Independent after foundational phase.
- **US2**: Independent after foundational phase.
- **US3**: Requires outputs from US1 parity artifacts and US2 custom-regression artifacts.

### Within Each User Story

- Story tests/gates before story implementation updates.
- Core behavior alignment before artifact status updates.
- Evidence capture before story checkpoint sign-off.
- Canonical publication push task (T046) is release-blocking and must complete before sign-off.

---

## Parallel Execution Examples

### User Story 1

```bash
# Parallel test prep for US1
T014, T015, T016

# Then implementation sequence
T017 -> T018 -> T019 -> T020 -> T021 -> T022 -> T043
```

### User Story 2

```bash
# Parallel regression coverage setup for US2
T023, T024, T025

# Then implementation sequence
T026 -> T027 -> T044 -> T045 -> T028 -> T029 -> T030
```

### User Story 3

```bash
# Parallel governance test coverage for US3
T031, T032

# Then implementation sequence
T033 -> T034 -> T035 -> T036 -> T037
```

---

## Implementation Strategy

### MVP First (User Story 1)

1. Complete Phase 1 and Phase 2.
2. Complete Phase 3 (US1).
3. Validate upstream-equivalent usage flows and parity artifacts.
4. Demo/review before expanding scope.

### Incremental Delivery

1. Deliver US1 (upstream parity baseline).
2. Deliver US2 (custom feature preservation hardening).
3. Deliver US3 (audit/governance automation).
4. Finish with Phase 6 polish and publication readiness.

### Parallel Team Strategy

1. Shared completion of Setup + Foundational phases.
2. Split US1 and US2 across developers after foundation checkpoint.
3. Start US3 once artifact outputs from US1/US2 are available.

---

## Notes

- `[P]` tasks are safe for parallel execution when dependencies are satisfied.
- `[USx]` labels map each task to an independently testable user story.
- Keep parity and custom evidence artifacts updated continuously to avoid end-phase audit drift.
