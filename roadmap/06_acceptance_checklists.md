# 06 Acceptance Checklists

## Engineering Checklist

- [ ] v13 seg/pose/obb YAMLs added and loadable.
- [ ] No detect regression on baseline smoke runs.
- [ ] OBB metrics dictionary/key alignment verified.
- [ ] Task preflight validators fail fast with actionable errors.
- [ ] DDP smoke run passes for detect/segment/pose/obb.

## QA Checklist

- [ ] Tiny-dataset smoke train succeeds per task.
- [ ] `val()` output contains expected metrics and plots.
- [ ] `predict()` works for each task type.
- [ ] Export tested per task with documented support matrix.
- [ ] Artifacts (weights, curves, logs, config snapshot) saved correctly.

## Docs Checklist

- [ ] Task-specific quickstarts added (seg/pose/obb).
- [ ] Dataset formatting examples added for each task.
- [ ] Troubleshooting section mapped to common errors.
- [ ] Version matrix and known limitations published.

## Release Criteria

- [ ] All phase gates passed.
- [ ] Benchmark summary generated.
- [ ] Repro script and environment pin file included.
