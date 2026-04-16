#!/usr/bin/env python3
"""Validate alignment artifact schemas before running gates."""

from __future__ import annotations

import json
import sys

from common_artifacts import (
    feature_artifacts_dir,
    load_baseline,
    load_custom_feature_registry,
    load_parity_exceptions,
    load_parity_inventory,
    save_gate_json,
    validate_exception_records,
)


def main() -> None:
    artifacts = feature_artifacts_dir()

    baseline = load_baseline()
    parity = load_parity_inventory()
    registry = load_custom_feature_registry()
    exceptions = load_parity_exceptions()

    failures: list[str] = []

    for key in ("repo_url", "reference_type", "reference_value", "resolved_commit", "scope_definition"):
        if not baseline.get(key):
            failures.append(f"baseline missing '{key}'")

    if not isinstance(parity.get("workflows", []), list) or not parity.get("workflows", []):
        failures.append("parity-inventory missing non-empty 'workflows' list")

    if not isinstance(registry.get("features", []), list) or not registry.get("features", []):
        failures.append("custom-feature-registry missing non-empty 'features' list")

    exc_validation = validate_exception_records(exceptions.get("exceptions", []))
    if exc_validation["incomplete"] > 0:
        failures.append("parity-exceptions contains incomplete records")

    report = {
        "status": "ok" if not failures else "fail",
        "artifacts_dir": str(artifacts),
        "checks": {
            "baseline": bool(baseline),
            "parity_workflows": len(parity.get("workflows", [])),
            "custom_features": len(registry.get("features", [])),
            "exception_validation": exc_validation,
        },
        "failures": failures,
    }

    out_path = save_gate_json("alignment-schema-check.json", report)
    print(json.dumps(report, indent=2))
    print(f"saved={out_path}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
