#!/usr/bin/env python3
"""Remap YOLO dataset classes by selected class IDs/names.

This utility first drops annotations for all non-selected classes, then remaps
the remaining selected classes to contiguous IDs (0..N-1), updates
data.yaml names/nc, and optionally deletes stale *.cache files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from e

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def _dump_yaml(path: Path, data: dict) -> None:
    import yaml

    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _names_to_list(names_field) -> list[str]:
    if isinstance(names_field, list):
        return [str(x) for x in names_field]
    if isinstance(names_field, dict):
        # expected format: {0: 'a', 1: 'b', ...}
        try:
            pairs = sorted(((int(k), str(v)) for k, v in names_field.items()), key=lambda x: x[0])
        except Exception as e:
            raise ValueError("Unsupported names mapping format in data.yaml") from e
        return [v for _, v in pairs]
    raise ValueError("data.yaml 'names' must be a list or dict")


def _resolve_selected_ids(
    all_names: list[str],
    include_ids: list[int],
    include_names: list[str],
) -> list[int]:
    name_to_id = {n.lower(): i for i, n in enumerate(all_names)}
    selected: list[int] = []

    for cls_id in include_ids:
        if cls_id < 0 or cls_id >= len(all_names):
            raise ValueError(f"--include-id {cls_id} out of range for dataset size {len(all_names)}")
        if cls_id not in selected:
            selected.append(cls_id)

    for cls_name in include_names:
        key = cls_name.strip().lower()
        if key not in name_to_id:
            raise ValueError(f"--include-name '{cls_name}' not found in dataset names={all_names}")
        cls_id = name_to_id[key]
        if cls_id not in selected:
            selected.append(cls_id)

    if not selected:
        raise ValueError("No selected classes. Use --include-id and/or --include-name")

    return selected


def _resolve_dataset_root(data_yaml: Path, data_dict: dict, root_override: str | None) -> Path:
    if root_override:
        return Path(root_override).resolve()

    path_field = data_dict.get("path")
    if path_field:
        p = Path(path_field)
        return p.resolve() if p.is_absolute() else (data_yaml.parent / p).resolve()

    return data_yaml.parent.resolve()


def _resolve_label_dirs(data_dict: dict, dataset_root: Path, explicit_label_dirs: list[str]) -> list[Path]:
    if explicit_label_dirs:
        dirs = [Path(p).resolve() for p in explicit_label_dirs]
        return [d for d in dirs if d.exists()]

    dirs: list[Path] = []
    for key in ("train", "val", "valid", "test"):
        val = data_dict.get(key)
        if not isinstance(val, str):
            continue

        p = Path(val)
        p = p.resolve() if p.is_absolute() else (dataset_root / p).resolve()

        cand: Path | None = None
        if p.name == "images":
            cand = p.parent / "labels"
        elif p.name == "labels":
            cand = p
        elif "/images/" in p.as_posix():
            cand = Path(p.as_posix().replace("/images/", "/labels/"))
        else:
            maybe = p.parent / "labels"
            if maybe.exists():
                cand = maybe

        if cand and cand.exists() and cand not in dirs:
            dirs.append(cand)

    return dirs


def _parse_label_class(token: str) -> int:
    # Supports integer or float-like class token (e.g. '9' or '9.0')
    return int(float(token))


def _filter_then_remap_labels(label_dirs: list[Path], old_to_new: dict[int, int], dry_run: bool) -> dict:
    summary = {
        "files_total": 0,
        "files_touched": 0,
        "lines_total": 0,
        "lines_selected": 0,
        "lines_kept": 0,
        "lines_dropped": 0,
        "lines_remapped": 0,
        "parse_errors": 0,
    }
    selected_ids = set(old_to_new)

    for label_dir in label_dirs:
        for txt in label_dir.rglob("*.txt"):
            summary["files_total"] += 1
            src = txt.read_text(encoding="utf-8").splitlines()
            kept_parts: list[list[str]] = []
            out_lines: list[str] = []
            changed = False

            # Phase 1: keep only selected classes and drop everything else.
            for ln in src:
                s = ln.strip()
                if not s:
                    continue
                summary["lines_total"] += 1

                parts = s.split()
                if not parts:
                    continue
                try:
                    old_id = _parse_label_class(parts[0])
                except Exception:
                    summary["parse_errors"] += 1
                    changed = True
                    summary["lines_dropped"] += 1
                    continue

                if old_id not in selected_ids:
                    changed = True
                    summary["lines_dropped"] += 1
                    continue

                kept_parts.append(parts)
                summary["lines_selected"] += 1

            # Phase 2: remap selected classes to contiguous IDs.
            for parts in kept_parts:
                old_id = _parse_label_class(parts[0])
                new_id = old_to_new[old_id]
                if new_id != old_id or parts[0] != str(new_id):
                    changed = True
                    summary["lines_remapped"] += 1
                parts[0] = str(new_id)
                out_lines.append(" ".join(parts))
                summary["lines_kept"] += 1

            if changed:
                summary["files_touched"] += 1
                if not dry_run:
                    txt.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")

    return summary


def _delete_caches(dataset_root: Path, dry_run: bool) -> list[str]:
    removed: list[str] = []
    for p in dataset_root.rglob("*.cache"):
        removed.append(str(p))
        if not dry_run:
            p.unlink(missing_ok=True)
    return removed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Keep/remap selected classes in a YOLO dataset")
    p.add_argument("--data", required=True, help="Path to dataset data.yaml")
    p.add_argument("--include-id", type=int, action="append", default=[], help="Class ID to keep (repeatable)")
    p.add_argument(
        "--include-name",
        action="append",
        default=[],
        help="Class name to keep (repeatable, case-insensitive)",
    )
    p.add_argument("--root", default=None, help="Optional dataset root override")
    p.add_argument(
        "--labels-dir",
        action="append",
        default=[],
        help="Optional explicit labels dir (repeatable). If omitted, inferred from data.yaml",
    )
    p.add_argument(
        "--out-data",
        default="",
        help="Optional output YAML path. Default overwrites input data.yaml unless --dry-run",
    )
    p.add_argument("--keep-cache", action="store_true", help="Do not delete *.cache files")
    p.add_argument("--dry-run", action="store_true", help="Preview changes without writing files")
    return p


def main() -> None:
    args = build_parser().parse_args()

    data_yaml = Path(args.data).resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    data_dict = _load_yaml(data_yaml)
    all_names = _names_to_list(data_dict.get("names"))
    selected_ids = _resolve_selected_ids(all_names, args.include_id, args.include_name)

    new_names = [all_names[i] for i in selected_ids]
    old_to_new = {old: new for new, old in enumerate(selected_ids)}

    dataset_root = _resolve_dataset_root(data_yaml, data_dict, args.root)
    label_dirs = _resolve_label_dirs(data_dict, dataset_root, args.labels_dir)
    if not label_dirs:
        raise RuntimeError("No label directories found. Provide --labels-dir explicitly.")

    remap_summary = _filter_then_remap_labels(label_dirs, old_to_new, args.dry_run)

    out_yaml = Path(args.out_data).resolve() if args.out_data else data_yaml
    updated = dict(data_dict)
    updated["nc"] = len(new_names)
    updated["names"] = new_names

    if not args.dry_run:
        _dump_yaml(out_yaml, updated)

    removed_cache = []
    if not args.keep_cache:
        removed_cache = _delete_caches(dataset_root, args.dry_run)

    report = {
        "dry_run": args.dry_run,
        "data_yaml": str(data_yaml),
        "out_data_yaml": str(out_yaml),
        "dataset_root": str(dataset_root),
        "label_dirs": [str(x) for x in label_dirs],
        "old_names": all_names,
        "selected_ids": selected_ids,
        "new_names": new_names,
        "old_to_new": old_to_new,
        "remap_summary": remap_summary,
        "cache_files_removed": removed_cache,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
