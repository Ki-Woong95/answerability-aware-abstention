#!/usr/bin/env python3
"""Archive AR artifacts, retaining AF/MOSS rows in shared evaluation tables.

Run from the project root. Preview is the default; --apply performs the plan.
Archived JSON paths are relocated, with original bytes saved beside each edit.
"""

import argparse
import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil


MODELS = {"audio-reasoner", "audio_reasoner"}


def is_ar(row):
    return row.get("model", row.get("model_name")) in MODELS


def rewrite_paths(value, mappings):
    if isinstance(value, dict):
        return {key: rewrite_paths(item, mappings) for key, item in value.items()}
    if isinstance(value, list):
        return [rewrite_paths(item, mappings) for item in value]
    if isinstance(value, str):
        for old, new in mappings:
            if value == old or value.startswith(old + "/"):
                return new + value[len(old) :]
    return value


def atomic_write(path, content):
    temporary = path.with_name(path.name + ".ar-archive.tmp")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        handle.write(content)
    temporary.replace(path)


def build_plan(root):
    moves = set()
    for base in (
        "data/trajectories",
        "data/gates",
        "data/interventions",
        "data/worldsense/trajectories",
        "data/worldsense/gates",
        "data/worldsense/interventions",
        "data/ablations/always_idk/umwp/trajectories",
        "data/ablations/always_idk/worldsense/trajectories",
    ):
        for name in MODELS:
            path = root / base / name
            if path.exists():
                moves.add(path.relative_to(root))
    for base in (
        "data/layer_selection",
        "data/calibration",
        "data/operating_point",
        "data/worldsense/layer_selection",
        "data/worldsense/calibration",
        "data/worldsense/operating_point",
        "data/ablations/always_idk/umwp/judge",
        "data/ablations/always_idk/worldsense/judge",
    ):
        for path in (root / base).glob("*"):
            if path.is_file() and any(name in path.name for name in MODELS):
                moves.add(path.relative_to(root))
    shared = []
    for base in (
        "data/judge",
        "data/judgments",
        "data/worldsense/judge",
        "data/analysis",
        "data/worldsense/analysis",
    ):
        for path in (root / base).rglob("*"):
            if path.suffix == ".jsonl":
                lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
                rows = [json.loads(line) for line in lines if line.strip()]
                removed = sum(is_ar(row) for row in rows)
                retained = "".join(
                    line
                    for line in lines
                    if not line.strip() or not is_ar(json.loads(line))
                )
            elif path.suffix == ".csv":
                with path.open(encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    fields, rows = reader.fieldnames, list(reader)
                removed = sum(is_ar(row) for row in rows)
                output = io.StringIO(newline="")
                writer = csv.DictWriter(output, fieldnames=fields)
                writer.writeheader()
                writer.writerows(row for row in rows if not is_ar(row))
                retained = output.getvalue()
            else:
                continue
            if removed:
                shared.append(
                    {
                        "path": str(path.relative_to(root)),
                        "removed_ar_rows": removed,
                        "retained_rows": len(rows) - removed,
                        "content": retained,
                        "original_sha256": hashlib.sha256(
                            path.read_bytes()
                        ).hexdigest(),
                    }
                )
    # Existing plots combine AR with other models; keep them as historical figures.
    for base in ("data/analysis", "data/worldsense/analysis"):
        for path in (root / base).glob("*/figures"):
            if path.is_dir():
                moves.add(path.relative_to(root))
    return sorted(moves), shared


def relocate_json(path, mappings):
    original = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in original.splitlines() if line.strip()]
        updated = rewrite_paths(rows, mappings)
        content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in updated)
    else:
        rows = json.loads(original)
        updated = rewrite_paths(rows, mappings)
        content = json.dumps(updated, ensure_ascii=False, indent=2) + "\n"
    if updated != rows:
        shutil.copy2(path, path.with_name(path.name + ".original"))
        atomic_write(path, content)


def archive(root, destination, apply=False):
    root, destination = root.resolve(), destination.resolve()
    destination.relative_to(root / "data/archive")
    if destination.exists():
        raise FileExistsError(destination)
    moves, shared = build_plan(root)
    plan = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "archive_root": str(destination),
        "status": "planned",
        "moves": [str(path) for path in moves],
        "shared_tables": [
            {k: v for k, v in item.items() if k != "content"} for item in shared
        ],
    }
    if not apply:
        return plan
    destination.mkdir(parents=True)
    manifest = destination / "archive_manifest.json"
    manifest.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # Snapshot every shared file before changing any live file.
    for item in shared:
        source = root / item["path"]
        target = destination / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for relative in moves:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        (root / relative).rename(target)
    mappings = []
    for relative in moves:
        mappings.extend(
            (
                (str(root / relative), str(destination / relative)),
                (str(relative), str((destination / relative).relative_to(root))),
            )
        )
    mappings.sort(key=lambda pair: len(pair[0]), reverse=True)
    for path in destination.rglob("*"):
        if path.suffix in (".json", ".jsonl") and path != manifest:
            relocate_json(path, mappings)
    for item in shared:
        source = root / item["path"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != item["original_sha256"]:
            raise RuntimeError(f"Shared file changed during archival: {source}")
        atomic_write(source, item["content"])
    plan["status"] = "complete"
    atomic_write(manifest, json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            archive(Path.cwd(), args.archive_root, args.apply),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
