#!/usr/bin/env python3
"""Plan/apply pair-level UMWP V10 question-leakage exclusions.

Run at project root alongside audit_umwp_split_leakage.py:
    python resolve_umwp_leakage.py             # dry run (no active-manifest edits)
    python resolve_umwp_leakage.py --apply     # backup + replace manifest

Only pair IDs are excluded; raw JSONL and audio are never touched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

from audit_umwp_split_leakage import (
    SPLITS,
    canonical_id,
    load_original_pairs,
    load_split,
    normalize_question,
)

PRIORITY = {"test": 0, "val": 1, "calibration": 2, "gate_train": 3}
EXPECTED_PAIR_COUNT = 2566
EXPECTED_SAME_PAIR = 5
EXPECTED_CROSS_A_GROUPS = 3
EXPECTED_A_DUP_GROUPS = 5


def stop(msg):
    raise SystemExit(f"STOP: {msg}")


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        dir=path.parent,
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as f:
        tmp = Path(f.name)
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def duplicate_groups(selected, sources, location, label=None):
    buckets = defaultdict(list)
    for pid in sorted(selected):
        for lab in (label,) if label else ("A", "U"):
            key = normalize_question(sources[pid][lab])
            buckets[key].append((pid, lab, location[pid]))
    return [
        (key, records)
        for key, records in buckets.items()
        if len({p for p, _, _ in records}) > 1
    ]


def cross_groups(groups):
    return [(k, rows) for k, rows in groups if len({split for _, _, split in rows}) > 1]


def summarize_removal(pid, reasons, location):
    return {"pair_id": pid, "split": location[pid], "reasons": sorted(reasons[pid])}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".", type=Path)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    root = args.root.resolve()
    active = root / "data/splits/umwp_final_split.json"
    original = root / "data/raw/umwp/StandardDataset.jsonl"
    if not active.is_file() or not original.is_file():
        stop("Expected active split manifest and raw UMWP file do not exist")

    current_bytes = active.read_bytes()
    current_hash = hashlib.sha256(current_bytes).hexdigest()
    splits = load_split(active)
    if sum(map(len, splits.values())) != EXPECTED_PAIR_COUNT:
        stop(
            f"Expected {EXPECTED_PAIR_COUNT} QC-retained pairs, found {sum(map(len, splits.values()))}"
        )
    selected = set().union(*splits.values())
    location = {pid: s for s, ids in splits.items() for pid in ids}
    source = load_original_pairs(original, 2600)
    if not selected.issubset(source):
        stop("Some split IDs do not exist in original pairs")

    same = {
        pid
        for pid in selected
        if normalize_question(source[pid]["A"]) == normalize_question(source[pid]["U"])
    }
    a_groups = duplicate_groups(selected, source, location, "A")
    a_cross = cross_groups(a_groups)
    if (len(same), len(a_cross), len(a_groups)) != (
        EXPECTED_SAME_PAIR,
        EXPECTED_CROSS_A_GROUPS,
        EXPECTED_A_DUP_GROUPS,
    ):
        stop(
            "Leakage findings changed since review: "
            f"same A/U={len(same)}, cross-split A groups={len(a_cross)}, "
            f"all A duplicate groups={len(a_groups)}. Reaudit before proceeding"
        )

    reasons = defaultdict(set)
    for pid in same:
        reasons[pid].add("identical_A_and_U_question")
    survivors = []
    for key, records in sorted(a_groups, key=lambda item: item[0]):
        # Exclude inconsistent A/U pairs before choosing which duplicate A question to retain.
        candidate_ids = sorted({pid for pid, _, _ in records} - same)
        if not candidate_ids:
            survivors.append(
                {
                    "A_question_sha256": hashlib.sha256(key.encode()).hexdigest(),
                    "kept": None,
                    "duplicate_pair_ids": sorted({pid for pid, _, _ in records}),
                }
            )
            continue
        keep = min(
            candidate_ids,
            key=lambda pid: (PRIORITY[location[pid]], int(pid.split("_")[-1])),
        )
        survivors.append(
            {
                "A_question_sha256": hashlib.sha256(key.encode()).hexdigest(),
                "kept": keep,
                "kept_split": location[keep],
                "duplicate_pair_ids": sorted({pid for pid, _, _ in records}),
                "duplicate_splits": sorted({split for _, _, split in records}),
            }
        )
        for pid in candidate_ids:
            if pid != keep:
                reasons[pid].add("duplicate_A_question")
    excluded = set(reasons)
    remaining = selected - excluded

    residual_a = duplicate_groups(remaining, source, location, "A")
    residual_cross_any = cross_groups(duplicate_groups(remaining, source, location))
    residual_same = [
        pid
        for pid in remaining
        if normalize_question(source[pid]["A"]) == normalize_question(source[pid]["U"])
    ]
    if residual_a or residual_cross_any or residual_same:
        stop(
            f"Proposed removals did not resolve all issues: A duplicate groups={len(residual_a)}, "
            f"cross-split any-label groups={len(residual_cross_any)}, identical A/U={len(residual_same)}"
        )

    data = json.loads(current_bytes)
    container = data.get("splits", data)
    if not isinstance(container, dict) or any(
        s not in container or not isinstance(container[s], list) for s in SPLITS
    ):
        stop("Unexpected split manifest schema")
    for obj in (data, container):
        for key in obj:
            if (
                key not in SPLITS
                and key != "splits"
                and re.search(
                    r"count|n_pairs|n_trials|num_pairs|num_trials|total", key, re.I
                )
            ):
                stop(f"Manifest includes potentially stale cached counts: {key}")
    updated = {}
    counts = {}
    for split in SPLITS:
        entries = container[split]
        ids = [canonical_id(x) for x in entries]
        if set(ids) != splits[split] or len(ids) not in (
            len(splits[split]),
            2 * len(splits[split]),
        ):
            stop(f"Inconsistent pair/trial entries in {split}")
        updated[split] = [
            item for item, pid in zip(entries, ids) if pid not in excluded
        ]
        counts[split] = {
            "before": len(splits[split]),
            "excluded": len(splits[split] & excluded),
            "after": len(splits[split] - excluded),
        }
    for split in SPLITS:
        container[split] = updated[split]
    assert sum(x["after"] for x in counts.values()) == len(remaining)

    report = {
        "policy": "Pair-level exclusion of identical A/U questions; keep one A-question copy per exact-normalized duplicate group; prefer test, val, calibration, gate_train; smaller numeric ID ties",
        "original_manifest_sha256": current_hash,
        "normalization": "NFKC + casefold + whitespace collapse + remove whitespace before punctuation",
        "pre_filter_pairs": len(selected),
        "post_filter_pairs": len(remaining),
        "same_pair_identical_count": len(same),
        "all_duplicate_A_question_groups": len(a_groups),
        "cross_split_duplicate_A_question_groups": len(a_cross),
        "counts": counts,
        "exclusions": [
            summarize_removal(pid, reasons, location) for pid in sorted(excluded)
        ],
        "duplicate_A_group_decisions": survivors,
        "post_checks": {
            "same_pair_identical": len(residual_same),
            "duplicate_A_question_groups": len(residual_a),
            "cross_split_duplicate_any_label_groups": len(residual_cross_any),
        },
        "raw_and_wav_preserved": True,
    }
    qc_dir = root / "data/qc"
    plan = qc_dir / "umwp_leakage_removal_plan.json"
    atomic_json(plan, report)
    print(
        f"Existing QC-retained pairs: {len(selected)}; proposed exclusions: {len(excluded)}; remaining: {len(remaining)}"
    )
    for s in SPLITS:
        c = counts[s]
        print(f"{s}: {c['before']} -> {c['after']} (-{c['excluded']})")
    print("Excluded pair IDs:")
    for item in report["exclusions"]:
        print(f"  {item['pair_id']}@{item['split']}: {', '.join(item['reasons'])}")
    print("Duplicate A-question decisions:")
    for item in survivors:
        print(
            f"  {','.join(item['duplicate_pair_ids'])} -> keep {item['kept']}@{item.get('kept_split', 'none')}"
        )
    print(f"Review plan: {plan}")
    if not args.apply:
        print(
            "DRY RUN: active manifest unchanged; inspect plan, then rerun with --apply"
        )
        return

    backup = active.with_name("umwp_final_split.before_leakage_qc.json")
    if backup.exists():
        if backup.read_bytes() != current_bytes:
            stop(f"Existing backup differs from current pre-apply manifest: {backup}")
    else:
        with backup.open("xb") as f:
            f.write(current_bytes)
            f.flush()
            os.fsync(f.fileno())
    atomic_json(active, data)
    actual = load_split(active)
    if sum(len(x) for x in actual.values()) != len(remaining) or any(
        actual[s] != splits[s] - excluded for s in SPLITS
    ):
        stop(
            "Postwrite split integrity check failed! Backup exists, manual intervention needed"
        )
    print(f"APPLIED: {active}")
    print(f"Pre-leakage manifest backup: {backup}")
    print(
        "Audio and raw records preserved. Re-run leakage audit with --expected-retained <remaining_count>."
    )


if __name__ == "__main__":
    main()
