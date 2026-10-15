#!/usr/bin/env python3
"""Read-only cross-split duplicate-question audit for UMWP V10.

Run from the V10 project root:
    python audit_umwp_split_leakage.py

This checks the QC-filtered ACTIVE split against original UMWP raw records.
It NEVER rewrites split manifests, audio, source records, or labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

SPLITS = ("gate_train", "calibration", "val", "test")
PID_PATTERN = re.compile(r"^umwp_(\d+)(?:_[AU])?$")


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def canonical_id(value: object) -> str:
    if isinstance(value, dict):
        for key in ("pair_id", "source_id", "trial_id", "id"):
            if key in value:
                return canonical_id(value[key])
        fail(f"Split entry has no recognizable id: {value!r}")
    if isinstance(value, bool):
        fail("Boolean split entry is invalid")
    if isinstance(value, int) or (isinstance(value, str) and value.strip().isdigit()):
        return f"umwp_{int(value)}"
    if isinstance(value, str):
        match = PID_PATTERN.fullmatch(value.strip())
        if match:
            return f"umwp_{int(match.group(1))}"
    fail(f"Unrecognized split entry: {value!r}")


def load_split(path: Path) -> dict[str, set[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    group = data.get("splits", data)
    if not isinstance(group, dict) or any(s not in group for s in SPLITS):
        fail(f"Expected split lists {SPLITS} in {path}")
    mapping: dict[str, set[str]] = {}
    for split in SPLITS:
        if not isinstance(group[split], list):
            fail(f"Invalid split list: {split}")
        ids = [canonical_id(e) for e in group[split]]
        counts = Counter(ids)
        if any(v not in (1, 2) for v in counts.values()):
            fail(f"{split}: invalid trial/pair multiplicity")
        if len(set(counts.values())) > 1:
            fail(f"{split}: mixed pair-level and trial-level IDs")
        mapping[split] = set(ids)
    locations = defaultdict(list)
    for split, ids in mapping.items():
        for pid in ids:
            locations[pid].append(split)
    overlaps = {pid: s for pid, s in locations.items() if len(s) > 1}
    if overlaps:
        fail(f"Pair IDs cross splits: {list(overlaps.items())[:5]}")
    return mapping


def normalize_question(text: str) -> str:
    """Conservative exact-ish key: retain digits, operators, punctuation and words."""
    t = unicodedata.normalize("NFKC", text).casefold()
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    return t


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def get_text(record: dict, row_no: int) -> str:
    for key in ("question", "Question", "problem", "Problem", "query", "Q"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    fail(f"Raw record {row_no} lacks a nonempty question. Keys: {list(record)[:30]}")


def get_id(record: dict, row_no: int) -> str:
    for key in ("id", "ID"):
        if key in record:
            value = record[key]
            if isinstance(value, bool):
                break
            if isinstance(value, int) or (
                isinstance(value, str) and value.strip().isdigit()
            ):
                return str(int(value))
    fail(f"Raw record {row_no} lacks a numeric ID. Keys: {list(record)[:30]}")


def load_original_pairs(path: Path, expected_pairs: int) -> dict[str, dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                fail(f"Invalid record {line_no}: not a JSON object")
            records.append((line_no, obj))
    if len(records) != 2 * expected_pairs:
        fail(f"Unexpected raw data size: {len(records)} (expected {2*expected_pairs})")

    # The verified UMWP source stores A records first and U records in the second half.
    first, last = records[:expected_pairs], records[expected_pairs:]
    a_by_id = {}
    for no, obj in first:
        aid = get_id(obj, no)
        if aid in a_by_id:
            fail(f"Repeated A id: {aid}")
        a_by_id[aid] = obj
    out = {}
    for no, u in last:
        refs = u.get("relevant_ids")
        if not isinstance(refs, list) or len(refs) != 1:
            fail(
                f"U record {no}: expected exactly one relevant_ids reference; found {refs!r}"
            )
        aid = refs[0]
        if not isinstance(aid, (str, int)) or not str(aid).strip().isdigit():
            fail(f"U record {no}: invalid A reference: {aid!r}")
        aid = str(int(aid))
        if aid not in a_by_id:
            fail(f"U record {no} refers to missing A id {aid}")
        pid = f"umwp_{aid}"
        if pid in out:
            fail(f"Duplicated reference/pair id {pid}")
        a = a_by_id[aid]
        out[pid] = {
            "A": get_text(a, no),
            "U": get_text(u, no),
            "A_raw_id": aid,
            "U_raw_id": get_id(u, no),
            "source": str(a.get("source", a.get("dataset", ""))),
        }
    if len(out) != expected_pairs:
        fail(f"Missing source pairs: {len(out)} != {expected_pairs}")
    return out


def summarize_groups(
    records: list[dict], field: str, cross_only: bool
) -> tuple[list[dict], list[dict]]:
    buckets = defaultdict(list)
    for rec in records:
        buckets[rec[field]].append(rec)
    groups = []
    cross = []
    for key, items in buckets.items():
        pair_ids = {x["pair_id"] for x in items}
        if len(pair_ids) < 2:
            continue
        splits = sorted({x["split"] for x in items})
        group = {
            "question_hash": hash_key(key),
            "example_question": items[0]["question"],
            "splits": splits,
            "pair_count": len(pair_ids),
            "members": [
                {
                    "pair_id": x["pair_id"],
                    "split": x["split"],
                    "label": x["label"],
                    "source": x["source"],
                }
                for x in sorted(
                    items, key=lambda x: (x["split"], x["pair_id"], x["label"])
                )
            ],
        }
        groups.append(group)
        if len(splits) > 1:
            cross.append(group)
    groups.sort(key=lambda x: (-len(x["splits"]), -x["pair_count"], x["question_hash"]))
    cross.sort(key=lambda x: (-len(x["splits"]), -x["pair_count"], x["question_hash"]))
    return groups, cross


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--split", default="data/splits/umwp_final_split.json")
    p.add_argument("--raw", default="data/raw/umwp/StandardDataset.jsonl")
    p.add_argument("--output", default="data/qc/umwp_cross_split_duplicates.json")
    p.add_argument("--expected-retained", type=int, default=2566)
    args = p.parse_args()
    root = args.root.resolve()
    split_path = root / args.split
    raw_path = root / args.raw
    if not split_path.is_file() or not raw_path.is_file():
        fail(f"Missing input(s): {split_path} or {raw_path}")
    splits = load_split(split_path)
    retained = sum(len(v) for v in splits.values())
    if retained != args.expected_retained:
        fail(
            f"Retained pairs {retained}, expected {args.expected_retained}. Check split / QC first."
        )
    source = load_original_pairs(raw_path, expected_pairs=2600)
    selected = set().union(*splits.values())
    missing = selected - set(source)
    if missing:
        fail(
            f"Split IDs do not match original A IDs: {sorted(missing)[:8]}; aborting instead of guessing mapping"
        )
    if len(selected) != retained:
        fail("Duplicate pair IDs in retained split population")

    records = []
    same_pair = []
    for split in SPLITS:
        for pid in sorted(splits[split]):
            entry = source[pid]
            if normalize_question(entry["A"]) == normalize_question(entry["U"]):
                same_pair.append(
                    {"pair_id": pid, "split": split, "A": entry["A"], "U": entry["U"]}
                )
            for label in ("A", "U"):
                records.append(
                    {
                        "pair_id": pid,
                        "split": split,
                        "label": label,
                        "source": entry["source"],
                        "question": entry[label],
                        "normalized": normalize_question(entry[label]),
                    }
                )

    # Primary: same A question in different split => possible evaluation leakage.
    a_records = [x for x in records if x["label"] == "A"]
    a_all, a_cross = summarize_groups(a_records, "normalized", cross_only=True)
    all_groups, all_cross = summarize_groups(records, "normalized", cross_only=True)
    u_records = [x for x in records if x["label"] == "U"]
    u_all, u_cross = summarize_groups(u_records, "normalized", cross_only=True)
    report = {
        "audit_type": "read_only_conservative_normalized_question_duplicates",
        "normalization": "NFKC + casefold + whitespace collapse + remove spaces before punctuation; keep all digits/operators/punctuation",
        "split_file": str(split_path.relative_to(root)),
        "raw_file": str(raw_path.relative_to(root)),
        "retained_pair_counts": {s: len(ids) for s, ids in splits.items()},
        "retained_pairs": retained,
        "trial_texts_analyzed": len(records),
        "same_pair_identical_A_and_U_count": len(same_pair),
        "same_pair_identical_A_and_U": same_pair,
        "duplicate_A_question_groups_all_splits": a_all,
        "cross_split_duplicate_A_question_groups": a_cross,
        "cross_split_duplicate_U_question_groups": u_cross,
        "cross_split_duplicate_any_label_groups": all_cross,
        "within_or_cross_duplicate_U_question_groups": u_all,
        "within_or_cross_duplicate_any_label_groups": all_groups,
        "warnings": [
            "Exact normalized matches only; semantic paraphrases are not detected.",
            "No split assignments or source/audio files were changed.",
            "Cross-split duplicates require review before a frozen test evaluation.",
        ],
    }
    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("UMWP cross-split leakage audit (READ ONLY)")
    print("Retained pairs:", retained, "/ 2600")
    print("By split:", ", ".join(f"{s}={len(splits[s])}" for s in SPLITS))
    print("Identical A/U texts in same pair:", len(same_pair))
    print("Cross-split identical A-question groups:", len(a_cross))
    print("Cross-split identical U-question groups:", len(u_cross))
    print("Cross-split identical text, any label:", len(all_cross))
    print("Within-split or cross-split duplicate A-question groups:", len(a_all))
    print("Saved:", out)
    if all_cross or same_pair:
        print("\nREVIEW REQUIRED — first matching groups:")
        for group in all_cross[:8]:
            members = ", ".join(
                f"{m['pair_id']}_{m['label']}@{m['split']}" for m in group["members"]
            )
            print(f"  {members} | {group['example_question'][:120]!r}")
        if same_pair:
            print(
                "Within-pair identical A/U:",
                ", ".join(x["pair_id"] for x in same_pair[:15]),
            )
        print(
            "No manifest edits made. Do not launch capture until duplicates have been reviewed."
        )
    else:
        print(
            "No conservative exact-text cross-split duplicates detected; near duplicates still need separate review."
        )


if __name__ == "__main__":
    main()
