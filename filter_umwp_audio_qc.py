#!/usr/bin/env python3
"""Pair-level UMWP audio-length QC. Keep original audio and data intact.

Usage (from project root):
    python filter_umwp_audio_qc.py
    python filter_umwp_audio_qc.py --apply

The second command atomically replaces the active split JSON after backing it up.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import soundfile as sf

SPLITS = ("gate_train", "calibration", "val", "test")
WAV_RE = re.compile(r"^(umwp_\d+)_(A|U)$")
PAIR_RE = re.compile(r"^(umwp_\d+)(?:_[AU])?$")


def die(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def atomic_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
        json.dump(data, tmp, ensure_ascii=False, indent=2)
        tmp.write("\n")
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(tmp_path, path)


def pair_id_from_entry(entry: object) -> str:
    if isinstance(entry, int) and not isinstance(entry, bool):
        return f"umwp_{entry}"
    if isinstance(entry, str):
        entry = entry.strip()
        if entry.isdigit():
            return f"umwp_{int(entry)}"
        match = PAIR_RE.fullmatch(entry)
        if match:
            return match.group(1)
    if isinstance(entry, dict):
        for key in ("pair_id", "source_id", "trial_id", "id"):
            if key in entry:
                return pair_id_from_entry(entry[key])
    die(
        f"Unrecognized split entry: {str(entry)[:250]}; check manifest schema before proceeding"
    )


def split_lists(root: object) -> dict[str, list]:
    if not isinstance(root, dict):
        die("Split JSON must be an object")
    container = root.get("splits", root)
    if not isinstance(container, dict) or any(k not in container for k in SPLITS):
        die(
            f"Expected keys {SPLITS} at top level or under 'splits'; found {list(container) if isinstance(container,dict) else type(container).__name__}"
        )
    for split in SPLITS:
        if not isinstance(container[split], list):
            die(
                f"Split {split} is not a list; aborting rather than guessing its schema"
            )
    # Do not preserve stale cached counts if they are present in unexpected metadata.
    for obj in (root, container):
        for key in obj:
            if (
                key not in SPLITS
                and key != "splits"
                and re.search(
                    r"count|n_pairs|n_trials|num_pairs|num_trials|total",
                    key,
                    flags=re.I,
                )
            ):
                die(
                    f"Manifest contains metadata field '{key}' that might become stale after filtering"
                )
    return container


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("."), help="V10 repository root")
    ap.add_argument("--max-seconds", type=float, default=29.0)
    ap.add_argument("--expect-pairs", type=int, default=2600)
    ap.add_argument("--expect-excluded", type=int, default=34)
    ap.add_argument(
        "--apply", action="store_true", help="Back up and update ACTIVE split manifest"
    )
    args = ap.parse_args()
    if args.max_seconds <= 0:
        die("max-seconds must be positive")
    root = args.root.resolve()
    audio_root = root / "data/tts/umwp"
    split_path = root / "data/splits/umwp_final_split.json"
    qc_dir = root / "data/qc"
    filtered_path = root / "data/splits/umwp_final_split_qc29.json"
    if not audio_root.is_dir() or not split_path.is_file():
        die(f"Expected {audio_root} and {split_path} to exist")
    if args.max_seconds != 29.0:
        die(
            "This V10 QC output is deliberately fixed at 29.0s; do not change the threshold without reviewing output names"
        )

    wavs = sorted(audio_root.rglob("*.wav"))
    pairs = defaultdict(dict)
    for wav in wavs:
        split = wav.parent.name
        match = WAV_RE.fullmatch(wav.stem)
        if split not in SPLITS or match is None:
            die(f"Unexpected WAV path: {wav}")
        pid, label = match.groups()
        info = sf.info(str(wav))
        if info.samplerate <= 0 or info.frames <= 0:
            die(f"Invalid audio header: {wav}")
        key = (split, pid)
        if label in pairs[key]:
            die(f"Duplicate WAV: {wav}")
        pairs[key][label] = {
            "duration": info.frames / info.samplerate,
            "frames": info.frames,
            "rate": info.samplerate,
        }

    if len(wavs) != args.expect_pairs * 2 or len(pairs) != args.expect_pairs:
        die(
            f"Incomplete/unexpected inventory: {len(wavs)} WAVs, {len(pairs)} pairs; expected {args.expect_pairs * 2} and {args.expect_pairs}"
        )
    for key, record in pairs.items():
        if set(record) != {"A", "U"}:
            die(f"Incomplete pair {key}: {list(record)}")

    # No pair ID may appear under more than one split directory.
    pid_split = {}
    for split, pid in pairs:
        if pid in pid_split:
            die(f"Cross-split duplicated pair ID: {pid} in {pid_split[pid]}, {split}")
        pid_split[pid] = split

    rows = []
    excluded = set()
    for (split, pid), ab in sorted(pairs.items()):
        over = any(
            ab[label]["frames"] > args.max_seconds * ab[label]["rate"]
            for label in ("A", "U")
        )
        if over:
            excluded.add(pid)
        rows.append(
            {
                "split": split,
                "pair_id": pid,
                "A_seconds": round(ab["A"]["duration"], 6),
                "U_seconds": round(ab["U"]["duration"], 6),
                "max_seconds": round(max(ab["A"]["duration"], ab["U"]["duration"]), 6),
                "eligible": not over,
                "reason": "over_29s_either_A_or_U" if over else "",
            }
        )
    if len(excluded) != args.expect_excluded:
        die(
            f"Excluded {len(excluded)} pairs, expected {args.expect_excluded}; stopping to avoid unintended manifest changes"
        )

    manifest = json.loads(split_path.read_text(encoding="utf-8"))
    container = split_lists(manifest)
    counts = {}
    for split in SPLITS:
        entries = container[split]
        ids = [pair_id_from_entry(x) for x in entries]
        ids_set = set(ids)
        disk_set = {pid for s, pid in pairs if s == split}
        keep_set = disk_set - excluded
        if ids_set not in (disk_set, keep_set):
            die(
                f"{split}: manifest pair IDs ({len(ids_set)}) disagree with both full disk inventory ({len(disk_set)}) and QC subset ({len(keep_set)}). Example missing: {sorted(disk_set-ids_set)[:3]}; extra: {sorted(ids_set-disk_set)[:3]}"
            )
        if len(ids) not in (len(ids_set), len(ids_set) * 2):
            die(f"{split}: unexpected multiplicity of A/U vs pair entries")
        if len(ids) == len(ids_set) * 2:
            trial_ids = [
                (
                    str(x.get("trial_id", x.get("id", "")))
                    if isinstance(x, dict)
                    else str(x)
                )
                for x in entries
            ]
            for pid in ids_set:
                pid_entries = [
                    x for x, entry_id in zip(trial_ids, ids) if entry_id == pid
                ]
                if not (
                    len(pid_entries) == 2
                    and any(v.endswith("_A") for v in pid_entries)
                    and any(v.endswith("_U") for v in pid_entries)
                ):
                    die(
                        f"{split}: trial-level entries for {pid} must represent one A and one U"
                    )
        container[split] = [item for item, pid in zip(entries, ids) if pid in keep_set]
        counts[split] = {
            "original_pairs": len(disk_set),
            "excluded_pairs": len(disk_set - keep_set),
            "retained_pairs": len(keep_set),
        }

    if (
        sum(c["retained_pairs"] for c in counts.values())
        != args.expect_pairs - args.expect_excluded
    ):
        die("Unexpected retained count")

    qc_dir.mkdir(parents=True, exist_ok=True)
    csv_path = qc_dir / "umwp_audio_qc29_pairs.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(
            out,
            fieldnames=[
                "split",
                "pair_id",
                "A_seconds",
                "U_seconds",
                "max_seconds",
                "eligible",
                "reason",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    summary_path = qc_dir / "umwp_audio_qc29_summary.json"
    atomic_json(
        summary_path,
        {
            "max_seconds": args.max_seconds,
            "criterion": "Exclude whole pair if either WAV has more than 29 seconds of audio",
            "total_pairs": len(pairs),
            "excluded_pairs": len(excluded),
            "retained_pairs": len(pairs) - len(excluded),
            "counts_by_split": counts,
            "excluded_pair_ids": sorted(excluded),
            "original_audio_retained": True,
        },
    )
    atomic_json(filtered_path, manifest)

    print(f"WAVs: {len(wavs)}, pairs: {len(pairs)}")
    print(
        f"Excluded (>29s in either A/U): {len(excluded)}; retained: {len(pairs)-len(excluded)}"
    )
    for split in SPLITS:
        c = counts[split]
        print(
            f"{split:12s}: {c['original_pairs']:4d} -> {c['retained_pairs']:4d} (excluded {c['excluded_pairs']})"
        )
    print(f"QC CSV: {csv_path}")
    print(f"QC summary: {summary_path}")
    print(f"Filtered manifest: {filtered_path}")

    if args.apply:
        backup = split_path.with_name("umwp_final_split.before_qc29.json")
        if not backup.exists():
            shutil.copy2(split_path, backup)
        else:
            prior = json.loads(backup.read_text(encoding="utf-8"))
            prior_lists = split_lists(prior)
            for split in SPLITS:
                original_disk_ids = {pid for s, pid in pairs if s == split}
                if {
                    pair_id_from_entry(x) for x in prior_lists[split]
                } != original_disk_ids:
                    die(
                        f"Existing backup at {backup} is not the original full manifest. No overwrite performed"
                    )
        atomic_json(split_path, manifest)
        print(f"ACTIVE manifest updated: {split_path}")
        print(f"ORIGINAL manifest preserved: {backup}")
    else:
        print(
            "DRY RUN only: active manifest unchanged. Run again with --apply to activate QC filtering."
        )


if __name__ == "__main__":
    main()
