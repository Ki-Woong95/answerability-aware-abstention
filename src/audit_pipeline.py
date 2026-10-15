"""Read-only audit of current UMWP/WorldSense inputs and saved pipeline outputs.

Run: python -m src.audit_pipeline --dataset umwp --report logs/audits/pipeline_audit.json
UMWP is the default; WorldSense is inspected only when explicitly selected.
Add --states to read and check every current trajectory tensor (CPU only).
Archived forced-choice experiments are deliberately outside the current protocol.
"""

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf
import torch

from src.capture.run import TASK_PROMPT as UMWP_PROMPT, load_split_map, read_jsonl
from src.capture.run_umwp_always_idk import TASK_PROMPT as UMWP_IDK_PROMPT
from src.capture.storage import metadata_to_trajectory
from src.capture.umwp_validation import validate_manifest
from src.gate.datasets import get_dataset_config, load_split_ids
from src.gate.select_layer import MODEL_LAYERS
from src.models.base import COMPLETE_TOKEN_CAPTURE


def issue(block, name, example):
    entry = block.setdefault("issues", {}).setdefault(
        name, {"count": 0, "examples": []}
    )
    entry["count"] += 1
    if len(entry["examples"]) < 10:
        entry["examples"].append(example)


def expected_inputs(dataset):
    config = get_dataset_config(dataset)
    split_sets = {
        split: load_split_ids(config, split)
        for split in ("gate_train", "calibration", "val", "test")
    }
    pairs = read_jsonl(
        Path(
            f"data/processed/{dataset}_pairs"
            + ("_openended" if dataset == "worldsense" else "")
            + ".jsonl"
        )
    )
    manifest = read_jsonl(Path(f"data/tts/{dataset}_manifest.jsonl"))
    if dataset == "umwp":
        split_map = load_split_map(config.split_path)
        manifest = [row for row in manifest if str(row["pair_id"]) in split_map]
        validate_manifest(manifest, pairs, split_map)
        pair_map = {p["pair_id"]: p for p in pairs}
        expected = {}
        for row in manifest:
            row = dict(row)
            row["gold_answer"] = (
                pair_map[row["pair_id"]]["answerable"]["answer"]
                if row["condition"] == "answerable"
                else []
            )
            expected[row["trial_id"]] = row
    else:
        from src.capture.worldsense_validation import validate_trial_audio

        tts = {row["source_id"]: row for row in manifest}
        trials = read_jsonl(Path("data/processed/worldsense_trials_openended.jsonl"))
        expected = {}
        for row in trials:
            validate_trial_audio(row, tts[row["source_id"]])
            expected[row["trial_id"]] = dict(
                trial_id=row["trial_id"],
                pair_id=row["source_id"],
                condition=row["condition"],
                split=row["split"],
                source="worldsense",
                category=None,
                voice=tts[row["source_id"]]["voice"],
                audio_path=row["audio_path"],
                raw_text=row["audio_text"],
                tts_text=row["audio_text"],
                gold_answer=(
                    [str(row["gold_answer"])]
                    if row["condition"] == "answerable"
                    else []
                ),
                question=row["question"],
            )
    retained_ids = set().union(*split_sets.values())
    expected_split_trials = {s: config.expected_counts(s)[2] for s in split_sets}
    block = {
        "pairs": len(retained_ids),
        "original_pairs": len(pairs),
        "trials": len(expected),
        "expected_capture_trials": sum(expected_split_trials.values()),
        "expected_split_trials": expected_split_trials,
        "split_pairs": {s: len(ids) for s, ids in split_sets.items()},
    }
    for split, ids in split_sets.items():
        for other, other_ids in split_sets.items():
            if split < other and ids & other_ids:
                issue(
                    block,
                    "split_overlap",
                    {"splits": [split, other], "ids": sorted(ids & other_ids)[:10]},
                )
    groups = defaultdict(list)
    audio_paths = {}
    for row in expected.values():
        if row["pair_id"] not in split_sets[row["split"]]:
            issue(block, "wrong_split", row["trial_id"])
        text = " ".join(row["raw_text"].lower().split())
        groups[text].append(
            {
                "pair_id": row["pair_id"],
                "condition": row["condition"],
                "split": row["split"],
            }
        )
        audio_paths[row["audio_path"]] = row
    for entries in groups.values():
        if len({entry["split"] for entry in entries}) > 1:
            # WorldSense A/U share one context, so collapse repeated pair entries.
            issue(
                block,
                "same_text_across_splits",
                list({e["pair_id"]: e for e in entries}.values()),
            )
    for audio_path, row in audio_paths.items():
        try:
            info = sf.info(audio_path)
            if info.samplerate != 24000 or info.channels != 1 or info.frames <= 0:
                raise ValueError(f"invalid WAV header: {info}")
        except Exception as exc:
            issue(block, "audio_file", {"path": audio_path, "error": str(exc)})
    block["audio_files_checked"] = len(audio_paths)
    return expected, block


def read_metadata_snapshot(path):
    """Read complete lines once without disturbing an actively appended file."""
    lines = path.read_bytes().splitlines(keepends=True)
    incomplete = bool(lines and not lines[-1].endswith(b"\n"))
    if incomplete:
        lines.pop()
    return [json.loads(line) for line in lines if line.strip()], incomplete


def validate_audited_state(row):
    if not row.get("checkpoints"):
        if any(
            row.get(name) != []
            for name in (
                "checkpoints",
                "fixed_checkpoints",
                "sentence_checkpoints",
                "boundary_events",
            )
        ):
            raise ValueError(
                "zero-checkpoint metadata requires explicit empty checkpoint/event lists"
            )
        if (
            row.get("eligible_checkpoint_count") != 0
            or row.get("monitoring_status") != "no_eligible_reasoning_checkpoint"
        ):
            raise ValueError("missing zero-checkpoint completion metadata")
    # The loader verifies rank, finite values, list/tensor agreement, norm shape,
    # metadata dimensions, and the dimensioned empty payload before construction.
    trajectory = metadata_to_trajectory(row)
    result = trajectory.generation
    dimensions = (row["num_layers"], row["hidden_dim"])
    if (
        dimensions != (result.num_layers, result.hidden_size)
        or dimensions[0] != MODEL_LAYERS[row["model_name"]]
        or dimensions[1] <= 0
    ):
        raise ValueError("state dimensions do not match metadata/model")
    if trajectory.checkpoints:
        if any(
            tuple(state.shape) != dimensions for state in result.all_layers.values()
        ):
            raise ValueError("state dimensions do not match metadata/model")
    return trajectory


def audit_capture(path, dataset, expected, prompt, check_states):
    initial_size = path.stat().st_size
    rows, incomplete = read_metadata_snapshot(path)
    block = {
        "dataset": dataset,
        "rows": len(rows),
        "conditions": dict(Counter(r["condition"] for r in rows)),
        "legacy_ar_rows": 0,
        "legacy_ar_2047_eos": 0,
        "state_files_checked": 0,
        "zero_checkpoint_state_files_checked": 0,
        "incomplete_trailing_line": incomplete,
    }
    seen = set()
    expected_split = path.parent.name
    for index, row in enumerate(rows, 1):
        trial_id = row["trial_id"]
        if trial_id in seen:
            issue(block, "duplicate_trial", trial_id)
        seen.add(trial_id)
        current = expected.get(trial_id)
        if current is None:
            issue(block, "unknown_trial", trial_id)
        else:
            fields = (
                "pair_id",
                "condition",
                "split",
                "source",
                "category",
                "voice",
                "audio_path",
                "raw_text",
                "tts_text",
                "gold_answer",
            )
            for field in fields:
                if row.get(field) != current[field]:
                    issue(block, "stale_" + field, trial_id)
            current_prompt = prompt.format(question=current.get("question", ""))
            if row.get("prompt_text") != current_prompt:
                issue(block, "stale_prompt_text", trial_id)
        if (
            row.get("model_name") != path.parent.parent.name
            or row.get("split") != expected_split
        ):
            issue(block, "model_or_directory_mismatch", trial_id)
        if len(row.get("token_ids", [])) != row.get("num_generated_tokens"):
            issue(block, "token_count", trial_id)
        if (
            row.get("model_name") == "audio-reasoner"
            and row.get("token_capture_method") != COMPLETE_TOKEN_CAPTURE
        ):
            block["legacy_ar_rows"] += 1
            block["legacy_ar_2047_eos"] += (
                row.get("num_generated_tokens") == 2047
                and row.get("stop_reason") == "eos"
            )
        state_path = Path(row["state_path"])
        if not state_path.exists():
            issue(block, "missing_state", trial_id)
        elif check_states:
            try:
                trajectory = validate_audited_state(row)
                block["state_files_checked"] += 1
                block[
                    "zero_checkpoint_state_files_checked"
                ] += not trajectory.checkpoints
            except Exception as exc:
                issue(block, "invalid_state", {"trial_id": trial_id, "error": str(exc)})
        if index % 500 == 0:
            print(f"{path}: {index}/{len(rows)} checked", flush=True)
    expected_ids = {
        tid
        for tid, row in expected.items()
        if row["split"] == expected_split
        and (expected_split != "calibration" or row["condition"] == "answerable")
    }
    block["expected_trials"] = len(expected_ids)
    block["missing_trials"] = len(expected_ids - seen)
    block["metadata_grew_during_scan"] = path.stat().st_size != initial_size
    return block


def audit_interventions(path, meta_by_id):
    rows = read_jsonl(path)
    block = {"rows": len(rows), "triggered": 0, "legacy_retokenized_ar": 0}
    seen = set()
    for row in rows:
        tid = row["trial_id"]
        if tid in seen:
            issue(block, "duplicate_trial", tid)
        seen.add(tid)
        meta = meta_by_id.get(tid)
        if not meta:
            issue(block, "unknown_trial", tid)
            continue
        if row.get("vanilla_text") != meta.get("text") or row.get(
            "vanilla_num_generated_tokens"
        ) != meta.get("num_generated_tokens"):
            issue(block, "stale_vanilla", tid)
        if row.get("triggered"):
            block["triggered"] += 1
            result = row.get("intervention_result")
            if row.get("intervention_error") or not result:
                issue(block, "generation_error", tid)
                continue
            cp = row["brake_checkpoint"]
            if result["prefix_token_ids"] != meta["token_ids"][:cp]:
                issue(block, "wrong_prefix", tid)
            ids = result["continuation_token_ids"]
            if len(ids) != result["num_continuation_tokens"]:
                issue(block, "continuation_count", tid)
            if (
                row.get("model_name") == "audio-reasoner"
                and result.get("token_capture_method") != COMPLETE_TOKEN_CAPTURE
            ):
                block["legacy_retokenized_ar"] += 1
                engine_n = result.get("engine_generated_tokens")
                if engine_n is not None and engine_n != len(ids):
                    issue(
                        block,
                        "engine_vs_retokenized_count",
                        {"trial_id": tid, "engine": engine_n, "stored": len(ids)},
                    )
    return block


def run(report_path, check_states, dataset="umwp"):
    if dataset not in ("umwp", "worldsense"):
        raise ValueError(f"Unsupported audit dataset: {dataset}")
    torch.set_num_threads(1)
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "states_checked": check_states,
        "inputs": {},
        "captures": {},
        "gates": {},
        "interventions": {},
        "judgments": {},
    }

    def save():
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    for dataset in (dataset,):
        expected, report["inputs"][dataset] = expected_inputs(dataset)
        config = get_dataset_config(dataset)
        if dataset == "umwp":
            prompts = (UMWP_PROMPT, UMWP_IDK_PROMPT)
        else:
            from src.capture.run_worldsense import TASK_PROMPT as WS_PROMPT
            from src.capture.run_worldsense_always_idk import (
                TASK_PROMPT as WS_IDK_PROMPT,
            )

            prompts = (WS_PROMPT, WS_IDK_PROMPT)
        roots = (
            (config.trajectory_root, prompts[0]),
            (Path(f"data/ablations/always_idk/{dataset}/trajectories"), prompts[1]),
        )
        for root, prompt in roots:
            for path in sorted(root.glob("*/*/metadata.jsonl")):
                print(f"Auditing {path}", flush=True)
                report["captures"][str(path)] = audit_capture(
                    path, dataset, expected, prompt, check_states
                )
                save()
        for path in sorted(config.gate_root.glob("*/all_layers/train_summary.json")):
            summary = json.loads(path.read_text())
            report["gates"][str(path)] = {
                "captured_pairs": summary["n_pairs"],
                "expected_pairs": len(load_split_ids(config, "gate_train")),
            }
        for path in sorted(config.intervention_root.glob("*/*/seed_*.jsonl")):
            model, split = path.parent.parent.name, path.parent.name
            meta_path = config.trajectory_root / model / split / "metadata.jsonl"
            meta_by_id = {r["trial_id"]: r for r in read_jsonl(meta_path)}
            report["interventions"][str(path)] = audit_interventions(path, meta_by_id)
        for root in (
            config.judge_root,
            Path(f"data/ablations/always_idk/{dataset}/judge"),
        ):
            for path in sorted(root.glob("*_judged_*.jsonl")):
                rows = read_jsonl(path)
                block = {"rows": len(rows), "without_input_fingerprint": 0}
                seen = set()
                for row in rows:
                    key = (row["model"], row.get("seed"), row["trial_id"])
                    if key in seen:
                        issue(block, "duplicate_judgment", list(key))
                    seen.add(key)
                    block["without_input_fingerprint"] += not bool(
                        row.get("input_fingerprint")
                    )
                report["judgments"][str(path)] = block
        save()
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save()
    print(f"Saved audit: {report_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path, default=Path("logs/audits/pipeline_audit.json")
    )
    parser.add_argument("--states", action="store_true")
    parser.add_argument("--dataset", choices=("umwp", "worldsense"), default="umwp")
    args = parser.parse_args()
    run(args.report, args.states, args.dataset)
