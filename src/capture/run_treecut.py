"""TreeCut-only capture adapter; preserves UMWP prompt and native generation."""

import argparse, json
from contextlib import nullcontext
from pathlib import Path
from src.capture.run import build_model, validate_result
from src.capture.storage import (
    save_trajectory_states,
    trajectory_to_metadata,
    append_metadata,
)
from src.capture.trajectory import Trajectory
from src.capture.validation import load_completed, validate_completed
from src.config import MAX_NEW_TOKENS
from src.data.frozen_treecut import (
    ROOT,
    TASK_PROMPT,
    load_trials,
    artifact_provenance,
    sha256,
)
from src.gate.datasets import get_dataset_config, validate_trajectory_rows
from src.models.audio_delivery import validate_audio_delivery
from src.capture.treecut_failures import (
    load_failures,
    write_failure,
    capture_population,
    recoverable_generation_failure,
)


def select_trials(trials, split, limit_pairs=None):
    rows = [
        r
        for r in trials
        if r["split"] == split
        and (split != "calibration" or r["condition"] == "answerable")
    ]
    if limit_pairs is not None:
        if limit_pairs <= 0:
            raise ValueError("limit_pairs must be positive")
        ids = set(list(dict.fromkeys(r["pair_id"] for r in rows))[:limit_pairs])
        rows = [r for r in rows if r["pair_id"] in ids]
    return rows


def run(args, model_factory=build_model):
    if args.split == "test" and not args.allow_test:
        raise RuntimeError("TreeCut test requires --allow-test")
    cfg = get_dataset_config("treecut")
    is_smoke = getattr(args, "smoke_stratified", False)
    if (
        args.limit_pairs is not None or is_smoke or getattr(args, "pair_id", None)
    ) and args.output.resolve() == cfg.trajectory_root.resolve():
        raise RuntimeError("Partial smoke output must be isolated")
    rows = select_trials(load_trials(), args.split, args.limit_pairs)
    if getattr(args, "pair_id", None):
        requested = set(args.pair_id)
        if not requested <= {r["pair_id"] for r in rows}:
            raise RuntimeError("Requested pair outside selected frozen split")
        rows = [r for r in rows if r["pair_id"] in requested]
    if is_smoke:
        if args.split != "gate_train" or args.limit_pairs is not None:
            raise RuntimeError(
                "Stratified smoke requires gate_train and no limit override"
            )
        chosen = []
        for n, d in [(2, 2), (4, 2), (4, 4)]:
            candidates = [
                r
                for r in rows
                if r["condition"] == "answerable"
                and (r["numVars"], r["ansDepth"]) == (n, d)
            ]
            if not candidates:
                raise RuntimeError("Missing smoke stratum")
            chosen.append(
                (min if n == 2 else max)(candidates, key=lambda r: r["audio_duration"])[
                    "pair_id"
                ]
            )
        rows = [r for r in rows if r["pair_id"] in chosen]
    if not rows:
        raise RuntimeError("No selected TreeCut trials")
    destination = args.output / args.model / args.split
    states = destination / "states"
    metadata = destination / "metadata.jsonl"
    if metadata.exists() and not args.resume:
        raise FileExistsError("Capture exists; use --resume")
    completed = load_completed(metadata) if args.resume else {}
    ids = {r["trial_id"] for r in rows}
    if set(completed) - ids:
        raise RuntimeError("Unselected trial IDs in existing capture")
    provenance = artifact_provenance(args.model)
    failures = load_failures(destination, rows, args.model, provenance)
    for row in rows:
        audio = ROOT / row["audio_path"]
        if sha256(audio) != row["audio_sha256"]:
            raise RuntimeError("Audio fingerprint changed")
        info = validate_audio_delivery(audio, args.model)
        if info.duration >= 30:
            raise RuntimeError("TreeCut audio violates strict 30-second limit")
        if row["trial_id"] in completed:
            expected = {
                **row,
                **provenance,
                "model_name": args.model,
                "prompt_text": TASK_PROMPT,
            }
            validate_completed(
                completed[row["trial_id"]], expected, states / (row["trial_id"] + ".pt")
            )
    active_ids = {r["trial_id"] for r in rows if r["pair_id"] not in failures}
    if active_ids <= set(completed):
        _, report = capture_population(rows, destination, args.model, provenance)
        (destination / "capture_coverage.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(
            f"All usable trajectories complete; excluded pairs={len(failures)}",
            flush=True,
        )
        return
    states.mkdir(parents=True, exist_ok=True)
    model = model_factory(args.model)
    if args.model == "af_next":
        model.token_text_errors = "replace"
    snapshot = json.loads(
        (ROOT / f"docs/provenance/phase2b/{args.model}_tokenizer.json").read_text()
    )
    model.model_id = snapshot["tokenizer_path"]
    model.load()
    print(f"TreeCut {args.model} {args.split}: {len(rows)} trials", flush=True)
    for i, row in enumerate(rows, 1):
        if row["trial_id"] in completed or row["pair_id"] in failures:
            continue
        state_path = states / (row["trial_id"] + ".pt")
        if state_path.exists():
            raise FileExistsError(f"Orphan state preserved: {state_path}")
        print(f"[{i}/{len(rows)}] {row['trial_id']} {row['condition']}", flush=True)
        if getattr(args, "alignment_audit", False):
            from scripts.audits.run_v10_gpu_smoke import AlignmentAudit

            context = AlignmentAudit(model)
        else:
            context = nullcontext()
        try:
            with context as audit:
                result = model.generate_full(
                    str(ROOT / row["audio_path"]),
                    TASK_PROMPT,
                    MAX_NEW_TOKENS,
                    (),
                    capture_sentence_boundaries=True,
                )
        except Exception as exc:
            if not recoverable_generation_failure(exc) or getattr(
                args, "alignment_audit", False
            ):
                raise
            failures[row["pair_id"]] = write_failure(
                destination, row, args.model, provenance, exc, model
            )
            print(
                f"PAIR EXCLUDED {row['pair_id']}: {type(exc).__name__}: {exc}",
                flush=True,
            )
            continue
        if audit is not None:
            audit.verify(result)
        validate_result(result, model)
        if (
            result.model_name != args.model
            or result.boundary_version != provenance["boundary_version"]
        ):
            raise RuntimeError("Model/boundary mismatch")
        if (
            args.model == "audio-reasoner"
            and result.generation_stop_policy != provenance["generation_stop_policy"]
        ):
            raise RuntimeError("AR stop-policy mismatch")
        if result.tokenizer_fingerprint != snapshot["tokenizer_fingerprint"]:
            raise RuntimeError("Tokenizer fingerprint mismatch")
        trajectory = Trajectory(
            trial_id=row["trial_id"],
            pair_id=row["pair_id"],
            condition=row["condition"],
            split=row["split"],
            source=row["source"],
            category=None,
            voice=row["voice"],
            audio_path=row["audio_path"],
            raw_text=row["raw_text"],
            tts_text=row["tts_text"],
            prompt_text=TASK_PROMPT,
            gold_answer=tuple(row["gold_answer"]),
            generation=result,
        )
        saved = {**row, **trajectory_to_metadata(trajectory, state_path), **provenance}
        decoder = getattr(model, "last_token_text_decoder", None)
        if decoder is not None:
            saved["token_text_decode_policy"] = (
                "native-replace-v1" if decoder.utf8.errors == "replace" else "strict-v1"
            )
            saved["malformed_utf8_events"] = decoder.replacement_events
            if decoder.replacement_events:
                print(
                    "UTF8 REPLACEMENTS " + json.dumps(decoder.replacement_events),
                    flush=True,
                )
        validate_trajectory_rows(
            [saved], cfg, args.split, args.model, require_complete=False
        )
        save_trajectory_states(trajectory, state_path)
        append_metadata(saved, metadata)
        print(
            f"  tokens={result.num_generated_tokens} checkpoints={len(result.sentence_checkpoints)} stop={result.stop_reason}",
            flush=True,
        )
    active, report = capture_population(rows, destination, args.model, provenance)
    (destination / "capture_coverage.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    usable = [
        r for r in load_completed(metadata).values() if r["pair_id"] not in failures
    ]
    validate_trajectory_rows(
        usable,
        cfg,
        args.split,
        args.model,
        require_complete=args.limit_pairs is None
        and not is_smoke
        and not getattr(args, "pair_id", None),
    )
    if {r["trial_id"] for r in usable} != {r["trial_id"] for r in active}:
        raise RuntimeError("Unaccounted-for TreeCut capture trials")
    print("CAPTURE COVERAGE " + json.dumps(report), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model", required=True, choices=["af_next", "moss", "audio-reasoner"]
    )
    p.add_argument(
        "--split", required=True, choices=["gate_train", "calibration", "val", "test"]
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--allow-test", action="store_true")
    p.add_argument("--limit-pairs", type=int)
    p.add_argument("--alignment-audit", action="store_true")
    p.add_argument(
        "--pair-id",
        action="append",
        help="Isolated diagnostic pair from the requested frozen split",
    )
    p.add_argument(
        "--smoke-stratified",
        action="store_true",
        help="Isolated three-pair shortest/longest audio and structural smoke",
    )
    p.add_argument("--output", type=Path, default=Path("data/treecut/trajectories"))
    run(p.parse_args())


if __name__ == "__main__":
    main()
