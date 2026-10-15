#!/usr/bin/env python3
"""v9_final validation online intervention using frozen selected layer and per-seed tau5."""
from __future__ import annotations

import argparse
import gc
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from src.provenance import fingerprint
from src.eval.efficiency import GENERATED_OUTPUT_TOKEN_METRIC
from src.capture.v10_prompt import intervention_prompt_provenance
from src.models.audio_reasoner_stop import stop_policy_provenance
from src.intervention.storage import load_existing

from src.gate.calibrate_tau5 import SEEDS, load_gate, load_selected_layer
from src.gate.datasets import (
    GateDatasetConfig,
    get_dataset_config,
    validate_artifact_dataset,
    validate_trajectory_rows,
)
from src.gate.diagnose_val_operating_point import load_tau5
from src.gate.select_layer import (
    MODEL_LAYERS,
    canonical_condition,
    load_sentence_states,
    read_metadata,
    validate_val_rows,
)


def first_crossing_details(states, layer_number, gate, mean, std, tau):
    if states.shape[0] == 0:
        return None
    x = states[:, layer_number - 1, :]
    x = (x - mean) / std
    with torch.no_grad():
        probs = torch.sigmoid(gate(x)).reshape(-1)
        cumulative = torch.cumsum(probs, 0) / torch.arange(
            1, probs.numel() + 1, dtype=probs.dtype
        )
        hits = torch.nonzero(cumulative.to(torch.float64) >= tau, as_tuple=False)
    if hits.numel() == 0:
        return None
    i = int(hits[0, 0])
    return {
        "sentence_index_1based": i + 1,
        "gate_probability": float(probs[i]),
        "cumulative_mean": float(cumulative[i]),
    }


def resolve_audio_path(row: dict[str, Any], metadata_path: Path) -> Path:
    keys = (
        "audio_path",
        "wav_path",
        "audio_file",
        "audio_filepath",
        "source_audio_path",
        "audio",
    )
    raw = next((row.get(k) for k in keys if row.get(k)), None)
    if isinstance(raw, dict):
        raw = raw.get("path") or raw.get("audio")
    if not raw:
        raise KeyError(
            f"{row.get('trial_id')}: no audio path field; available keys={sorted(row.keys())}"
        )
    p = Path(str(raw)).expanduser()
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p.resolve()
    candidate = (metadata_path.parent / p).resolve()
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"{row.get('trial_id')}: audio not found: {raw}")


def resolve_prompt_text(row: dict[str, Any]) -> str:
    # Deliberately do NOT fall back to row['text']; in capture metadata that may be vanilla model output.
    keys = ("prompt_text", "question", "question_text", "input_text", "prompt")
    value = next(
        (
            row.get(k)
            for k in keys
            if isinstance(row.get(k), str) and row.get(k).strip()
        ),
        None,
    )
    if value is None:
        raise KeyError(
            f"{row.get('trial_id')}: no prompt/question field; available keys={sorted(row.keys())}"
        )
    return value


def resolve_token_ids(row: dict[str, Any]) -> list[int]:
    for key in ("token_ids", "generated_token_ids", "output_token_ids"):
        value = row.get(key)
        if value is not None:
            ids = [int(x) for x in value]
            if not ids:
                raise RuntimeError(f"{row.get('trial_id')}: empty {key}")
            return ids
    raise KeyError(f"{row.get('trial_id')}: no vanilla token-id field")


def extract_final(text: str) -> str | None:
    from src.eval.response_format import inspect_response_format

    return inspect_response_format(text or "", model_name="audio-reasoner")[
        "final_text"
    ]


def is_abstention(final_text: str | None) -> bool | None:
    if final_text is None:
        return None
    t = re.sub(r"[’']", "'", final_text.lower()).strip()
    return (
        "i don't know" in t
        or "i do not know" in t
        or "cannot determine" in t
        or "can't determine" in t
    )


def runner_class(model_name: str):
    if model_name == "af_next":
        from src.intervention.af_next import AFNextInterventionRunner

        return AFNextInterventionRunner
    if model_name == "audio-reasoner":
        from src.intervention.audio_reasoner import AudioReasonerInterventionRunner

        return AudioReasonerInterventionRunner
    if model_name == "moss":
        from src.intervention.moss import MOSSInterventionRunner

        return MOSSInterventionRunner
    raise ValueError(model_name)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def verify_against_diagnostic(
    model_name: str,
    planned: dict[int, list[dict[str, Any]]],
    dataset_config: GateDatasetConfig,
) -> None:
    path = dataset_config.operating_point_root / f"{model_name}_val_tau5.json"
    if not path.exists():
        print(f"warning: diagnostic not found, skipping cross-check: {path}")
        return
    diag = json.load(path.open("r", encoding="utf-8"))
    validate_artifact_dataset(diag, dataset_config, path, model_name=model_name)
    for seed in SEEDS:
        rows = planned[seed]
        a = sum(r["triggered"] for r in rows if r["condition"] == "answerable")
        u = sum(r["triggered"] for r in rows if r["condition"] == "unanswerable")
        expected = diag["seed_results"][str(seed)]
        if a != int(expected["answerable_false_brakes"]) or u != int(
            expected["unanswerable_brakes"]
        ):
            raise RuntimeError(
                f"seed {seed}: trigger counts {a}/{u} disagree with diagnostic {expected['answerable_false_brakes']}/{expected['unanswerable_brakes']}"
            )
    print("diagnostic cross-check: PASS")


def plan_interventions(model_name: str, dataset_config: GateDatasetConfig):
    metadata_path = (
        dataset_config.trajectory_root / model_name / "val" / "metadata.jsonl"
    )
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    rows = read_metadata(metadata_path)
    n_a, n_u, n_pairs = validate_val_rows(rows)
    validate_trajectory_rows(rows, dataset_config, "val", model_name)
    expected_a, expected_u, expected_total = dataset_config.expected_counts("val")
    if (n_a, n_u, len(rows)) != (expected_a, expected_u, expected_total):
        raise RuntimeError(
            f"{dataset_config.name}: expected val A/U/total="
            f"{expected_a}/{expected_u}/{expected_total}, got {n_a}/{n_u}/{len(rows)}"
        )
    layer, _ = load_selected_layer(model_name, dataset_config)
    tau5, _ = load_tau5(model_name, layer, dataset_config)
    n_layers = MODEL_LAYERS[model_name]
    loaded = {
        seed: load_gate(model_name, layer, seed, torch.device("cpu"), dataset_config)
        for seed in SEEDS
    }
    planned = {seed: [] for seed in SEEDS}

    for i, row in enumerate(rows, 1):
        condition = canonical_condition(row["condition"])
        states = load_sentence_states(
            row=row, metadata_path=metadata_path, n_layers=n_layers
        )
        sentence_cps = [int(x) for x in row.get("sentence_checkpoints", [])]
        if len(sentence_cps) != states.shape[0]:
            raise RuntimeError(
                f"{row['trial_id']}: {len(sentence_cps)} sentence checkpoints vs {states.shape[0]} sentence states"
            )
        token_ids = resolve_token_ids(row)
        for seed in SEEDS:
            gate, mean, std = loaded[seed]
            hit = first_crossing_details(states, layer, gate, mean, std, tau5[seed])
            from src.data.frozen_umwp import v10_artifact_provenance

            base = {
                **v10_artifact_provenance(dataset_config.name, model_name),
                **intervention_prompt_provenance(),
                "generated_output_token_metric": GENERATED_OUTPUT_TOKEN_METRIC,
                **stop_policy_provenance(model_name),
                "trial_id": str(row["trial_id"]),
                "pair_id": str(row.get("pair_id") or row.get("source_id") or ""),
                "condition": condition,
                "family": row.get("family"),
                "seed": seed,
                "tau5": tau5[seed],
                "selected_layer_number": layer,
                "selected_layer_index": layer - 1,
                "vanilla_num_generated_tokens": len(token_ids),
                "vanilla_text": row.get("text"),
                "triggered": hit is not None,
                "input_fingerprint": fingerprint(
                    {
                        key: row.get(key)
                        for key in (
                            "audio_path",
                            "raw_text",
                            "tts_text",
                            "prompt_text",
                            "gold_answer",
                            "token_ids",
                            "text",
                        )
                    }
                ),
            }
            base["eligible_checkpoint_count"] = len(sentence_cps)
            base["monitoring_status"] = row.get("monitoring_status")
            base["brake_checkpoint"] = None
            if not sentence_cps:
                base["trajectory_score"] = 0.0
                base["gate_output_text"] = row["text"]
            if hit is not None:
                k = hit["sentence_index_1based"]
                cp = sentence_cps[k - 1]
                if cp <= 0 or cp > len(token_ids):
                    raise RuntimeError(
                        f"{row['trial_id']}: brake checkpoint {cp} invalid for {len(token_ids)} tokens"
                    )
                base.update(hit)
                base["brake_checkpoint"] = cp
                base["completion_fraction_at_brake"] = cp / len(token_ids)
            planned[seed].append(base)
        if i % 100 == 0 or i == len(rows):
            print(f"[{i}/{len(rows)}] intervention points planned", flush=True)

    del loaded
    gc.collect()
    verify_against_diagnostic(model_name, planned, dataset_config)
    return rows, metadata_path, layer, tau5, planned, (n_a, n_u, n_pairs)


def run(
    model_name: str,
    resume: bool,
    retry_errors: bool,
    preflight: bool,
    max_jobs: int | None,
    dataset_config: GateDatasetConfig,
) -> None:
    rows, metadata_path, layer, tau5, planned, counts = plan_interventions(
        model_name, dataset_config
    )
    row_by_id = {str(r["trial_id"]): r for r in rows}
    n_a, n_u, n_pairs = counts
    print("=" * 88)
    print("V9_FINAL VALIDATION ONLINE INTERVENTION")
    print("=" * 88)
    print(f"dataset        : {dataset_config.name}")
    print(f"model          : {model_name}")
    print(f"selected layer : {layer} (index {layer - 1})")
    print(f"val A / U      : {n_a} / {n_u} ({n_pairs} pairs)")
    print(f"seeds          : {SEEDS}")
    print(
        "policy         : first sentence boundary where cumulative mean >= frozen tau5"
    )
    print("test           : NOT read")
    for seed in SEEDS:
        a = sum(r["triggered"] for r in planned[seed] if r["condition"] == "answerable")
        u = sum(
            r["triggered"] for r in planned[seed] if r["condition"] == "unanswerable"
        )
        print(
            f"seed={seed}: tau5={tau5[seed]:.6f}, triggered A={a}, U={u}, total={a+u}"
        )
    if preflight:
        # Validate metadata fields only for trajectories that will actually invoke the model.
        for seed in SEEDS:
            for p in planned[seed]:
                if p["triggered"]:
                    row = row_by_id[p["trial_id"]]
                    resolve_audio_path(row, metadata_path)
                    resolve_prompt_text(row)
                    resolve_token_ids(row)
        print("preflight: PASS (no intervention model loaded)")
        return

    out_dir = dataset_config.intervention_root / model_name / "val"
    paths = {seed: out_dir / f"seed_{seed}.jsonl" for seed in SEEDS}
    existing, cache = {}, {}
    for seed in SEEDS:
        if paths[seed].exists() and not resume:
            raise FileExistsError(f"{paths[seed]} exists; use --resume")
        done, local_cache = load_existing(
            paths[seed],
            retry_errors=False,
            planned=planned[seed],
            dataset_config=dataset_config,
            model_name=model_name,
            split="val",
            seed=seed,
        )
        existing[seed] = done
        cache.update(local_cache)

    # Retry only after all three seed files have passed provenance checks.
    if retry_errors:
        for seed in SEEDS:
            done, _ = load_existing(
                paths[seed],
                retry_errors=True,
                planned=planned[seed],
                dataset_config=dataset_config,
                model_name=model_name,
                split="val",
                seed=seed,
            )
            existing[seed] = done

    # First write all non-triggered records; they require no model generation.
    for seed in SEEDS:
        for p in planned[seed]:
            if p["trial_id"] in existing[seed] or p["triggered"]:
                continue
            record = {
                "schema_version": 1,
                "protocol": "v9_final_validation_online_intervention",
                "dataset": dataset_config.name,
                "model_name": model_name,
                "split": "val",
                **p,
                "intervention_run": False,
                "intervention_result": None,
                "post_intervention_final": None,
                "post_intervention_abstains": None,
                "test_read": False,
            }
            append_jsonl(paths[seed], record)
            existing[seed][p["trial_id"]] = record

    jobs: dict[tuple[str, int], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for seed in SEEDS:
        for p in planned[seed]:
            if not p["triggered"] or p["trial_id"] in existing[seed]:
                continue
            jobs[(p["trial_id"], int(p["brake_checkpoint"]))].append((seed, p))

    print(f"unique triggered intervention jobs remaining: {len(jobs)}")
    if not jobs:
        print(f"complete: {out_dir}")
        return

    Runner = runner_class(model_name)
    runner = Runner()
    completed_jobs = 0
    for job_index, ((trial_id, brake_checkpoint), targets) in enumerate(
        jobs.items(), 1
    ):
        if max_jobs is not None and completed_jobs >= max_jobs:
            print(f"stopping after --max-jobs {max_jobs}")
            break
        row = row_by_id[trial_id]
        result = cache.get((trial_id, brake_checkpoint))
        error = None
        if result is None:
            try:
                result = runner.continue_from_prefix(
                    audio_path=resolve_audio_path(row, metadata_path),
                    prompt_text=resolve_prompt_text(row),
                    vanilla_token_ids=resolve_token_ids(row),
                    brake_checkpoint=brake_checkpoint,
                )
                cache[(trial_id, brake_checkpoint)] = result
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                print(f"ERROR {trial_id} cp={brake_checkpoint}: {error}", flush=True)

        for seed, p in targets:
            final_text = (
                extract_final(result.get("continuation_text", "")) if result else None
            )
            continuation_tokens = (
                int(result.get("num_continuation_tokens", 0)) if result else None
            )
            vanilla_tokens = int(p["vanilla_num_generated_tokens"])
            post_tokens = (
                brake_checkpoint + continuation_tokens
                if continuation_tokens is not None
                else None
            )
            record = {
                "schema_version": 1,
                "protocol": "v9_final_validation_online_intervention",
                "dataset": dataset_config.name,
                "model_name": model_name,
                "split": "val",
                **p,
                "intervention_run": error is None,
                "intervention_result": result,
                "intervention_error": error,
                "post_intervention_final": final_text,
                "post_intervention_abstains": is_abstention(final_text),
                "post_intervention_generated_tokens": post_tokens,
                "generated_token_reduction_vs_vanilla": (
                    (vanilla_tokens - post_tokens) if post_tokens is not None else None
                ),
                "test_read": False,
            }
            append_jsonl(paths[seed], record)
            existing[seed][trial_id] = record
        completed_jobs += 1
        if job_index % 10 == 0 or job_index == len(jobs):
            print(
                f"[{job_index}/{len(jobs)}] unique intervention jobs processed",
                flush=True,
            )

    print(f"saved: {out_dir}")
    print(
        "Next: judge/transition analysis on validation outputs; keep layer/tau5 frozen. Do NOT touch test."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="umwp", choices=("umwp", "worldsense"))
    parser.add_argument("--model", required=True, choices=tuple(MODEL_LAYERS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Plan/validate only; do not load intervention model",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=None,
        help="Optional smoke-test limit on unique triggered generations",
    )
    parser.add_argument(
        "--output", type=Path, help="Optional intervention root for a fresh run"
    )
    args = parser.parse_args()
    from dataclasses import replace

    dataset_config = get_dataset_config(args.dataset)
    if args.output is not None:
        dataset_config = replace(dataset_config, intervention_root=args.output)
    run(
        args.model,
        resume=args.resume,
        retry_errors=args.retry_errors,
        preflight=args.preflight,
        max_jobs=args.max_jobs,
        dataset_config=dataset_config,
    )


if __name__ == "__main__":
    main()
