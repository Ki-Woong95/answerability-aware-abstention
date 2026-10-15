#!/usr/bin/env python3
"""Check complete capture, gates and intervention; defaults preserve the AR queue."""

import argparse
import importlib
import json
from pathlib import Path
import torch

from src.capture.storage import metadata_to_trajectory
from src.gate.datasets import (
    get_dataset_config,
    load_split_ids,
    validate_artifact_dataset,
    validate_trajectory_rows,
)
from src.intervention.storage import load_existing
from src.models.base import COMPLETE_TOKEN_CAPTURE


MODEL = "audio-reasoner"


def read_rows(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def check_capture(dataset, split, always_idk=False, model=MODEL):
    config = get_dataset_config(dataset)
    root = (
        Path(f"data/ablations/always_idk/{dataset}/trajectories")
        if always_idk
        else config.trajectory_root
    )
    path = root / model / split / "metadata.jsonl"
    rows = read_rows(path)
    validate_trajectory_rows(rows, config, split, model)
    for index, row in enumerate(rows, 1):
        if model == MODEL and row.get("token_capture_method") != COMPLETE_TOKEN_CAPTURE:
            raise RuntimeError(f"{row['trial_id']}: legacy token capture")
        expected_state = path.parent / "states" / f"{row['trial_id']}.pt"
        if Path(row["state_path"]).resolve() != expected_state.resolve():
            raise RuntimeError(
                f"{row['trial_id']}: state_path is outside current output"
            )
        metadata_to_trajectory(row)
        if index % 250 == 0:
            print(f"verified {index}/{len(rows)} trajectories", flush=True)
    print(
        f"PASS: {dataset}/{model}/{split}, always_idk={always_idk}, {len(rows)} complete trajectories"
    )


def check_gates(dataset, model=MODEL):
    from src.gate.select_layer import (
        MODEL_LAYERS,
        SEEDS,
        discover_layers,
        load_all_gates,
    )

    config = get_dataset_config(dataset)
    path = config.gate_root / model / "all_layers" / "train_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    validate_artifact_dataset(summary, config, path)
    expected = load_split_ids(config, "gate_train")
    if (
        summary.get("model_name") != model
        or summary.get("training_complete") is not True
        or set(summary.get("training_pair_ids", [])) != expected
        or summary.get("n_pairs") != len(expected)
        or summary.get("epochs") != 20
        or summary.get("n_layers") != MODEL_LAYERS[model]
        or summary.get("seeds") != list(SEEDS)
    ):
        raise RuntimeError(
            f"{path}: gate training does not match the full 20-epoch protocol"
        )
    layers = discover_layers(model, config)
    gates, means, stds, _ = load_all_gates(model, layers, torch.device("cpu"), config)
    for layer in layers:
        if (
            not torch.isfinite(means[layer]).all()
            or not torch.isfinite(stds[layer]).all()
            or not (stds[layer] > 0).all()
        ):
            raise RuntimeError(f"{model}/layer_{layer}: invalid normalizer")
        for seed in SEEDS:
            if any(
                not torch.isfinite(parameter).all()
                for parameter in gates[layer][seed].parameters()
            ):
                raise RuntimeError(
                    f"{model}/layer_{layer}/seed_{seed}: non-finite weights"
                )
    print(f"PASS: {dataset}/{model}, {len(layers) * len(SEEDS)} complete finite gates")


def check_intervention(dataset, split, model=MODEL):
    config = get_dataset_config(dataset)
    module = importlib.import_module(f"src.intervention.run_{split}_intervention")
    rows, _, _, _, planned, _ = module.plan_interventions(model, config)
    row_by_id = {row["trial_id"]: row for row in rows}
    for seed, jobs in planned.items():
        path = config.intervention_root / model / split / f"seed_{seed}.jsonl"
        completed, _ = load_existing(
            path,
            retry_errors=False,
            planned=jobs,
            dataset_config=config,
            model_name=model,
            split=split,
            seed=seed,
        )
        if set(completed) != {job["trial_id"] for job in jobs}:
            raise RuntimeError(f"{path}: incomplete intervention")
        for trial_id, row in completed.items():
            if not row["triggered"]:
                continue
            result = row.get("intervention_result")
            if (
                row.get("intervention_error")
                or not row.get("intervention_run")
                or not result
            ):
                raise RuntimeError(
                    f"{path}: {trial_id} intervention failed: {row.get('intervention_error')}"
                )
            if (
                model == MODEL
                and result.get("token_capture_method") != COMPLETE_TOKEN_CAPTURE
            ):
                raise RuntimeError(f"{trial_id}: legacy continuation token capture")
            prefix = row_by_id[trial_id]["token_ids"][: int(row["brake_checkpoint"])]
            if result["prefix_token_ids"] != prefix:
                raise RuntimeError(f"{trial_id}: intervention prefix mismatch")
            if (
                len(result["continuation_token_ids"])
                != result["num_continuation_tokens"]
            ):
                raise RuntimeError(f"{trial_id}: continuation count mismatch")
        print(
            f"PASS: {dataset}/{model}/{split}/seed_{seed}: {len(completed)} rows, no failed jobs"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("capture", "gates", "intervention"))
    parser.add_argument("--dataset", choices=("umwp", "worldsense"), required=True)
    parser.add_argument(
        "--split", choices=("gate_train", "calibration", "val", "test"), required=True
    )
    parser.add_argument("--always-idk", action="store_true")
    parser.add_argument("--model", choices=("af_next", MODEL, "moss"), default=MODEL)
    args = parser.parse_args()
    if args.stage == "capture":
        check_capture(args.dataset, args.split, args.always_idk, args.model)
    elif args.stage == "gates":
        if args.split != "gate_train" or args.always_idk:
            parser.error("gates requires gate_train and no --always-idk")
        check_gates(args.dataset, args.model)
    else:
        if args.split not in ("val", "test") or args.always_idk:
            parser.error("intervention requires val/test and no --always-idk")
        check_intervention(args.dataset, args.split, args.model)


if __name__ == "__main__":
    main()
