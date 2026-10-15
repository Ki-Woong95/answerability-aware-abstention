#!/usr/bin/env python3
"""v9_final TEST operating-point diagnostic using frozen selected layer and calibrated tau5."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.gate.calibrate_tau5 import (
    SEEDS,
    load_gate,
    load_selected_layer,
    trajectory_score,
)
from src.gate.datasets import (
    GateDatasetConfig,
    get_dataset_config,
    validate_artifact_dataset,
    validate_trajectory_rows,
)
from src.gate.select_layer import (
    MODEL_LAYERS,
    canonical_condition,
    load_sentence_states,
    read_metadata,
    validate_val_rows,
)


def validate_test_rows(rows):
    for row in rows:
        split = row.get("split")
        if split != "test":
            raise RuntimeError(
                f"{row.get('trial_id')}: refusing non-test split {split!r}"
            )
    proxy = [{**row, "split": "val"} for row in rows]
    return validate_val_rows(proxy)


def load_tau5(
    model_name: str,
    selected_layer: int,
    dataset_config: GateDatasetConfig | None = None,
) -> tuple[dict[int, float], dict[str, Any]]:
    dataset_config = (
        get_dataset_config("umwp") if dataset_config is None else dataset_config
    )
    path = dataset_config.calibration_root / f"{model_name}_tau5.json"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    validate_artifact_dataset(payload, dataset_config, path, model_name=model_name)
    if payload.get("model_name") != model_name:
        raise RuntimeError(f"{path}: model mismatch")
    if int(payload["selected_layer_number"]) != selected_layer:
        raise RuntimeError(
            f"{path}: calibrated layer {payload['selected_layer_number']} != selected layer {selected_layer}"
        )
    if payload.get("threshold_frozen_after_calibration") is not True:
        raise RuntimeError(f"{path}: threshold is not marked frozen")
    if payload.get("test_read") is True:
        raise RuntimeError(f"{path}: calibration touched test")
    results = payload.get("seed_results", {})
    tau5 = {seed: float(results[str(seed)]["tau5"]) for seed in SEEDS}
    if not all(
        np.isfinite(value) and 0 < value <= np.nextafter(1.0, np.inf)
        for value in tau5.values()
    ):
        raise RuntimeError(f"{path}: invalid tau5 threshold")
    return tau5, payload


def first_crossing(
    states: torch.Tensor,
    layer_number: int,
    gate,
    mean: torch.Tensor,
    std: torch.Tensor,
    tau: float,
    device: torch.device,
) -> int | None:
    if states.shape[0] == 0:
        return None
    x = states[:, layer_number - 1, :].to(device, non_blocking=True)
    x = (x - mean) / std
    with torch.no_grad():
        probs = torch.sigmoid(gate(x)).reshape(-1)
        cumulative = torch.cumsum(probs, 0) / torch.arange(
            1, probs.numel() + 1, device=device, dtype=probs.dtype
        )
        hit = torch.nonzero(cumulative.to(torch.float64) >= tau, as_tuple=False)
    return None if hit.numel() == 0 else int(hit[0, 0].item()) + 1


def run(
    model_name: str,
    device: torch.device,
    allow_test: bool,
    dataset_config: GateDatasetConfig,
) -> None:
    if not allow_test:
        raise RuntimeError("Refusing to touch test split without --allow-test")
    selected_layer, layer_info = load_selected_layer(model_name, dataset_config)
    tau5, calibration_info = load_tau5(model_name, selected_layer, dataset_config)
    metadata_path = (
        dataset_config.trajectory_root / model_name / "test" / "metadata.jsonl"
    )
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    rows = read_metadata(metadata_path)
    n_a, n_u, n_pairs = validate_test_rows(rows)
    validate_trajectory_rows(rows, dataset_config, "test", model_name)
    n_layers = MODEL_LAYERS[model_name]
    loaded = {
        seed: load_gate(model_name, selected_layer, seed, device, dataset_config)
        for seed in SEEDS
    }
    stats = {seed: {"answerable": [], "unanswerable": []} for seed in SEEDS}
    crossings = {seed: {"answerable": [], "unanswerable": []} for seed in SEEDS}

    print("=" * 80)
    print("V9_FINAL TEST OPERATING-POINT DIAGNOSTIC")
    print("=" * 80)
    print(f"dataset        : {dataset_config.name}")
    print(f"model          : {model_name}")
    print(f"selected layer : {selected_layer} (index {selected_layer - 1})")
    print(f"test A / U     : {n_a} / {n_u} ({n_pairs} pairs)")
    print(f"seeds          : {SEEDS}")
    print(
        "policy         : first sentence boundary where cumulative mean >= frozen tau5"
    )
    print("intervention   : NOT run here")
    print("test           : READ (final evaluation; frozen protocol)")

    for i, row in enumerate(rows, 1):
        condition = canonical_condition(row["condition"])
        states = load_sentence_states(
            row=row, metadata_path=metadata_path, n_layers=n_layers
        )
        for seed in SEEDS:
            gate, mean, std = loaded[seed]
            score = trajectory_score(states, selected_layer, gate, mean, std, device)
            crossed = score >= tau5[seed]
            stats[seed][condition].append(bool(crossed))
            if crossed:
                crossings[seed][condition].append(
                    first_crossing(
                        states, selected_layer, gate, mean, std, tau5[seed], device
                    )
                )
        if i % 50 == 0 or i == len(rows):
            print(f"[{i}/{len(rows)}] trajectories scored", flush=True)

    seed_results: dict[str, Any] = {}
    print("\nSeed operating points")
    print("-" * 84)
    print("seed      tau5      A-false       U-brake     A first-k     U first-k")
    print("-" * 84)
    for seed in SEEDS:
        a = np.asarray(stats[seed]["answerable"], dtype=np.int64)
        u = np.asarray(stats[seed]["unanswerable"], dtype=np.int64)
        a_rate, u_rate = float(a.mean()), float(u.mean())
        a_cross = crossings[seed]["answerable"]
        u_cross = crossings[seed]["unanswerable"]
        a_first = float(np.mean(a_cross)) if a_cross else None
        u_first = float(np.mean(u_cross)) if u_cross else None
        seed_results[str(seed)] = {
            "tau5": tau5[seed],
            "n_answerable": int(len(a)),
            "n_unanswerable": int(len(u)),
            "answerable_false_brakes": int(a.sum()),
            "answerable_false_brake_rate": a_rate,
            "unanswerable_brakes": int(u.sum()),
            "unanswerable_brake_rate": u_rate,
            "mean_first_crossing_sentence_A": a_first,
            "mean_first_crossing_sentence_U": u_first,
        }
        a_first_s = "-" if a_first is None else f"{a_first:.2f}"
        u_first_s = "-" if u_first is None else f"{u_first:.2f}"
        print(
            f"{seed:>4d}  {tau5[seed]:>9.6f}  {int(a.sum()):>3d}/{len(a):<3d} {a_rate:>7.3%}  {int(u.sum()):>3d}/{len(u):<3d} {u_rate:>7.3%}  {a_first_s:>10s}  {u_first_s:>10s}"
        )

    mean_a = float(
        np.mean([seed_results[str(s)]["answerable_false_brake_rate"] for s in SEEDS])
    )
    mean_u = float(
        np.mean([seed_results[str(s)]["unanswerable_brake_rate"] for s in SEEDS])
    )
    from src.data.frozen_umwp import v10_artifact_provenance

    output = {
        **v10_artifact_provenance(dataset_config.name, model_name),
        "schema_version": 1,
        "protocol": "v9_final_test_operating_point",
        "dataset": dataset_config.name,
        "model_name": model_name,
        "split": "test",
        "selected_layer_number": selected_layer,
        "selected_layer_index": selected_layer - 1,
        "selected_validation_mean_auc": layer_info.get("selected_mean_auc"),
        "tau5_source": str(dataset_config.calibration_root / f"{model_name}_tau5.json"),
        "calibration_method": calibration_info.get("calibration_method"),
        "risk_target": calibration_info.get("risk_target"),
        "seeds": list(SEEDS),
        "n_answerable": n_a,
        "n_unanswerable": n_u,
        "n_pairs": n_pairs,
        "decision_rule": "brake iff trajectory reaches cumulative_mean >= frozen per-seed tau5; first crossing is online brake point",
        "intervention_run": False,
        "test_read": True,
        "seed_results": seed_results,
        "mean_answerable_false_brake_rate_across_seeds": mean_a,
        "mean_unanswerable_brake_rate_across_seeds": mean_u,
    }
    dataset_config.operating_point_root.mkdir(parents=True, exist_ok=True)
    path = dataset_config.operating_point_root / f"{model_name}_test_tau5.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nmean across seeds: A-false={mean_a:.3%}, U-brake={mean_u:.3%}")
    print(f"saved: {path}")
    print(
        "Next: TEST online intervention/reassessment with the same frozen layer and tau5. Do NOT retune anything."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=("umwp", "worldsense"),
        default="umwp",
        help="Dataset namespace. Default keeps the frozen UMWP paths.",
    )
    parser.add_argument("--model", required=True, choices=tuple(MODEL_LAYERS))
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args()
    run(
        args.model,
        torch.device(args.device),
        allow_test=args.allow_test,
        dataset_config=get_dataset_config(args.dataset),
    )


if __name__ == "__main__":
    main()
