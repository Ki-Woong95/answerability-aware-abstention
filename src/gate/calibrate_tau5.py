#!/usr/bin/env python3
"""v9_final A-only calibration for the online sentence-boundary intervention threshold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import binom

from src.gate.datasets import (
    GateDatasetConfig,
    get_dataset_config,
    load_split_ids,
    validate_artifact_dataset,
    validate_trajectory_rows,
)
from src.gate.model import MLP64Gate
from src.gate.select_layer import (
    MODEL_LAYERS,
    load_sentence_states,
    read_metadata,
    standardizer_from_checkpoint,
)

SEEDS = (17, 29, 43)
TARGET_A_FALSE_BRAKE = 0.05
LTT_DELTA = 0.05
ALLOWED_GATE_PROTOCOLS = {
    "v8_sentence_boundary_cumulative_mean",
    "v9_sentence_boundary_cumulative_mean",
}


def _dataset_config(config: GateDatasetConfig | None) -> GateDatasetConfig:
    # Keep imported helper functions backward-compatible with the frozen UMWP
    # intervention code, which calls them without a dataset argument.
    return get_dataset_config("umwp") if config is None else config


def load_selected_layer(
    model_name: str,
    dataset_config: GateDatasetConfig | None = None,
) -> tuple[int, dict[str, Any]]:
    dataset_config = _dataset_config(dataset_config)
    path = dataset_config.layer_selection_root / f"{model_name}_all_layers.json"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        result = json.load(f)
    validate_artifact_dataset(result, dataset_config, path, model_name=model_name)
    if result.get("model_name") != model_name:
        raise RuntimeError(f"{path}: model mismatch")
    if result.get("split") not in {"val", "validation"}:
        raise RuntimeError(f"{path}: layer was not selected on validation")
    if result.get("test_read") is True:
        raise RuntimeError(f"{path}: refusing layer selection that read test")
    layer = int(result["selected_layer_number"])
    if not 1 <= layer <= MODEL_LAYERS[model_name]:
        raise RuntimeError(f"{path}: invalid selected layer {layer}")
    return layer, result


def load_calibration_rows(
    model_name: str,
    dataset_config: GateDatasetConfig | None = None,
) -> tuple[list[dict[str, Any]], Path]:
    dataset_config = _dataset_config(dataset_config)
    calibration_ids = load_split_ids(dataset_config, "calibration")
    expected = dataset_config.expected_counts("calibration")[0]
    if len(calibration_ids) != expected:
        raise RuntimeError(
            f"{dataset_config.name}: expected {expected} calibration pairs, "
            f"got {len(calibration_ids)}"
        )

    metadata_path = (
        dataset_config.trajectory_root / model_name / "calibration" / "metadata.jsonl"
    )
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"{metadata_path} (capture the calibration split before threshold calibration)"
        )
    rows = read_metadata(metadata_path)
    selected: list[dict[str, Any]] = []
    seen_trials: set[str] = set()
    seen_pairs: set[str] = set()
    for row in rows:
        trial_id = str(row.get("trial_id") or "")
        pair_id = str(row.get("pair_id") or row.get("source_id") or "")
        condition = str(row.get("condition") or "").strip().lower()
        if not trial_id or not pair_id:
            raise RuntimeError("calibration metadata row missing trial_id/pair_id")
        if trial_id in seen_trials:
            raise RuntimeError(f"duplicate calibration trial_id: {trial_id}")
        seen_trials.add(trial_id)
        if str(row.get("split", "")) != "calibration":
            raise RuntimeError(
                f"{trial_id}: expected split='calibration', got {row.get('split')!r}"
            )
        if pair_id not in calibration_ids:
            raise RuntimeError(
                f"{trial_id}: pair_id {pair_id!r} is not in final calibration split"
            )
        if condition in {"a", "answerable"}:
            if pair_id in seen_pairs:
                raise RuntimeError(f"duplicate answerable calibration pair: {pair_id}")
            seen_pairs.add(pair_id)
            selected.append(row)
        elif condition not in {"u", "unanswerable"}:
            raise RuntimeError(f"{trial_id}: unknown condition {condition!r}")

    if len(selected) != expected:
        raise RuntimeError(
            f"{model_name}: expected {expected} answerable calibration trajectories, "
            f"got {len(selected)}"
        )
    if seen_pairs != calibration_ids:
        missing = sorted(calibration_ids - seen_pairs)
        extra = sorted(seen_pairs - calibration_ids)
        raise RuntimeError(
            f"{model_name}: calibration A pair mismatch; missing={missing[:5]}, extra={extra[:5]}"
        )
    validate_trajectory_rows(selected, dataset_config, "calibration", model_name)
    return sorted(selected, key=lambda x: str(x["trial_id"])), metadata_path


def load_gate(
    model_name: str,
    layer_number: int,
    seed: int,
    device: torch.device,
    dataset_config: GateDatasetConfig | None = None,
) -> tuple[MLP64Gate, torch.Tensor, torch.Tensor]:
    dataset_config = _dataset_config(dataset_config)
    path = (
        dataset_config.gate_root
        / model_name
        / "all_layers"
        / f"layer_{layer_number:02d}"
        / f"seed_{seed}.pt"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    validate_artifact_dataset(checkpoint, dataset_config, path, model_name=model_name)
    if checkpoint.get("protocol") not in ALLOWED_GATE_PROTOCOLS:
        raise RuntimeError(
            f"{path}: unsupported protocol {checkpoint.get('protocol')!r}"
        )
    if checkpoint.get("model_name") != model_name:
        raise RuntimeError(f"{path}: model mismatch")
    if checkpoint.get("training_positions") != "sentence_boundaries_only":
        raise RuntimeError(f"{path}: gate was not trained on sentence boundaries only")
    if (
        int(checkpoint["layer_number_1based"]) != layer_number
        or int(checkpoint["layer_index_0based"]) != layer_number - 1
    ):
        raise RuntimeError(f"{path}: layer mismatch")
    if int(checkpoint["seed"]) != seed:
        raise RuntimeError(f"{path}: seed mismatch")
    hidden_dim = int(checkpoint["hidden_dim"])
    gate = MLP64Gate(hidden_dim=hidden_dim).to(device)
    gate.load_state_dict(checkpoint["gate_state_dict"])
    gate.eval()
    mean, std = standardizer_from_checkpoint(checkpoint, device=device)
    return gate, mean, std


def trajectory_score(
    states: torch.Tensor,
    layer_number: int,
    gate: MLP64Gate,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> float:
    if states.shape[0] == 0:
        return 0.0
    x = states[:, layer_number - 1, :].to(device, non_blocking=True)
    x = (x - mean) / std
    with torch.no_grad():
        probs = torch.sigmoid(gate(x)).reshape(-1)
    cumulative = torch.cumsum(probs, dim=0) / torch.arange(
        1, probs.numel() + 1, device=device, dtype=probs.dtype
    )
    return float(cumulative.max().cpu())


def calibrate_tau5_ltt(scores: np.ndarray) -> dict[str, Any]:
    """Fixed-sequence LTT-style calibration for R_A(tau)=P_A(S>=tau) <= 0.05."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("scores must be a non-empty finite 1-D array")
    n = len(scores)

    # H_tau: R_A(tau) >= 0.05. For nested thresholds tested from conservative
    # to aggressive, BinomCDF(k; n, .05) is the one-sided fixed-sequence p-value.
    certified_k = -1
    for k in range(n + 1):
        if float(binom.cdf(k, n, TARGET_A_FALSE_BRAKE)) <= LTT_DELTA:
            certified_k = k
        else:
            break
    if certified_k < 0:
        raise RuntimeError(
            "No threshold can be certified at the requested risk/confidence level"
        )

    ordered = np.sort(scores)
    cutoff = ordered[n - certified_k - 1]
    threshold = np.nextafter(cutoff, np.inf)
    observed = int(np.sum(scores >= threshold))
    empirical = observed / n
    p_value = float(binom.cdf(observed, n, TARGET_A_FALSE_BRAKE))
    if observed > certified_k or p_value > LTT_DELTA + 1e-15:
        raise RuntimeError("LTT calibration failed internal verification")

    return {
        "tau5": float(threshold),
        "target_true_A_false_brake_risk": TARGET_A_FALSE_BRAKE,
        "ltt_delta": LTT_DELTA,
        "confidence": 1.0 - LTT_DELTA,
        "n_answerable": n,
        "certified_max_false_brakes": int(certified_k),
        "observed_false_brakes": observed,
        "empirical_A_false_brake_rate": float(empirical),
        "binomial_p_value_at_target_risk": p_value,
    }


def run(
    model_name: str,
    device: torch.device,
    dataset_config: GateDatasetConfig,
) -> None:
    layer_number, layer_selection = load_selected_layer(model_name, dataset_config)
    rows, metadata_path = load_calibration_rows(model_name, dataset_config)
    n_layers = MODEL_LAYERS[model_name]

    print("=" * 80)
    print("V9_FINAL A-ONLY TAU5 CALIBRATION")
    print("=" * 80)
    print(f"dataset      : {dataset_config.name}")
    print(f"model        : {model_name}")
    print(f"selected layer: {layer_number} (index {layer_number - 1})")
    print(f"calibration A: {len(rows)}")
    print(f"risk target  : A false-brake <= {TARGET_A_FALSE_BRAKE:.2%}")
    print(f"confidence   : {1.0 - LTT_DELTA:.0%}")
    print("score        : S = max_k cumulative_mean(p_1..p_k)")
    print("purpose      : frozen online intervention operating threshold")
    print("test         : NOT read")

    from src.data.frozen_umwp import v10_artifact_provenance

    result: dict[str, Any] = {
        **v10_artifact_provenance(dataset_config.name, model_name),
        "zero_checkpoint_count": sum(not r["sentence_checkpoints"] for r in rows),
        "zero_checkpoint_score": 0.0,
        "schema_version": 1,
        "protocol": "v9_final_tau5_calibration",
        "dataset": dataset_config.name,
        "model_name": model_name,
        "selected_layer_number": layer_number,
        "selected_layer_index": layer_number - 1,
        "selected_validation_mean_auc": layer_selection.get("selected_mean_auc"),
        "selected_validation_sd_auc": layer_selection.get("selected_sd_auc"),
        "calibration_split": "calibration",
        "calibration_data": "answerable trajectories only",
        "n_answerable": len(rows),
        "seeds": list(SEEDS),
        "positions": "sentence_boundaries_only",
        "trajectory_score": "max_cumulative_mean",
        "risk_target": "P_A(S >= tau5) <= 0.05",
        "calibration_method": "fixed_sequence_LTT_one_sided_binomial",
        "ltt_delta": LTT_DELTA,
        "brake_rule": "online: intervene at the first sentence boundary k where cumulative_mean(p_1..p_k) >= tau5",
        "threshold_purpose": "online_intervention_operating_point",
        "threshold_frozen_after_calibration": True,
        "test_read": False,
        "seed_results": {},
    }

    for seed in SEEDS:
        gate, mean, std = load_gate(
            model_name, layer_number, seed, device, dataset_config
        )
        scores = []
        for i, row in enumerate(rows, 1):
            states = load_sentence_states(
                row=row, metadata_path=metadata_path, n_layers=n_layers
            )
            scores.append(
                trajectory_score(states, layer_number, gate, mean, std, device)
            )
            if i % 50 == 0 or i == len(rows):
                print(f"seed={seed} [{i}/{len(rows)}]", flush=True)
        scores_np = np.asarray(scores, dtype=np.float64)
        calibrated = calibrate_tau5_ltt(scores_np)
        from src.gate.v10_eligibility import validate_positive_threshold

        validate_positive_threshold(calibrated["tau5"])
        calibrated.update(
            {
                "score_mean": float(scores_np.mean()),
                "score_std": float(scores_np.std(ddof=1)),
                "score_min": float(scores_np.min()),
                "score_max": float(scores_np.max()),
            }
        )
        result["seed_results"][str(seed)] = calibrated
        print(
            f"seed={seed}: tau5={calibrated['tau5']:.6f}, A-false={calibrated['empirical_A_false_brake_rate']:.3%} ({calibrated['observed_false_brakes']}/{len(rows)}), p={calibrated['binomial_p_value_at_target_risk']:.4g}"
        )

    dataset_config.calibration_root.mkdir(parents=True, exist_ok=True)
    path = dataset_config.calibration_root / f"{model_name}_tau5.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"saved: {path}")
    print(
        "Next: frozen validation intervention with these per-seed tau5 values. Do NOT touch test."
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
    args = parser.parse_args()
    run(
        args.model,
        torch.device(args.device),
        get_dataset_config(args.dataset),
    )


if __name__ == "__main__":
    main()
