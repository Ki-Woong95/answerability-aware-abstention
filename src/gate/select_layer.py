#!/usr/bin/env python3
"""Select a decoder layer by mean validation trajectory AUROC across seeds.

Exact ties choose the lower layer. Threshold calibration is separate."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.gate.datasets import (
    GateDatasetConfig,
    get_dataset_config,
    validate_artifact_dataset,
    validate_trajectory_rows,
)
from src.gate.model import MLP64Gate


SEEDS = (17, 29, 43)
MODEL_LAYERS = {
    "af_next": 28,
    "audio-reasoner": 32,
    "moss": 36,
}


EXPECTED_PROTOCOL = [
    "v8_sentence_boundary_cumulative_mean",
    "v9_sentence_boundary_cumulative_mean",
]


def read_metadata(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_no}: invalid JSON") from exc
    return rows


def canonical_condition(value: Any) -> str:
    v = str(value).strip().lower()
    if v in {"a", "answerable"}:
        return "answerable"
    if v in {"u", "unanswerable"}:
        return "unanswerable"
    raise ValueError(f"unknown condition: {value!r}")


def resolve_state_path(row: dict[str, Any], metadata_path: Path) -> Path:
    raw = row.get("state_path")
    if not raw:
        raise KeyError(f"{row.get('trial_id')}: missing state_path")

    p = Path(raw).expanduser()
    if p.is_absolute() and p.exists():
        return p

    # Normal convention: project-root-relative path.
    if p.exists():
        return p.resolve()

    # Fallback: metadata-relative path.
    candidate = (metadata_path.parent / p).resolve()
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"{row.get('trial_id')}: state file not found: {raw}")


def _as_tensor(x: Any) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.detach().cpu()
    return torch.as_tensor(x)


def _sorted_mapping_to_tensor(
    mapping: dict[Any, Any],
) -> tuple[list[int], torch.Tensor]:
    items = sorted((int(k), _as_tensor(v)) for k, v in mapping.items())
    if not items:
        raise RuntimeError("empty all_layers mapping")
    checkpoints = [k for k, _ in items]
    states = torch.stack([v for _, v in items], dim=0)
    return checkpoints, states


def parse_state_payload(payload: Any) -> tuple[list[int] | None, torch.Tensor]:
    """Return (stored checkpoints or None, states [T,L,H] or [L,T,H])."""
    if torch.is_tensor(payload) or isinstance(payload, np.ndarray):
        states = _as_tensor(payload)
        checkpoints = None
    elif isinstance(payload, dict):
        all_layers = None
        for key in (
            "all_layers",
            "states",
            "layer_states",
            "hidden_states",
            "X",
        ):
            if key in payload:
                all_layers = payload[key]
                break

        if all_layers is None:
            raise KeyError(
                "state payload has no all-layer field; tried "
                "all_layers/states/layer_states/hidden_states/X"
            )

        if isinstance(all_layers, dict):
            checkpoints, states = _sorted_mapping_to_tensor(all_layers)
        else:
            states = _as_tensor(all_layers)
            checkpoints = None
            for key in (
                "checkpoints",
                "checkpoint_tokens",
                "positions",
                "stored_positions",
            ):
                if key in payload:
                    checkpoints = [int(x) for x in payload[key]]
                    break
    else:
        raise TypeError(f"unsupported state payload type: {type(payload)}")

    states = states.float().cpu()
    if states.ndim != 3:
        raise RuntimeError(
            f"expected all-layer states [T,L,H], got {tuple(states.shape)}"
        )

    return checkpoints, states


def orient_states(
    states: torch.Tensor,
    n_layers: int,
    n_checkpoints_hint: int | None,
) -> torch.Tensor:
    """Normalize all-layer state tensor to [T,L,H]."""
    if states.shape[1] == n_layers:
        return states
    if states.shape[0] == n_layers:
        return states.transpose(0, 1).contiguous()

    if n_checkpoints_hint is not None:
        if states.shape[0] == n_checkpoints_hint:
            return states
        if states.shape[1] == n_checkpoints_hint:
            return states.transpose(0, 1).contiguous()

    raise RuntimeError(
        f"cannot orient states {tuple(states.shape)} for {n_layers} layers"
    )


def load_sentence_states(
    row: dict[str, Any],
    metadata_path: Path,
    n_layers: int,
) -> torch.Tensor:
    """Load sentence-boundary all-layer states and return CPU float32 [S,L,H]."""
    sentence_cps = [int(x) for x in row.get("sentence_checkpoints", [])]
    if not sentence_cps and (
        row.get("eligible_checkpoint_count") != 0
        or row.get("monitoring_status") != "no_eligible_reasoning_checkpoint"
    ):
        raise RuntimeError(
            f"{row.get('trial_id')}: missing zero-checkpoint completion provenance"
        )

    path = resolve_state_path(row, metadata_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    stored_cps, states = parse_state_payload(payload)

    metadata_cps = [int(x) for x in row.get("checkpoints", [])]
    for name, positions in (
        ("sentence_checkpoints", sentence_cps),
        ("checkpoints", metadata_cps),
    ):
        if any(cp <= 0 for cp in positions) or any(
            b <= a for a, b in zip(positions, positions[1:])
        ):
            raise RuntimeError(
                f"{row.get('trial_id')}: {name} must be positive and strictly increasing"
            )
    if stored_cps is not None:
        if any(cp <= 0 for cp in stored_cps) or any(
            b <= a for a, b in zip(stored_cps, stored_cps[1:])
        ):
            raise RuntimeError(
                f"{row.get('trial_id')}: stored checkpoints must be positive and strictly increasing"
            )
        if metadata_cps and stored_cps != metadata_cps:
            raise RuntimeError(
                f"{row.get('trial_id')}: metadata checkpoints do not match state payload"
            )
    if stored_cps is None and metadata_cps:
        stored_cps = metadata_cps

    hint = len(stored_cps) if stored_cps is not None else None
    states = orient_states(states, n_layers=n_layers, n_checkpoints_hint=hint)

    if states.shape[1] != n_layers:
        raise RuntimeError(
            f"{row.get('trial_id')}: expected {n_layers} layers, got {states.shape[1]}"
        )

    # Current capture normally stores sentence positions only, but keep this
    # union-checkpoint path for compatibility with older saved payloads.
    if stored_cps is not None:
        if len(stored_cps) != states.shape[0]:
            raise RuntimeError(
                f"{row.get('trial_id')}: {len(stored_cps)} checkpoints vs "
                f"{states.shape[0]} state rows"
            )
        cp_to_index = {cp: i for i, cp in enumerate(stored_cps)}
        missing = [cp for cp in sentence_cps if cp not in cp_to_index]
        if missing:
            raise RuntimeError(
                f"{row.get('trial_id')}: sentence checkpoints missing from state payload: "
                f"{missing[:10]}"
            )
        states = states[[cp_to_index[cp] for cp in sentence_cps]]
    else:
        if states.shape[0] != len(sentence_cps):
            raise RuntimeError(
                f"{row.get('trial_id')}: state file has no checkpoint positions and "
                f"T={states.shape[0]} != sentence_count={len(sentence_cps)}"
            )

    if states.shape[2] <= 0:
        raise RuntimeError(f"{row.get('trial_id')}: unknown hidden dimension")
    if not torch.isfinite(states).all():
        raise RuntimeError(f"{row.get('trial_id')}: non-finite hidden states")

    return states.contiguous()


def validate_val_rows(rows: list[dict[str, Any]]) -> tuple[int, int, int]:
    """QC validation rows without hard-coding a dataset size."""
    if not rows:
        raise RuntimeError("validation metadata is empty")

    seen_trials: set[str] = set()
    grouped: dict[str, set[str]] = defaultdict(set)
    n_a = 0
    n_u = 0

    for row in rows:
        trial_id = str(row.get("trial_id") or "")
        if not trial_id:
            raise KeyError("validation row missing trial_id")
        if trial_id in seen_trials:
            raise RuntimeError(f"duplicate validation trial_id: {trial_id}")
        seen_trials.add(trial_id)

        split = str(row.get("split", ""))
        if split not in {"val", "validation"}:
            raise RuntimeError(f"{trial_id}: refusing non-validation split {split!r}")

        cond = canonical_condition(row.get("condition"))
        if cond == "answerable":
            n_a += 1
        else:
            n_u += 1

        pair_id = str(row.get("pair_id") or row.get("source_id") or "")
        if not pair_id:
            raise KeyError(f"{trial_id}: missing pair_id/source_id")
        if cond in grouped[pair_id]:
            raise RuntimeError(f"{pair_id}: duplicate {cond} validation trajectory")
        grouped[pair_id].add(cond)

    incomplete = [
        pid for pid, conds in grouped.items() if conds != {"answerable", "unanswerable"}
    ]
    if incomplete:
        raise RuntimeError(
            f"validation has {len(incomplete)} incomplete A/U pairs; first: {incomplete[:5]}"
        )

    if n_a == 0 or n_u == 0:
        raise RuntimeError(f"validation requires both classes, got A={n_a}, U={n_u}")

    return n_a, n_u, len(grouped)


def standardizer_from_checkpoint(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    payload = checkpoint["standardizer"]

    mean = torch.as_tensor(payload["mean"], dtype=torch.float32, device=device)
    M2 = torch.as_tensor(payload["M2"], dtype=torch.float32, device=device)
    count = int(payload["count"])
    eps = float(payload["eps"])

    # RunningStandardizer.transform() requires at least two observations and uses:
    #   sqrt(max(M2/count, eps^2))
    if count < 2:
        raise ValueError(f"standardizer count must be >=2, got {count}")

    variance = M2 / count
    floor = torch.tensor(eps * eps, dtype=variance.dtype, device=device)
    std = torch.sqrt(torch.maximum(variance, floor))

    return mean, std


def discover_layers(model_name: str, dataset_config: GateDatasetConfig) -> list[int]:
    root = dataset_config.gate_root / model_name / "all_layers"
    layer_dirs = sorted(root.glob("layer_*"))

    if not layer_dirs:
        raise RuntimeError(f"No trained all-layer gates found: {root}")

    layers: list[int] = []
    for d in layer_dirs:
        try:
            layer_number = int(d.name.split("_")[1])
        except (IndexError, ValueError) as exc:
            raise RuntimeError(f"Malformed layer directory: {d}") from exc
        layers.append(layer_number)

    expected = MODEL_LAYERS[model_name]
    expected_layers = list(range(1, expected + 1))
    if layers != expected_layers:
        raise RuntimeError(
            f"{model_name}: expected layer directories 1..{expected}, found {layers}"
        )

    return layers


def load_all_gates(
    model_name: str,
    layers: list[int],
    device: torch.device,
    dataset_config: GateDatasetConfig,
) -> tuple[
    dict[int, dict[int, MLP64Gate]],
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
    int,
]:
    """Return gates[layer][seed], means[layer], stds[layer], hidden_dim."""
    gates: dict[int, dict[int, MLP64Gate]] = {}
    means: dict[int, torch.Tensor] = {}
    stds: dict[int, torch.Tensor] = {}
    hidden_dim: int | None = None

    for layer_number in layers:
        gates[layer_number] = {}
        reference_mean: torch.Tensor | None = None
        reference_std: torch.Tensor | None = None

        for seed in SEEDS:
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
            validate_artifact_dataset(
                checkpoint, dataset_config, path, model_name=model_name
            )

            if checkpoint.get("protocol") not in EXPECTED_PROTOCOL:
                raise RuntimeError(
                    f"{path}: protocol={checkpoint.get('protocol')!r}, "
                    f"expected {EXPECTED_PROTOCOL!r}"
                )
            if checkpoint.get("model_name") != model_name:
                raise RuntimeError(
                    f"{path}: model mismatch: {checkpoint.get('model_name')!r}"
                )
            if checkpoint.get("training_positions") != "sentence_boundaries_only":
                raise RuntimeError(
                    f"{path}: gate was not trained on sentence-boundary positions only"
                )
            if int(checkpoint["layer_number_1based"]) != layer_number:
                raise RuntimeError(f"{path}: layer number mismatch")
            if int(checkpoint["layer_index_0based"]) != layer_number - 1:
                raise RuntimeError(f"{path}: layer index mismatch")
            if int(checkpoint["seed"]) != seed:
                raise RuntimeError(f"{path}: seed mismatch")

            this_hidden_dim = int(checkpoint["hidden_dim"])
            if hidden_dim is None:
                hidden_dim = this_hidden_dim
            elif this_hidden_dim != hidden_dim:
                raise RuntimeError(
                    f"{path}: hidden_dim={this_hidden_dim}, expected {hidden_dim}"
                )

            gate = MLP64Gate(hidden_dim=this_hidden_dim).to(device)
            gate.load_state_dict(checkpoint["gate_state_dict"])
            gate.eval()
            gates[layer_number][seed] = gate

            mean, std = standardizer_from_checkpoint(checkpoint, device=device)
            if reference_mean is None:
                reference_mean = mean
                reference_std = std
            else:
                if not torch.equal(mean, reference_mean):
                    raise RuntimeError(
                        f"Layer {layer_number}: standardizer mean differs across seeds"
                    )
                if not torch.equal(std, reference_std):
                    raise RuntimeError(
                        f"Layer {layer_number}: standardizer std differs across seeds"
                    )

        assert reference_mean is not None and reference_std is not None
        means[layer_number] = reference_mean
        stds[layer_number] = reference_std

    assert hidden_dim is not None
    return gates, means, stds, hidden_dim


def trajectory_score(probs: torch.Tensor) -> float:
    """S = max_k mean(p_1, ..., p_k)."""
    probs = probs.reshape(-1)
    if probs.numel() == 0:
        return 0.0

    cumulative_mean = torch.cumsum(probs, dim=0) / torch.arange(
        1,
        probs.numel() + 1,
        device=probs.device,
        dtype=probs.dtype,
    )
    return float(cumulative_mean.max().detach().cpu())


def run(
    model_name: str, device: torch.device, dataset_config: GateDatasetConfig
) -> None:
    if model_name not in MODEL_LAYERS:
        raise ValueError(model_name)

    metadata_path = (
        dataset_config.trajectory_root / model_name / "val" / "metadata.jsonl"
    )
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    rows = read_metadata(metadata_path)
    n_a, n_u, n_pairs = validate_val_rows(rows)
    validate_trajectory_rows(rows, dataset_config, "val", model_name)

    layers = discover_layers(model_name, dataset_config)
    gates, means, stds, hidden_dim = load_all_gates(
        model_name=model_name,
        layers=layers,
        device=device,
        dataset_config=dataset_config,
    )
    n_layers = MODEL_LAYERS[model_name]

    print("=" * 88)
    print("VALIDATION ALL-LAYER SELECTION")
    print("=" * 88)
    print(f"dataset      : {dataset_config.name}")
    print(f"model        : {model_name}")
    print(f"split        : val ONLY")
    print(f"trials       : {len(rows)}")
    print(f"pairs        : {n_pairs}")
    print(f"A / U        : {n_a} / {n_u}")
    print(f"layers       : {n_layers}")
    print(f"hidden dim   : {hidden_dim}")
    print(f"seeds        : {SEEDS}")
    print(f"device       : {device}")
    print("positions    : sentence boundaries ONLY")
    print("score        : S = max cumulative mean gate probability")
    print("threshold    : NOT calibrated here")
    print("test         : NOT read")
    print()

    # scores[layer][seed][condition] -> trajectory-level S values
    scores: dict[int, dict[int, dict[str, list[float]]]] = {
        layer: {seed: {"answerable": [], "unanswerable": []} for seed in SEEDS}
        for layer in layers
    }

    for row_index, row in enumerate(rows, 1):
        states = load_sentence_states(
            row=row,
            metadata_path=metadata_path,
            n_layers=n_layers,
        )

        if states.shape[1] != n_layers:
            raise RuntimeError(
                f"{row['trial_id']}: expected {n_layers} layers, got {states.shape[1]}"
            )
        if states.shape[2] != hidden_dim:
            raise RuntimeError(
                f"{row['trial_id']}: hidden_dim={states.shape[2]}, expected {hidden_dim}"
            )

        condition = canonical_condition(row["condition"])
        states = states.to(device, non_blocking=True)

        if states.shape[0] == 0:
            for layer_number in layers:
                for seed in SEEDS:
                    scores[layer_number][seed][condition].append(0.0)
            continue

        for layer_number in layers:
            layer_index = layer_number - 1
            x = states[:, layer_index, :]
            x = (x - means[layer_number]) / stds[layer_number]

            with torch.no_grad():
                for seed in SEEDS:
                    probs = torch.sigmoid(gates[layer_number][seed](x))
                    scores[layer_number][seed][condition].append(
                        trajectory_score(probs)
                    )

        if row_index % 50 == 0 or row_index == len(rows):
            print(f"[{row_index}/{len(rows)}] trajectories scored", flush=True)

    results: list[dict[str, Any]] = []

    for layer_number in layers:
        seed_aucs: dict[int, float] = {}

        for seed in SEEDS:
            a_scores = np.asarray(
                scores[layer_number][seed]["answerable"], dtype=np.float64
            )
            u_scores = np.asarray(
                scores[layer_number][seed]["unanswerable"], dtype=np.float64
            )

            labels = np.concatenate(
                [
                    np.zeros(len(a_scores), dtype=np.int64),
                    np.ones(len(u_scores), dtype=np.int64),
                ]
            )
            trajectory_scores = np.concatenate([a_scores, u_scores])
            seed_aucs[seed] = float(roc_auc_score(labels, trajectory_scores))

        auc_values = np.asarray([seed_aucs[seed] for seed in SEEDS], dtype=np.float64)
        results.append(
            {
                "layer_number": layer_number,
                "layer_index": layer_number - 1,
                "seed_17_auc": seed_aucs[17],
                "seed_29_auc": seed_aucs[29],
                "seed_43_auc": seed_aucs[43],
                "mean_auc": float(auc_values.mean()),
                "sd_auc": float(auc_values.std(ddof=1)),
            }
        )

    # Frozen rule inherited from V8: highest seed-mean AUROC; exact tie -> lower layer.
    best = max(
        results,
        key=lambda x: (x["mean_auc"], -x["layer_number"]),
    )

    print()
    print("Validation trajectory AUROC")
    print("-" * 76)
    print("Layer     seed17     seed29     seed43       mean         sd")
    print("-" * 76)
    for r in results:
        marker = "  <== SELECTED" if r["layer_number"] == best["layer_number"] else ""
        print(
            f"{r['layer_number']:>5d}  "
            f"{r['seed_17_auc']:>9.4f}  "
            f"{r['seed_29_auc']:>9.4f}  "
            f"{r['seed_43_auc']:>9.4f}  "
            f"{r['mean_auc']:>9.4f}  "
            f"{r['sd_auc']:>9.4f}"
            f"{marker}"
        )

    from src.data.frozen_umwp import v10_artifact_provenance

    output = {
        **v10_artifact_provenance(dataset_config.name, model_name),
        "zero_checkpoint_counts": {
            c: sum(
                not r["sentence_checkpoints"]
                for r in rows
                if canonical_condition(r["condition"]) == c
            )
            for c in ("answerable", "unanswerable")
        },
        "zero_checkpoint_score": 0.0,
        "schema_version": 1,
        "protocol": "v9_validation_layer_selection",
        "gate_protocol": EXPECTED_PROTOCOL,
        "dataset": dataset_config.name,
        "model_name": model_name,
        "split": "val",
        "n_trials": len(rows),
        "n_pairs": n_pairs,
        "n_answerable": n_a,
        "n_unanswerable": n_u,
        "seeds": list(SEEDS),
        "positions": "sentence_boundaries_only",
        "aggregation": "cumulative_mean",
        "trajectory_score": "max_cumulative_mean",
        "positive_class": "unanswerable",
        "selection_metric": "mean trajectory AUROC across seeds 17,29,43",
        "tie_break": "exact mean-AUROC tie -> lower layer number",
        "threshold_calibrated": False,
        "test_read": False,
        "selected_layer_number": best["layer_number"],
        "selected_layer_index": best["layer_index"],
        "selected_mean_auc": best["mean_auc"],
        "selected_sd_auc": best["sd_auc"],
        "layers": results,
        "next_step": "A-only calibration on selected layer; derive per-seed tau5. Do not touch test.",
    }

    dataset_config.layer_selection_root.mkdir(parents=True, exist_ok=True)
    json_path = dataset_config.layer_selection_root / f"{model_name}_all_layers.json"
    csv_path = dataset_config.layer_selection_root / f"{model_name}_all_layers.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "layer_number",
                "layer_index",
                "seed_17_auc",
                "seed_29_auc",
                "seed_43_auc",
                "mean_auc",
                "sd_auc",
                "generation_stop_policy",
            ],
        )
        writer.writeheader()
        writer.writerows(
            dict(row, generation_stop_policy=output.get("generation_stop_policy"))
            for row in results
        )

    print()
    print("LAYER SELECTION COMPLETE")
    print(f"SELECTED: layer {best['layer_number']} " f"(index {best['layer_index']})")
    print(f"mean AUROC = {best['mean_auc']:.4f} ± {best['sd_auc']:.4f}")
    print(f"saved: {json_path}")
    print(f"saved: {csv_path}")
    print("Next: A-only calibration on this layer. Do NOT touch test.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=("umwp", "worldsense"),
        default="umwp",
        help="Dataset namespace. Default keeps the frozen UMWP paths.",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=tuple(MODEL_LAYERS),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Scoring device. No speech model is loaded.",
    )
    args = parser.parse_args()

    run(
        model_name=args.model,
        device=torch.device(args.device),
        dataset_config=get_dataset_config(args.dataset),
    )


if __name__ == "__main__":
    main()
