#!/usr/bin/env python3
"""Train independent decoder-layer gates with a pair-balanced A/U loss.

Normalization uses training states only; validation and test are not read."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.gate.datasets import (
    GateDatasetConfig,
    get_dataset_config,
    load_split_ids,
    validate_trajectory_rows,
)
from src.gate.model import MLP64Gate
from src.gate.normalization import RunningStandardizer


SEEDS = (17, 29, 43)
MODEL_LAYERS = {
    "af_next": 28,
    "audio-reasoner": 32,
    "moss": 36,
}

EPOCHS = 20
ORDER_SEED = 20260917
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 5.0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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


def canonical_condition(value: str) -> str:
    v = str(value).strip().lower()
    if v in {"a", "answerable"}:
        return "answerable"
    if v in {"u", "unanswerable"}:
        return "unanswerable"
    raise ValueError(f"unknown condition: {value!r}")


def build_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    for row in rows:
        split = str(row.get("split", ""))
        if split != "gate_train":
            raise RuntimeError(
                f"Refusing non-gate_train row {row.get('trial_id')}: split={split!r}"
            )

        pair_id = str(row.get("pair_id") or row.get("source_id") or "")
        if not pair_id:
            raise KeyError(f"{row.get('trial_id')}: missing pair_id/source_id")

        cond = canonical_condition(row["condition"])
        if cond in grouped[pair_id]:
            raise RuntimeError(f"{pair_id}: duplicate {cond}")
        grouped[pair_id][cond] = row

    pairs = []
    for pair_id in sorted(grouped):
        pair = grouped[pair_id]
        if set(pair) != {"answerable", "unanswerable"}:
            raise RuntimeError(f"{pair_id}: incomplete A/U pair: {sorted(pair)}")
        pairs.append(
            {
                "pair_id": pair_id,
                "answerable": pair["answerable"],
                "unanswerable": pair["unanswerable"],
            }
        )

    if not pairs:
        raise RuntimeError("No complete gate_train A/U pairs found")
    return pairs


def resolve_state_path(row: dict[str, Any], metadata_path: Path) -> Path:
    raw = row.get("state_path")
    if not raw:
        raise KeyError(f"{row.get('trial_id')}: missing state_path")

    p = Path(raw).expanduser()
    if p.is_absolute() and p.exists():
        return p

    # Project-root-relative path (normal V9 metadata convention).
    if p.exists():
        return p.resolve()

    # Fallback for metadata containing a path relative to metadata.jsonl.
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
    """Return (stored checkpoints or None, states [T,L,H])."""
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

    # Preferred shape is [T,L,H]. Some save code may write [L,T,H].
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
    """Normalize shape to [T,L,H]."""
    if states.shape[1] == n_layers:
        return states
    if states.shape[0] == n_layers:
        return states.transpose(0, 1).contiguous()

    # If checkpoint count is informative, use it as a final disambiguator.
    if n_checkpoints_hint is not None:
        if states.shape[0] == n_checkpoints_hint:
            return states
        if states.shape[1] == n_checkpoints_hint:
            return states.transpose(0, 1).contiguous()

    raise RuntimeError(
        f"cannot orient states {tuple(states.shape)} for {n_layers} layers"
    )


def sentence_only_states(
    row: dict[str, Any],
    metadata_path: Path,
    n_layers: int,
) -> torch.Tensor:
    """Load and return sentence-boundary states [S,L,H]."""
    sentence_cps = [int(x) for x in row.get("sentence_checkpoints", [])]
    if not sentence_cps and (
        row.get("eligible_checkpoint_count") != 0
        or row.get("monitoring_status") != "no_eligible_reasoning_checkpoint"
    ):
        raise RuntimeError(
            f"{row.get('trial_id')}: missing zero-checkpoint completion provenance"
        )

    state_path = resolve_state_path(row, metadata_path)
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
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

    # V9 normally stores sentence checkpoints only; this also safely handles union-style payloads.
    if stored_cps is not None:
        if len(stored_cps) != states.shape[0]:
            raise RuntimeError(
                f"{row.get('trial_id')}: {len(stored_cps)} checkpoints vs "
                f"{states.shape[0]} state rows"
            )
        index = {cp: i for i, cp in enumerate(stored_cps)}
        missing = [cp for cp in sentence_cps if cp not in index]
        if missing:
            raise RuntimeError(
                f"{row.get('trial_id')}: sentence checkpoints missing from state payload: "
                f"{missing[:10]}"
            )
        states = states[[index[cp] for cp in sentence_cps]]
    else:
        # Safe only when the file itself contains exactly sentence-boundary states.
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


def infer_hidden_dim(
    pair: dict[str, Any],
    metadata_path: Path,
    n_layers: int,
) -> int:
    states = sentence_only_states(pair["answerable"], metadata_path, n_layers)
    return int(states.shape[-1])


def fit_layer_standardizers(
    pairs: list[dict[str, Any]],
    metadata_path: Path,
    n_layers: int,
    hidden_dim: int,
) -> list[RunningStandardizer]:
    standardizers = [RunningStandardizer(hidden_dim) for _ in range(n_layers)]

    for i, pair in enumerate(pairs, 1):
        a = sentence_only_states(pair["answerable"], metadata_path, n_layers).numpy()
        u = sentence_only_states(pair["unanswerable"], metadata_path, n_layers).numpy()

        for layer_idx in range(n_layers):
            standardizers[layer_idx].update(
                np.concatenate([a[:, layer_idx, :], u[:, layer_idx, :]], axis=0)
            )

        if i % 100 == 0 or i == len(pairs):
            print(f"  normalization {i}/{len(pairs)} pairs", flush=True)

    return standardizers


def cache_normalized_pairs(
    pairs: list[dict[str, Any]],
    metadata_path: Path,
    n_layers: int,
    standardizers: list[RunningStandardizer],
) -> list[dict[str, Any]]:
    """
    Cache one pair at a time in RAM as [S,L,H].
    Float16 cache cuts memory in half; conversion to float32 happens at training.
    """
    cached = []
    for i, pair in enumerate(pairs, 1):
        a = sentence_only_states(pair["answerable"], metadata_path, n_layers).numpy()
        u = sentence_only_states(pair["unanswerable"], metadata_path, n_layers).numpy()

        a_norm = np.empty_like(a, dtype=np.float16)
        u_norm = np.empty_like(u, dtype=np.float16)
        for layer_idx in range(n_layers):
            a_norm[:, layer_idx, :] = (
                standardizers[layer_idx]
                .transform(a[:, layer_idx, :])
                .astype(np.float16)
            )
            u_norm[:, layer_idx, :] = (
                standardizers[layer_idx]
                .transform(u[:, layer_idx, :])
                .astype(np.float16)
            )

        cached.append({"pair_id": pair["pair_id"], "a": a_norm, "u": u_norm})
        if i % 100 == 0 or i == len(pairs):
            print(f"  caching {i}/{len(pairs)} pairs", flush=True)

    return cached


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_gate_bank(
    n_layers: int,
    hidden_dim: int,
    device: torch.device,
) -> tuple[list[dict[int, MLP64Gate]], list[dict[int, torch.optim.Optimizer]]]:
    gates: list[dict[int, MLP64Gate]] = []
    optimizers: list[dict[int, torch.optim.Optimizer]] = []

    for layer_idx in range(n_layers):
        layer_gates = {}
        layer_opts = {}
        for seed in SEEDS:
            # Layer-dependent offset avoids identical initial weights across layers
            # while remaining deterministic.
            init_seed = seed + 1000 * (layer_idx + 1)
            seed_everything(init_seed)
            gate = MLP64Gate(hidden_dim=hidden_dim).to(device)
            opt = torch.optim.AdamW(
                gate.parameters(),
                lr=LR,
                weight_decay=WEIGHT_DECAY,
            )
            layer_gates[seed] = gate
            layer_opts[seed] = opt
        gates.append(layer_gates)
        optimizers.append(layer_opts)

    return gates, optimizers


def train_one_pair(
    gate: MLP64Gate,
    optimizer: torch.optim.Optimizer,
    a: np.ndarray,
    u: np.ndarray,
    device: torch.device,
    dropout_seed: int,
) -> tuple[float, float, float, float]:
    seed_everything(dropout_seed)

    h_a = torch.from_numpy(a.astype(np.float32, copy=False)).to(device)
    h_u = torch.from_numpy(u.astype(np.float32, copy=False)).to(device)

    gate.train()
    optimizer.zero_grad(set_to_none=True)

    logits_a = gate(h_a)
    logits_u = gate(h_u)

    loss_a = F.binary_cross_entropy_with_logits(
        logits_a, torch.zeros_like(logits_a), reduction="mean"
    )
    loss_u = F.binary_cross_entropy_with_logits(
        logits_u, torch.ones_like(logits_u), reduction="mean"
    )
    loss = 0.5 * (loss_a + loss_u)

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(gate.parameters(), GRAD_CLIP)
    optimizer.step()

    return (
        float(loss.detach().cpu()),
        float(loss_a.detach().cpu()),
        float(loss_u.detach().cpu()),
        float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm),
    )


def standardizer_payload(s: RunningStandardizer) -> dict[str, Any]:
    return {
        "count": int(s.count),
        "mean": torch.from_numpy(s.mean.copy()),
        "M2": torch.from_numpy(s.M2.copy()),
        "eps": float(s.eps),
    }


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def save_gate(
    path: Path,
    dataset_name: str,
    model_name: str,
    layer_number: int,
    seed: int,
    hidden_dim: int,
    gate: MLP64Gate,
    standardizer: RunningStandardizer,
    epochs: int,
    training_pair_ids: list[str],
    training_complete: bool,
) -> None:
    from src.data.frozen_umwp import v10_artifact_provenance

    gate.eval()
    atomic_torch_save(
        {
            **v10_artifact_provenance(dataset_name, model_name),
            "schema_version": 1,
            "protocol": "v9_sentence_boundary_cumulative_mean",
            "dataset": dataset_name,
            "model_name": model_name,
            "layer_number_1based": int(layer_number),
            "layer_index_0based": int(layer_number - 1),
            "seed": int(seed),
            "hidden_dim": int(hidden_dim),
            "architecture": "mlp64",
            "training_positions": "sentence_boundaries_only",
            "training_loss": (
                "0.5 * mean_BCE(A,0) + 0.5 * mean_BCE(U,1); "
                "cumulative mean NOT in training loss"
            ),
            "epochs": int(epochs),
            "training_pair_ids": training_pair_ids,
            "training_complete": training_complete,
            "lr": float(LR),
            "weight_decay": float(WEIGHT_DECAY),
            "grad_clip": float(GRAD_CLIP),
            "gate_state_dict": gate.state_dict(),
            "standardizer": standardizer_payload(standardizer),
        },
        path,
    )


def train_model(
    model_name: str,
    device: torch.device,
    epochs: int,
    limit_pairs: int | None,
    overwrite: bool,
    dataset_config: GateDatasetConfig,
) -> None:
    if model_name not in MODEL_LAYERS:
        raise ValueError(model_name)
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    n_layers = MODEL_LAYERS[model_name]
    metadata_path = (
        dataset_config.trajectory_root / model_name / "gate_train" / "metadata.jsonl"
    )
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    rows = read_jsonl(metadata_path)
    validate_trajectory_rows(
        rows, dataset_config, "gate_train", model_name, require_complete=False
    )
    generation_failure_coverage = None
    expected_ids = load_split_ids(dataset_config, "gate_train")
    if dataset_config.name == "treecut":
        from src.capture.treecut_failures import canonical_population

        active, generation_failure_coverage = canonical_population(
            dataset_config, "gate_train", model_name
        )
        expected_ids = {r["pair_id"] for r in active}
        rows = [r for r in rows if r["pair_id"] in expected_ids]
    pairs = build_pairs(rows)
    captured_ids = {pair["pair_id"] for pair in pairs}
    extra = captured_ids - expected_ids
    if extra:
        raise RuntimeError(
            f"{metadata_path}: pairs outside frozen gate_train split: {sorted(extra)[:5]}"
        )
    if limit_pairs is None and captured_ids != expected_ids:
        raise RuntimeError(
            f"{metadata_path}: incomplete gate_train capture: expected {len(expected_ids)} pairs, "
            f"got {len(captured_ids)}. Finish capture or use --limit-pairs for an explicit smoke run."
        )
    if limit_pairs is not None:
        if limit_pairs <= 0 or len(pairs) < limit_pairs:
            raise ValueError(f"--limit-pairs must be between 1 and {len(pairs)}")
        pairs = pairs[:limit_pairs]

    capture_complete = captured_ids == expected_ids and limit_pairs is None
    from src.data.frozen_umwp import v10_artifact_provenance
    from src.gate.v10_eligibility import filter_pairs_for_state_fitting

    # Validate even excluded state files, so missing/corrupt captures are not coverage loss.
    for pair in pairs:
        for condition_name in ("answerable", "unanswerable"):
            sentence_only_states(pair[condition_name], metadata_path, n_layers)
    pairs, coverage_report = filter_pairs_for_state_fitting(pairs, model_name)
    if generation_failure_coverage is not None:
        coverage_report["generation_failure_coverage"] = generation_failure_coverage
    report_path = dataset_config.gate_root / model_name / "state_fitting_coverage.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(coverage_report, indent=2) + "\n")
    print(json.dumps(coverage_report, indent=2), flush=True)
    if not pairs:
        raise RuntimeError("No complete nonempty A/U pairs; refusing state fitting")

    output_root = dataset_config.gate_root / model_name / "all_layers"
    if output_root.exists() and not overwrite:
        existing = list(output_root.glob("layer_*/seed_*.pt"))
        if existing:
            raise FileExistsError(
                f"{output_root} already contains trained gates. "
                "Use --overwrite only if you intentionally want to retrain."
            )
    output_root.mkdir(parents=True, exist_ok=True)

    hidden_dim = infer_hidden_dim(pairs[0], metadata_path, n_layers)

    print("=" * 88)
    print("V9 ALL-LAYER GATE TRAINING")
    print("=" * 88)
    print(f"dataset     : {dataset_config.name}")
    print(f"model       : {model_name}")
    print(f"trials      : {len(pairs) * 2}")
    print(f"pairs       : {len(pairs)}")
    print(f"layers      : {n_layers}")
    print(f"hidden dim  : {hidden_dim}")
    print(f"seeds       : {SEEDS}")
    print(f"epochs      : {epochs}")
    print(f"device      : {device}")
    print("positions   : sentence boundaries ONLY")
    print("split       : gate_train ONLY")
    print()

    print("Fitting train-only normalization for every layer...", flush=True)
    standardizers = fit_layer_standardizers(
        pairs=pairs,
        metadata_path=metadata_path,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
    )

    print("Caching normalized sentence-boundary states...", flush=True)
    cached = cache_normalized_pairs(
        pairs=pairs,
        metadata_path=metadata_path,
        n_layers=n_layers,
        standardizers=standardizers,
    )

    gates, optimizers = make_gate_bank(n_layers, hidden_dim, device)

    history = []
    for epoch in range(epochs):
        order = list(range(len(cached)))
        random.Random(ORDER_SEED + epoch).shuffle(order)

        # sums[layer][seed] -> loss stats
        sums = [
            {
                seed: {"loss": 0.0, "loss_a": 0.0, "loss_u": 0.0, "grad": 0.0}
                for seed in SEEDS
            }
            for _ in range(n_layers)
        ]

        for step, idx in enumerate(order, 1):
            item = cached[idx]
            for layer_idx in range(n_layers):
                a = item["a"][:, layer_idx, :]
                u = item["u"][:, layer_idx, :]

                for seed in SEEDS:
                    dropout_seed = (
                        seed * 1_000_003
                        + (layer_idx + 1) * 10_007
                        + (epoch + 1) * 1_009
                        + step
                    )
                    loss, loss_a, loss_u, grad = train_one_pair(
                        gate=gates[layer_idx][seed],
                        optimizer=optimizers[layer_idx][seed],
                        a=a,
                        u=u,
                        device=device,
                        dropout_seed=dropout_seed,
                    )
                    s = sums[layer_idx][seed]
                    s["loss"] += loss
                    s["loss_a"] += loss_a
                    s["loss_u"] += loss_u
                    s["grad"] += grad

            if step % 100 == 0 or step == len(order):
                print(
                    f"  epoch {epoch + 1:02d}/{epochs:02d} | "
                    f"pair {step:04d}/{len(order):04d}",
                    flush=True,
                )

        epoch_row = {"epoch": epoch + 1, "layers": {}}
        for layer_idx in range(n_layers):
            layer_key = f"layer_{layer_idx + 1:02d}"
            epoch_row["layers"][layer_key] = {}
            for seed in SEEDS:
                s = sums[layer_idx][seed]
                n = len(order)
                epoch_row["layers"][layer_key][str(seed)] = {
                    "mean_loss": s["loss"] / n,
                    "mean_loss_a": s["loss_a"] / n,
                    "mean_loss_u": s["loss_u"] / n,
                    "mean_grad_norm": s["grad"] / n,
                }

        history.append(epoch_row)

        mean_loss = np.mean(
            [
                epoch_row["layers"][f"layer_{l:02d}"][str(seed)]["mean_loss"]
                for l in range(1, n_layers + 1)
                for seed in SEEDS
            ]
        )
        print(f"epoch {epoch + 1:02d} mean bank loss = {mean_loss:.5f}", flush=True)

        # Epoch-level restart point.
        atomic_torch_save(
            {
                **v10_artifact_provenance(dataset_config.name, model_name),
                "dataset": dataset_config.name,
                "model_name": model_name,
                "epoch_completed": epoch + 1,
                "n_layers": n_layers,
                "hidden_dim": hidden_dim,
                "seeds": list(SEEDS),
                "gate_states": {
                    f"layer_{l + 1:02d}": {
                        str(seed): gates[l][seed].state_dict() for seed in SEEDS
                    }
                    for l in range(n_layers)
                },
                "optimizer_states": {
                    f"layer_{l + 1:02d}": {
                        str(seed): optimizers[l][seed].state_dict() for seed in SEEDS
                    }
                    for l in range(n_layers)
                },
                "standardizers": {
                    f"layer_{l + 1:02d}": standardizer_payload(standardizers[l])
                    for l in range(n_layers)
                },
                "history": history,
            },
            output_root / "training_state.pt",
        )

    print("Saving frozen per-layer gates...", flush=True)
    for layer_idx in range(n_layers):
        for seed in SEEDS:
            save_gate(
                path=(output_root / f"layer_{layer_idx + 1:02d}" / f"seed_{seed}.pt"),
                dataset_name=dataset_config.name,
                model_name=model_name,
                layer_number=layer_idx + 1,
                seed=seed,
                hidden_dim=hidden_dim,
                gate=gates[layer_idx][seed],
                standardizer=standardizers[layer_idx],
                epochs=epochs,
                training_pair_ids=[pair["pair_id"] for pair in pairs],
                training_complete=capture_complete,
            )

    with (output_root / "train_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    from src.data.frozen_umwp import v10_artifact_provenance

    summary = {
        **v10_artifact_provenance(dataset_config.name, model_name),
        "schema_version": 1,
        "protocol": "v9_sentence_boundary_cumulative_mean",
        "dataset": dataset_config.name,
        "model_name": model_name,
        "split": "gate_train",
        "n_trials": len(pairs) * 2,
        "n_pairs": len(pairs),
        "state_fitting_coverage": coverage_report,
        "training_pair_ids": [pair["pair_id"] for pair in pairs],
        "training_complete": capture_complete,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "seeds": list(SEEDS),
        "epochs": epochs,
        "order_seed": ORDER_SEED,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "architecture": "MLP64Gate: H -> 64 -> 1, GELU, Dropout(0.1)",
        "training_positions": "sentence_checkpoints only",
        "fixed_checkpoints_used_for_training": False,
        "normalization": "train-only feature-wise standardization, separate per layer",
        "loss": "0.5*mean_BCE(A,0) + 0.5*mean_BCE(U,1)",
        "cumulative_mean_used_in_training": False,
        "next_step": "score val sentence-boundary trajectories, compute S=max(cumulative_mean(p)), select layer by mean AUROC across seeds",
    }
    with (output_root / "train_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("TRAINING COMPLETE")
    print(f"saved: {output_root}")
    print("Next: validation layer selection. Do NOT touch test.")


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
        help="Gate training device. No speech model is loaded.",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--limit-pairs", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.limit_pairs is not None and args.limit_pairs <= 0:
        raise ValueError("--limit-pairs must be positive")

    train_model(
        model_name=args.model,
        device=torch.device(args.device),
        epochs=args.epochs,
        limit_pairs=args.limit_pairs,
        overwrite=args.overwrite,
        dataset_config=get_dataset_config(args.dataset),
    )


if __name__ == "__main__":
    main()
