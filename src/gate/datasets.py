from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GateDatasetConfig:
    name: str
    split_path: Path
    trajectory_root: Path
    gate_root: Path
    layer_selection_root: Path
    calibration_root: Path
    operating_point_root: Path
    intervention_root: Path
    judge_root: Path
    analysis_root: Path
    expected_calibration_a: int | None

    def expected_counts(self, split_name: str) -> tuple[int, int, int]:
        n_pairs = len(load_split_ids(self, split_name))

        if split_name == "calibration":
            if (
                self.expected_calibration_a is not None
                and n_pairs != self.expected_calibration_a
            ):
                raise RuntimeError(
                    f"{self.name}: calibration split has {n_pairs} ids, "
                    f"expected {self.expected_calibration_a}"
                )
            return n_pairs, 0, n_pairs

        return n_pairs, n_pairs, 2 * n_pairs


DATASETS: dict[str, GateDatasetConfig] = {
    "treecut": GateDatasetConfig(
        name="treecut",
        split_path=Path("data/splits/treecut_final_split.json"),
        trajectory_root=Path("data/treecut/trajectories"),
        gate_root=Path("data/treecut/gates"),
        layer_selection_root=Path("data/treecut/layer_selection"),
        calibration_root=Path("data/treecut/calibration"),
        operating_point_root=Path("data/treecut/operating_point"),
        intervention_root=Path("data/treecut/interventions"),
        judge_root=Path("data/treecut/judge"),
        analysis_root=Path("data/treecut/analysis"),
        expected_calibration_a=None,
    ),
    "umwp": GateDatasetConfig(
        name="umwp",
        split_path=Path("data/splits/umwp_final_split.json"),
        trajectory_root=Path("data/trajectories"),
        gate_root=Path("data/gates"),
        layer_selection_root=Path("data/layer_selection"),
        calibration_root=Path("data/calibration"),
        operating_point_root=Path("data/operating_point"),
        intervention_root=Path("data/interventions"),
        judge_root=Path("data/judge"),
        analysis_root=Path("data/analysis"),
        expected_calibration_a=None,
    ),
    "worldsense": GateDatasetConfig(
        name="worldsense",
        split_path=Path("data/splits/worldsense_openended_split.json"),
        trajectory_root=Path("data/worldsense/trajectories"),
        gate_root=Path("data/worldsense/gates"),
        layer_selection_root=Path("data/worldsense/layer_selection"),
        calibration_root=Path("data/worldsense/calibration"),
        operating_point_root=Path("data/worldsense/operating_point"),
        intervention_root=Path("data/worldsense/interventions"),
        judge_root=Path("data/worldsense/judge"),
        analysis_root=Path("data/worldsense/analysis"),
        expected_calibration_a=400,
    ),
}


def get_dataset_config(name: str) -> GateDatasetConfig:
    try:
        return DATASETS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown dataset {name!r}; choose one of {tuple(DATASETS)}"
        ) from exc


def _entry_id(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("pair_id", "source_id"):
            if value.get(key) is not None:
                return str(value[key])
    raise ValueError(f"Cannot extract pair/source id from split entry: {value!r}")


def load_split_ids(config: GateDatasetConfig, split_name: str) -> set[str]:
    path = config.split_path
    if not path.exists():
        raise FileNotFoundError(path)

    from src.data.frozen_umwp import verify_frozen_split

    verify_frozen_split(path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    # UMWP legacy split files may store split lists at the top level, while
    # WorldSense uses {"splits": {...}}. Support both without changing the
    # downstream gate protocol.
    split_block = payload.get("splits", payload)
    if split_name not in split_block:
        raise KeyError(f"{path}: missing split {split_name!r}")

    ids = [_entry_id(x) for x in split_block[split_name]]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"{path}: duplicate ids inside split {split_name!r}")
    for other_split in ("gate_train", "calibration", "val", "test"):
        if other_split != split_name and other_split in split_block:
            overlap = set(ids) & {_entry_id(x) for x in split_block[other_split]}
            if overlap:
                raise RuntimeError(
                    f"{path}: ids shared by {split_name}/{other_split}: {sorted(overlap)[:5]}"
                )
    return set(ids)


def validate_artifact_dataset(
    payload: dict[str, Any],
    config: GateDatasetConfig,
    path: Path,
    model_name: str | None = None,
) -> None:
    """Prevent cross-dataset artifact mixing while keeping legacy UMWP readable."""
    from src.data.frozen_umwp import CANONICAL_SPLIT, v10_artifact_provenance

    if config.name == "treecut":
        from src.data.frozen_treecut import validate_provenance

        validate_provenance(payload, model_name)
    if (
        config.name == "umwp"
        and config.split_path.resolve() == CANONICAL_SPLIT.resolve()
    ):
        expected = v10_artifact_provenance(
            config.name, model_name or payload.get("model_name")
        )
        if any(payload.get(key) != value for key, value in expected.items()):
            raise RuntimeError(
                f"{path}: missing/stale V10 artifact provenance (including AR generation stop policy)"
            )
    artifact_dataset = payload.get("dataset")
    if payload.get("training_complete") is False:
        raise RuntimeError(f"{path}: gate was trained on a partial smoke-run split")

    if artifact_dataset is None:
        # Existing frozen UMWP artifacts predate the dataset field. Preserve
        # backward compatibility, but require explicit provenance for any new
        # dataset so WorldSense can never silently consume a UMWP artifact.
        if config.name == "umwp":
            return
        raise RuntimeError(
            f"{path}: artifact has no dataset provenance; refusing to use it "
            f"for dataset={config.name!r}"
        )

    if str(artifact_dataset) != config.name:
        raise RuntimeError(
            f"{path}: artifact dataset={artifact_dataset!r}, "
            f"requested dataset={config.name!r}"
        )


def validate_trajectory_rows(
    rows, config, split, model_name=None, require_complete=True
):
    """Check membership, identities and paired coverage of the frozen split."""
    expected_ids = load_split_ids(config, split)
    seen_trials = set()
    grouped = {}
    from src.data.frozen_umwp import CANONICAL_SPLIT, frozen_split_record

    is_v10 = (
        config.name == "umwp"
        and config.split_path.resolve() == CANONICAL_SPLIT.resolve()
    )
    if is_v10:
        from src.capture.v10_prompt import TASK_PROMPT
        from src.models.probing import BOUNDARY_VERSION
        import hashlib

        rule_hash = hashlib.sha256(
            Path("src/models/probing.py").read_bytes()
        ).hexdigest()
    for row in rows:
        if config.name == "treecut":
            from src.data.frozen_treecut import TASK_PROMPT, validate_provenance

            validate_provenance(row, model_name)
            if row.get("dataset") != "treecut" or row.get("prompt_text") != TASK_PROMPT:
                raise RuntimeError(
                    "Refusing TreeCut trajectory with stale prompt/dataset"
                )
            if row.get("eligible_checkpoint_count") != len(
                row.get("sentence_checkpoints", [])
            ):
                raise RuntimeError(
                    "TreeCut checkpoint count disagrees with saved positions"
                )
        if is_v10:
            if row.get("model_name") == "audio-reasoner":
                from src.models.audio_reasoner_stop import STOP_POLICY

                if row.get("generation_stop_policy") != STOP_POLICY:
                    raise RuntimeError(
                        "Refusing stale AR generation stop policy in trajectories"
                    )
            if (
                row.get("prompt_text") != TASK_PROMPT
                or row.get("boundary_version") != BOUNDARY_VERSION
                or row.get("boundary_source_sha256") != rule_hash
                or row.get("frozen_split_sha256") != frozen_split_record()["sha256"]
            ):
                raise RuntimeError(
                    "Refusing stale V10 prompt, boundary rules or frozen-split provenance"
                )
            if row.get("eligible_checkpoint_count") != len(
                row.get("sentence_checkpoints", [])
            ):
                raise RuntimeError(
                    "V10 eligible checkpoint count disagrees with saved positions"
                )
        trial_id = str(row.get("trial_id") or "")
        pair_id = str(row.get("pair_id") or row.get("source_id") or "")
        if not trial_id or trial_id in seen_trials:
            raise RuntimeError(f"{split}: missing or duplicate trial_id {trial_id!r}")
        seen_trials.add(trial_id)
        if pair_id not in expected_ids:
            raise RuntimeError(
                f"{trial_id}: pair {pair_id!r} outside frozen {split} split"
            )
        if row.get("split") not in (
            {"val", "validation"} if split == "val" else {split}
        ):
            raise RuntimeError(f"{trial_id}: incorrect split {row.get('split')!r}")
        if model_name is not None and row.get("model_name") != model_name:
            raise RuntimeError(f"{trial_id}: model_name does not match {model_name!r}")
        condition = str(row.get("condition", "")).lower().strip()
        condition = {"a": "answerable", "u": "unanswerable"}.get(condition, condition)
        if condition not in ("answerable", "unanswerable"):
            raise RuntimeError(f"{trial_id}: unknown condition {condition!r}")
        conditions = grouped.setdefault(pair_id, set())
        if condition in conditions:
            raise RuntimeError(f"{pair_id}: duplicate {condition} trajectory")
        conditions.add(condition)
    if require_complete:
        if config.name == "treecut":
            from src.capture.treecut_failures import canonical_population

            active, _ = canonical_population(
                config, split, model_name or (rows[0]["model_name"] if rows else None)
            )
            expected_ids = {r["pair_id"] for r in active}
            grouped = {k: v for k, v in grouped.items() if k in expected_ids}
        captured_ids = set(grouped)
        if captured_ids != expected_ids:
            raise RuntimeError(
                f"incomplete {split} capture: expected {len(expected_ids)} pairs, "
                f"got {len(captured_ids)}"
            )
        expected_conditions = (
            {"answerable"} if split == "calibration" else {"answerable", "unanswerable"}
        )
        if any(conditions != expected_conditions for conditions in grouped.values()):
            raise RuntimeError(f"{split}: incomplete or unexpected A/U conditions")
