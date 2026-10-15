from __future__ import annotations

from src.eval.efficiency import GENERATED_OUTPUT_TOKEN_METRIC

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.gate.datasets import (
    GateDatasetConfig,
    get_dataset_config,
    validate_artifact_dataset,
    validate_trajectory_rows,
)


MODELS = ("af_next", "audio-reasoner", "moss")
MODEL_LABELS = {
    "af_next": "AF-Next",
    "audio-reasoner": "Audio-Reasoner",
    "moss": "MOSS",
}
SEEDS = (17, 29, 43)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def safe_rate(num: int | float, den: int | float) -> float:
    return float(num / den) if den else np.nan


def ms(values) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan, np.nan
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else np.nan
    return mean, sd


def load_judgments(
    config: GateDatasetConfig,
    split: str,
    rubric: str,
):
    vanilla_path = config.judge_root / f"{split}_vanilla_judged_{rubric}.jsonl"
    intervention_path = (
        config.judge_root / f"{split}_intervention_judged_{rubric}.jsonl"
    )

    if not vanilla_path.exists():
        raise FileNotFoundError(vanilla_path)
    if not intervention_path.exists():
        raise FileNotFoundError(intervention_path)

    vanilla_rows = read_jsonl(vanilla_path)
    intervention_rows = read_jsonl(intervention_path)

    vanilla_map = {}
    for row in vanilla_rows:
        validate_artifact_dataset(
            row, config, vanilla_path, model_name=row.get("model")
        )
        validate_judgment(row, rubric)
        key = (str(row["model"]), str(row["trial_id"]))
        if key in vanilla_map:
            raise RuntimeError(f"Duplicate vanilla judgment: {key}")
        vanilla_map[key] = row

    intervention_map = {}
    for row in intervention_rows:
        validate_artifact_dataset(
            row, config, intervention_path, model_name=row.get("model")
        )
        validate_judgment(row, rubric)
        key = (
            str(row["model"]),
            int(row["seed"]),
            str(row["trial_id"]),
        )
        if key in intervention_map:
            raise RuntimeError(f"Duplicate intervention judgment: {key}")
        intervention_map[key] = row

    expected_a, expected_u, expected_total = config.expected_counts(split)
    expected_vanilla = len(MODELS) * expected_total

    if len(vanilla_map) != expected_vanilla:
        raise RuntimeError(
            f"Expected {expected_vanilla} vanilla judgments, " f"got {len(vanilla_map)}"
        )

    return vanilla_rows, intervention_rows, vanilla_map, intervention_map


def validate_judgment(row, rubric):
    allowed = (
        {"correct", "incorrect", "abstain", "nonanswer"}
        if row.get("condition") == "answerable"
        else (
            {"abstain", "specific_answer", "nonanswer"}
            if row.get("condition") == "unanswerable"
            else set()
        )
    )
    if row.get("label") not in allowed:
        raise RuntimeError(f"{row.get('trial_id')}: invalid judgment label/condition")
    if row.get("rubric_version") != rubric:
        raise RuntimeError(
            f"{row.get('trial_id')}: judgment rubric does not match {rubric!r}"
        )


def load_trial_metadata(
    config: GateDatasetConfig,
    split: str,
) -> dict[tuple[str, str], dict]:
    metadata = {}
    _, _, expected_total = config.expected_counts(split)

    for model in MODELS:
        path = config.trajectory_root / model / split / "metadata.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)

        rows = read_jsonl(path)
        validate_trajectory_rows(rows, config, split, model)

        if len(rows) != expected_total:
            raise RuntimeError(
                f"{path}: expected {expected_total} rows, got {len(rows)}"
            )

        for row in rows:
            key = (model, str(row["trial_id"]))
            if key in metadata:
                raise RuntimeError(f"Duplicate trajectory metadata: {key}")
            metadata[key] = row

    return metadata


def build_trial_table(
    config: GateDatasetConfig,
    split: str,
    vanilla_map: dict,
    intervention_map: dict,
    metadata_map: dict,
) -> pd.DataFrame:
    records = []
    expected_trigger_keys = set()

    expected_a, expected_u, expected_total = config.expected_counts(split)

    for model in MODELS:
        for seed in SEEDS:
            path = config.intervention_root / model / split / f"seed_{seed}.jsonl"

            if not path.exists():
                raise FileNotFoundError(path)

            rows = read_jsonl(path)

            if len(rows) != expected_total:
                raise RuntimeError(
                    f"{model} seed={seed}: expected "
                    f"{expected_total} rows, got {len(rows)}"
                )

            ids = [str(row["trial_id"]) for row in rows]
            if len(ids) != len(set(ids)):
                raise RuntimeError(f"Duplicate trial_id within {path}")

            n_a = sum(row["condition"] == "answerable" for row in rows)
            n_u = sum(row["condition"] == "unanswerable" for row in rows)

            if n_a != expected_a or n_u != expected_u:
                raise RuntimeError(
                    f"{model} seed={seed}: expected "
                    f"{expected_a} A + {expected_u} U, "
                    f"got {n_a} A + {n_u} U"
                )

            for row in rows:
                validate_artifact_dataset(row, config, path)
                if (
                    row.get("model_name") != model
                    or row.get("seed") != seed
                    or row.get("split") != split
                ):
                    raise RuntimeError(
                        f"{path}: intervention model/seed/split mismatch"
                    )
                trial_id = str(row["trial_id"])
                condition = row["condition"]
                vanilla_key = (model, trial_id)

                if vanilla_key not in vanilla_map:
                    raise RuntimeError(f"Missing vanilla judgment: {vanilla_key}")

                if vanilla_key not in metadata_map:
                    raise RuntimeError(f"Missing trajectory metadata: {vanilla_key}")

                vanilla_judge = vanilla_map[vanilla_key]
                meta = metadata_map[vanilla_key]

                if meta.get("condition") != condition:
                    raise RuntimeError(
                        f"Trajectory condition mismatch: " f"{model} {trial_id}"
                    )

                if vanilla_judge["condition"] != condition:
                    raise RuntimeError(
                        f"Judge condition mismatch: " f"{model} {trial_id}"
                    )

                row_pair = row.get("pair_id")
                meta_pair = meta.get("pair_id")
                if (
                    row_pair is not None
                    and meta_pair is not None
                    and str(row_pair) != str(meta_pair)
                ):
                    raise RuntimeError(f"Pair mismatch: {model} {trial_id}")

                triggered = bool(row["triggered"])
                intervention_key = (model, seed, trial_id)

                if triggered:
                    expected_trigger_keys.add(intervention_key)

                    if intervention_key not in intervention_map:
                        raise RuntimeError(
                            f"Missing intervention judgment: " f"{intervention_key}"
                        )

                    intervention_judge = intervention_map[intervention_key]
                    if intervention_judge["condition"] != condition:
                        raise RuntimeError(
                            f"Intervention judge condition mismatch: {intervention_key}"
                        )
                    if row.get("intervention_error") or not row.get("intervention_run"):
                        raise RuntimeError(
                            f"Failed intervention cannot be included in metrics: {intervention_key}"
                        )
                    intervention_label = intervention_judge["label"]
                    intervention_hit_limit = bool(
                        intervention_judge.get(
                            "hit_token_limit",
                            False,
                        )
                    )
                    final_label = intervention_label
                else:
                    intervention_label = None
                    intervention_hit_limit = False
                    final_label = vanilla_judge["label"]

                vanilla_tokens = int(
                    row.get(
                        "vanilla_num_generated_tokens",
                        0,
                    )
                    or 0
                )

                brake_cp = row.get("brake_checkpoint")
                result = row.get("intervention_result") or {}

                continuation_tokens = int(
                    result.get(
                        "num_continuation_tokens",
                        0,
                    )
                    or 0
                )

                if triggered:
                    if brake_cp is None:
                        raise RuntimeError(
                            f"Triggered row missing brake checkpoint: "
                            f"{model} seed={seed} {trial_id}"
                        )
                    system_tokens = int(brake_cp) + continuation_tokens
                else:
                    system_tokens = vanilla_tokens

                if vanilla_tokens > 0:
                    token_reduction_fraction = (
                        vanilla_tokens - system_tokens
                    ) / vanilla_tokens
                else:
                    token_reduction_fraction = np.nan

                records.append(
                    {
                        "model": model,
                        "seed": seed,
                        "trial_id": trial_id,
                        "pair_id": row.get("pair_id"),
                        "source": meta.get("source"),
                        "category": meta.get("category"),
                        "family": (
                            row.get("family")
                            if row.get("family") is not None
                            else meta.get("family")
                        ),
                        "condition": condition,
                        "question_type": (
                            row.get("question_type")
                            if row.get("question_type") is not None
                            else meta.get("question_type")
                        ),
                        "n_entities": (
                            row.get("n_entities")
                            if row.get("n_entities") is not None
                            else meta.get("n_entities")
                        ),
                        "threshold": float(row["tau5"]),
                        "triggered": triggered,
                        "trajectory_score": row.get("trajectory_score"),
                        "brake_checkpoint": brake_cp,
                        "brake_sentence_index": row.get("sentence_index_1based"),
                        "brake_fraction_of_vanilla": row.get(
                            "completion_fraction_at_brake"
                        ),
                        "vanilla_tokens": vanilla_tokens,
                        "continuation_tokens": continuation_tokens,
                        "system_tokens": system_tokens,
                        "token_reduction_fraction": token_reduction_fraction,
                        "vanilla_label": vanilla_judge["label"],
                        "intervention_label": intervention_label,
                        "final_label": final_label,
                        "vanilla_hit_token_limit": bool(
                            vanilla_judge.get(
                                "hit_token_limit",
                                False,
                            )
                        ),
                        "intervention_hit_token_limit": intervention_hit_limit,
                    }
                )

    actual_intervention_keys = set(intervention_map)

    missing = expected_trigger_keys - actual_intervention_keys
    extra = actual_intervention_keys - expected_trigger_keys

    if missing:
        raise RuntimeError(
            f"{len(missing)} triggered trials missing "
            f"intervention judgments; "
            f"examples={sorted(missing)[:5]}"
        )

    if extra:
        raise RuntimeError(
            f"{len(extra)} intervention judgments do not "
            f"correspond to triggered trials; "
            f"examples={sorted(extra)[:5]}"
        )

    df = pd.DataFrame(records)

    expected_rows = len(MODELS) * len(SEEDS) * expected_total

    if len(df) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} system rows, " f"got {len(df)}")

    return df


def compute_seed_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for model in MODELS:
        for seed in SEEDS:
            d = df[(df.model == model) & (df.seed == seed)]

            a = d[d.condition == "answerable"]
            u = d[d.condition == "unanswerable"]

            at = a[a.triggered]
            ut = u[u.triggered]

            at_correct = at[at.vanilla_label == "correct"]
            ut_specific = ut[ut.vanilla_label == "specific_answer"]
            ut_abstain = ut[ut.vanilla_label == "abstain"]

            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "threshold": float(d.threshold.iloc[0]),
                    "a_false_rate": float(a.triggered.mean()),
                    "u_brake_rate": float(u.triggered.mean()),
                    "a_vanilla_accuracy": float((a.vanilla_label == "correct").mean()),
                    "a_system_accuracy": float((a.final_label == "correct").mean()),
                    "a_accuracy_delta": float(
                        (a.final_label == "correct").mean()
                        - (a.vanilla_label == "correct").mean()
                    ),
                    "a_vanilla_abstain": float((a.vanilla_label == "abstain").mean()),
                    "a_system_abstain": float((a.final_label == "abstain").mean()),
                    "u_vanilla_abstain": float((u.vanilla_label == "abstain").mean()),
                    "u_system_abstain": float((u.final_label == "abstain").mean()),
                    "u_abstain_delta": float(
                        (u.final_label == "abstain").mean()
                        - (u.vanilla_label == "abstain").mean()
                    ),
                    "u_vanilla_hallucinated": float(
                        (u.vanilla_label == "specific_answer").mean()
                    ),
                    "u_system_hallucinated": float(
                        (u.final_label == "specific_answer").mean()
                    ),
                    "u_hallucinated_delta": float(
                        (u.final_label == "specific_answer").mean()
                        - (u.vanilla_label == "specific_answer").mean()
                    ),
                    "u_vanilla_nonanswer": float(
                        (u.vanilla_label == "nonanswer").mean()
                    ),
                    "u_system_nonanswer": float((u.final_label == "nonanswer").mean()),
                    "u_triggered_n": int(len(ut)),
                    "u_triggered_post_abstain": safe_rate(
                        int((ut.intervention_label == "abstain").sum()),
                        len(ut),
                    ),
                    "u_triggered_post_hallucinated": safe_rate(
                        int((ut.intervention_label == "specific_answer").sum()),
                        len(ut),
                    ),
                    "u_triggered_post_nonanswer": safe_rate(
                        int((ut.intervention_label == "nonanswer").sum()),
                        len(ut),
                    ),
                    "u_hallucinated_to_abstain": safe_rate(
                        int((ut_specific.intervention_label == "abstain").sum()),
                        len(ut_specific),
                    ),
                    "u_abstain_to_abstain": safe_rate(
                        int((ut_abstain.intervention_label == "abstain").sum()),
                        len(ut_abstain),
                    ),
                    "a_triggered_n": int(len(at)),
                    "a_triggered_correct_n": int(len(at_correct)),
                    "a_correct_to_correct": safe_rate(
                        int((at_correct.intervention_label == "correct").sum()),
                        len(at_correct),
                    ),
                    "a_correct_to_abstain": safe_rate(
                        int((at_correct.intervention_label == "abstain").sum()),
                        len(at_correct),
                    ),
                    "a_correct_to_incorrect": safe_rate(
                        int((at_correct.intervention_label == "incorrect").sum()),
                        len(at_correct),
                    ),
                    "a_correct_to_nonanswer": safe_rate(
                        int((at_correct.intervention_label == "nonanswer").sum()),
                        len(at_correct),
                    ),
                    "a_brake_checkpoint_mean": float(at.brake_checkpoint.mean()),
                    "a_brake_fraction_mean": float(at.brake_fraction_of_vanilla.mean()),
                    "a_brake_sentence_mean": float(at.brake_sentence_index.mean()),
                    "u_brake_checkpoint_mean": float(ut.brake_checkpoint.mean()),
                    "u_brake_fraction_mean": float(ut.brake_fraction_of_vanilla.mean()),
                    "u_brake_sentence_mean": float(ut.brake_sentence_index.mean()),
                    "u_triggered_token_reduction_mean": float(
                        ut.token_reduction_fraction.mean()
                    ),
                    "u_all_token_reduction_mean": float(
                        u.token_reduction_fraction.mean()
                    ),
                    "a_all_token_reduction_mean": float(
                        a.token_reduction_fraction.mean()
                    ),
                    "a_intervention_token_limit_rate": safe_rate(
                        int(at.intervention_hit_token_limit.sum()),
                        len(at),
                    ),
                    "u_intervention_token_limit_rate": safe_rate(
                        int(ut.intervention_hit_token_limit.sum()),
                        len(ut),
                    ),
                }
            )

    return pd.DataFrame(rows)


def summarize_models(
    seed_metrics: pd.DataFrame,
) -> pd.DataFrame:
    numeric = [c for c in seed_metrics.columns if c not in {"model", "seed"}]

    rows = []

    for model in MODELS:
        d = seed_metrics[seed_metrics.model == model]

        row = {"model": model}

        for col in numeric:
            vals = d[col].to_numpy(dtype=float)
            vals = vals[~np.isnan(vals)]

            row[f"{col}_mean"] = float(vals.mean()) if len(vals) else np.nan

            row[f"{col}_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def compute_transitions(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for model in MODELS:
        for seed in SEEDS:
            for condition in (
                "answerable",
                "unanswerable",
            ):
                d = df[
                    (df.model == model)
                    & (df.seed == seed)
                    & (df.condition == condition)
                    & (df.triggered)
                ]

                grouped = (
                    d.groupby(
                        [
                            "vanilla_label",
                            "intervention_label",
                        ],
                        dropna=False,
                    )
                    .size()
                    .reset_index(name="count")
                )

                for _, r in grouped.iterrows():
                    rows.append(
                        {
                            "model": model,
                            "seed": seed,
                            "condition": condition,
                            "vanilla_label": r["vanilla_label"],
                            "intervention_label": r["intervention_label"],
                            "count": int(r["count"]),
                            "triggered_n": int(len(d)),
                            "proportion_all_triggered": safe_rate(
                                int(r["count"]),
                                len(d),
                            ),
                        }
                    )

    return pd.DataFrame(rows)


def compute_group_metrics(
    df: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    rows = []

    grouped = df.groupby(
        group_cols,
        dropna=False,
        sort=True,
    )

    for keys, d in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)

        a = d[d.condition == "answerable"]
        u = d[d.condition == "unanswerable"]

        if len(a) == 0 or len(u) == 0:
            continue

        row = dict(zip(group_cols, keys))

        row.update(
            {
                "n_a": int(len(a)),
                "n_u": int(len(u)),
                "a_false_rate": float(a.triggered.mean()),
                "u_brake_rate": float(u.triggered.mean()),
                "a_vanilla_accuracy": float((a.vanilla_label == "correct").mean()),
                "a_system_accuracy": float((a.final_label == "correct").mean()),
                "u_vanilla_abstain": float((u.vanilla_label == "abstain").mean()),
                "u_system_abstain": float((u.final_label == "abstain").mean()),
                "u_vanilla_hallucinated": float(
                    (u.vanilla_label == "specific_answer").mean()
                ),
                "u_system_hallucinated": float(
                    (u.final_label == "specific_answer").mean()
                ),
            }
        )

        rows.append(row)

    return pd.DataFrame(rows)


def compute_source_metrics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    return compute_group_metrics(
        df,
        ["model", "seed", "source"],
    )


def compute_source_category_metrics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    return compute_group_metrics(
        df,
        ["model", "seed", "source", "category"],
    )


def compute_family_metrics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    return compute_group_metrics(
        df,
        ["model", "seed", "family"],
    )


def compute_judge_diagnostics(
    vanilla_rows: list[dict],
    intervention_rows: list[dict],
) -> pd.DataFrame:
    rows = []

    for model in MODELS:
        v = [r for r in vanilla_rows if r["model"] == model]
        i = [r for r in intervention_rows if r["model"] == model]

        rows.append(
            {
                "model": model,
                "vanilla_n": len(v),
                "vanilla_token_limit_n": sum(
                    bool(
                        r.get(
                            "hit_token_limit",
                            False,
                        )
                    )
                    for r in v
                ),
                "vanilla_token_limit_rate": safe_rate(
                    sum(
                        bool(
                            r.get(
                                "hit_token_limit",
                                False,
                            )
                        )
                        for r in v
                    ),
                    len(v),
                ),
                "intervention_n": len(i),
                "intervention_token_limit_n": sum(
                    bool(
                        r.get(
                            "hit_token_limit",
                            False,
                        )
                    )
                    for r in i
                ),
                "intervention_token_limit_rate": safe_rate(
                    sum(
                        bool(
                            r.get(
                                "hit_token_limit",
                                False,
                            )
                        )
                        for r in i
                    ),
                    len(i),
                ),
            }
        )

    return pd.DataFrame(rows)


def print_summary(
    seed_metrics: pd.DataFrame,
) -> None:
    print("\n" + "=" * 84)
    print("V9_FINAL VALIDATION BEHAVIORAL SUMMARY " "— SENTENCE / CALIBRATED TAU5")
    print("=" * 84)
    print("± values below are SD across gate seeds, " "not sampling uncertainty.")

    for model in MODELS:
        d = seed_metrics[seed_metrics.model == model]

        def pct(col):
            m, s = ms(d[col])
            return m * 100, s * 100

        a_false = pct("a_false_rate")
        u_brake = pct("u_brake_rate")

        a_van = pct("a_vanilla_accuracy")
        a_sys = pct("a_system_accuracy")

        u_van = pct("u_vanilla_abstain")
        u_sys = pct("u_system_abstain")

        h_van = pct("u_vanilla_hallucinated")
        h_sys = pct("u_system_hallucinated")

        u_post = pct("u_triggered_post_abstain")
        u_recovery = pct("u_hallucinated_to_abstain")
        a_retain = pct("a_correct_to_correct")

        brake_frac = pct("u_brake_fraction_mean")

        print(f"\n{MODEL_LABELS[model]}")

        print(
            f"  Gate: "
            f"A-false={a_false[0]:.1f}±{a_false[1]:.1f}%  "
            f"U-brake={u_brake[0]:.1f}±{u_brake[1]:.1f}%"
        )

        print(
            f"  A accuracy: "
            f"vanilla={a_van[0]:.1f}%  "
            f"self-braking={a_sys[0]:.1f}±{a_sys[1]:.1f}%"
        )

        print(
            f"  U correct abstention: "
            f"vanilla={u_van[0]:.1f}%  "
            f"self-braking={u_sys[0]:.1f}±{u_sys[1]:.1f}%"
        )

        print(
            f"  U specific answer: "
            f"vanilla={h_van[0]:.1f}%  "
            f"self-braking={h_sys[0]:.1f}±{h_sys[1]:.1f}%"
        )

        print(f"  Triggered U -> abstention: " f"{u_post[0]:.1f}±{u_post[1]:.1f}%")

        print(
            f"  Triggered specific-answer U -> abstention: "
            f"{u_recovery[0]:.1f}±{u_recovery[1]:.1f}%"
        )

        print(
            f"  False-triggered previously-correct A retained: "
            f"{a_retain[0]:.1f}±{a_retain[1]:.1f}%"
        )

        print(
            f"  U brake position: "
            f"{brake_frac[0]:.1f}±{brake_frac[1]:.1f}% "
            f"of vanilla trajectory"
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="umwp",
        choices=("umwp", "worldsense"),
    )
    parser.add_argument(
        "--split",
        default="val",
    )
    parser.add_argument(
        "--rubric",
        default="v2",
    )

    args = parser.parse_args()

    config = get_dataset_config(args.dataset)

    output_root = config.analysis_root / args.split
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    (output_root / "generated_output_token_metric.json").write_text(
        json.dumps(GENERATED_OUTPUT_TOKEN_METRIC, indent=2) + "\n"
    )

    (
        vanilla_rows,
        intervention_rows,
        vanilla_map,
        intervention_map,
    ) = load_judgments(
        config=config,
        split=args.split,
        rubric=args.rubric,
    )

    print(f"Dataset: {config.name}")
    print(f"Vanilla judgments: " f"{len(vanilla_rows)}")
    print(f"Intervention judgments: " f"{len(intervention_rows)}")

    metadata_map = load_trial_metadata(
        config=config,
        split=args.split,
    )

    df = build_trial_table(
        config=config,
        split=args.split,
        vanilla_map=vanilla_map,
        intervention_map=intervention_map,
        metadata_map=metadata_map,
    )

    print(f"Joined system rows: {len(df)}")
    print(f"Triggered rows: " f"{int(df.triggered.sum())}")

    seed_metrics = compute_seed_metrics(df)
    model_summary = summarize_models(seed_metrics)
    transitions = compute_transitions(df)
    source_metrics = compute_source_metrics(df)
    source_category_metrics = compute_source_category_metrics(df)
    family_metrics = compute_family_metrics(df)

    judge_diagnostics = compute_judge_diagnostics(
        vanilla_rows,
        intervention_rows,
    )

    df.to_csv(
        output_root / "trial_level.csv",
        index=False,
    )
    seed_metrics.to_csv(
        output_root / "seed_metrics.csv",
        index=False,
    )
    model_summary.to_csv(
        output_root / "model_summary.csv",
        index=False,
    )
    transitions.to_csv(
        output_root / "transitions.csv",
        index=False,
    )
    source_metrics.to_csv(
        output_root / "source_metrics.csv",
        index=False,
    )
    source_category_metrics.to_csv(
        output_root / "source_category_metrics.csv",
        index=False,
    )
    family_metrics.to_csv(
        output_root / "family_metrics.csv",
        index=False,
    )
    judge_diagnostics.to_csv(
        output_root / "judge_diagnostics.csv",
        index=False,
    )

    print_summary(seed_metrics)

    print(f"\nSaved analysis to: " f"{output_root}")


if __name__ == "__main__":
    main()
