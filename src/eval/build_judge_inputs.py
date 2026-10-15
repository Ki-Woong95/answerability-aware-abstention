import argparse
import json
from pathlib import Path

from src.gate.datasets import (
    get_dataset_config,
    validate_artifact_dataset,
    validate_trajectory_rows,
)

MODELS = ("af_next", "audio-reasoner", "moss")
SEEDS = (17, 29, 43)


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def gold_answer(row):
    for key in (
        "answerable_answer",
        "gold_answer",
        "reference_answer",
        "target_answer",
        "answer",
    ):
        value = row.get(key)
        if value is not None and str(value).strip():
            return value
    raise KeyError(
        f"{row.get('trial_id')}: no gold answer field in trajectory metadata"
    )


def possible_answers(row):
    for key in (
        "possible_answers",
        "answer_options",
        "candidate_answers",
        "options",
    ):
        value = row.get(key)
        if isinstance(value, (list, tuple)) and value:
            return list(value)
    return []


def metadata_map(model, split, dataset_config):
    _, _, expected_trials = dataset_config.expected_counts(split)
    path = dataset_config.trajectory_root / model / split / "metadata.jsonl"
    rows = read_jsonl(path)
    validate_trajectory_rows(rows, dataset_config, split, model)
    if len(rows) != expected_trials:
        raise RuntimeError(f"{path}: expected {expected_trials} rows, got {len(rows)}")
    out = {str(r["trial_id"]): r for r in rows}
    if len(out) != len(rows):
        raise RuntimeError(f"duplicate trial_id in {path}")
    return out


def common_fields(model, row, meta, dataset_config):
    from src.data.frozen_umwp import v10_artifact_provenance

    item = {
        **v10_artifact_provenance(dataset_config.name, model),
        "dataset": dataset_config.name,
        "trial_id": str(row["trial_id"]),
        "model": model,
        "condition": row["condition"],
        "answerable_answer": (
            gold_answer(meta) if row["condition"] == "answerable" else None
        ),
        "possible_answers": possible_answers(meta),
    }

    # Important for WorldSense: the gold may be an option index (1/2), while
    # a model can answer with the semantic option text. Supplying the original
    # prompt lets the judge map semantic answers back to the numbered option.
    prompt_text = meta.get("prompt_text")
    if isinstance(prompt_text, str) and prompt_text.strip():
        item["input_prompt_text"] = prompt_text

    for key in (
        "pair_id",
        "family",
        "question_type",
        "n_entities",
        "source",
        "category",
    ):
        value = row.get(key, meta.get(key))
        if value is not None:
            item[key] = value
    return item


def validate_intervention_row_dataset(row, path, dataset_config):
    validate_artifact_dataset(
        row, dataset_config, path, model_name=row.get("model_name")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="umwp", choices=("umwp", "worldsense"))
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args()

    if args.split == "test" and not args.allow_test:
        raise RuntimeError("Refusing to touch test split without --allow-test")

    dataset_config = get_dataset_config(args.dataset)
    expected_a, expected_u, expected_trials = dataset_config.expected_counts(args.split)

    vanilla, interventions = {}, {}
    source_rows = triggered_rows = 0

    for model in MODELS:
        meta_by_id = metadata_map(model, args.split, dataset_config)

        for seed in SEEDS:
            path = (
                dataset_config.intervention_root
                / model
                / args.split
                / f"seed_{seed}.jsonl"
            )
            rows = read_jsonl(path)
            if len(rows) != expected_trials:
                raise RuntimeError(
                    f"{path}: expected {expected_trials} rows, got {len(rows)}"
                )
            if len({str(r["trial_id"]) for r in rows}) != len(rows):
                raise RuntimeError(f"duplicate trial_id in {path}")

            n_a = sum(r.get("condition") == "answerable" for r in rows)
            n_u = sum(r.get("condition") == "unanswerable" for r in rows)
            if (n_a, n_u) != (expected_a, expected_u):
                raise RuntimeError(
                    f"{path}: expected A/U={expected_a}/{expected_u}, "
                    f"got {n_a}/{n_u}"
                )

            for row in rows:
                validate_intervention_row_dataset(row, path, dataset_config)
                source_rows += 1

                if (
                    row.get("model_name") != model
                    or int(row.get("seed")) != seed
                    or row.get("split") != args.split
                ):
                    raise RuntimeError(
                        f"metadata mismatch in {path}: {row.get('trial_id')}"
                    )

                trial_id = str(row["trial_id"])
                if trial_id not in meta_by_id:
                    raise RuntimeError(
                        f"{path}: intervention trial missing from trajectory metadata: "
                        f"{trial_id}"
                    )

                meta = meta_by_id[trial_id]
                if (
                    row["condition"] != meta["condition"]
                    or row.get("pair_id") != meta["pair_id"]
                ):
                    raise RuntimeError(
                        f"{path}: intervention condition/pair differs from trajectory: {trial_id}"
                    )
                if (
                    row.get("vanilla_text") != meta["text"]
                    or row.get("vanilla_num_generated_tokens")
                    != meta["num_generated_tokens"]
                ):
                    raise RuntimeError(f"{path}: stale vanilla output: {trial_id}")
                base = common_fields(model, row, meta, dataset_config)

                vanilla_item = dict(base)
                vanilla_item.update(
                    {
                        "response_text": row.get("vanilla_text")
                        or meta.get("text")
                        or "",
                        "hit_token_limit": bool(
                            meta.get("stop_reason") == "max_new_tokens"
                        ),
                    }
                )

                key = (model, trial_id)
                if key in vanilla and vanilla[key] != vanilla_item:
                    raise RuntimeError(f"vanilla mismatch across seeds: {key}")
                vanilla[key] = vanilla_item

                if not row.get("triggered"):
                    continue

                triggered_rows += 1
                if not row.get("intervention_run") or row.get("intervention_error"):
                    raise RuntimeError(
                        f"triggered intervention failed: {model} seed={seed} "
                        f"{trial_id}"
                    )

                result = row.get("intervention_result") or {}
                text = result.get("continuation_text")
                n = int(result.get("num_continuation_tokens", 0) or 0)
                budget = int(result.get("continuation_budget", 2048) or 2048)

                if not isinstance(text, str):
                    raise RuntimeError(
                        f"missing continuation_text: {model} seed={seed} {trial_id}"
                    )

                if not text.strip() and n < budget:
                    raise RuntimeError(
                        f"empty continuation before token limit: "
                        f"{model} seed={seed} {trial_id} n={n} budget={budget}"
                    )

                item = dict(base)
                item.update(
                    {
                        "seed": seed,
                        "response_text": text,
                        "hit_token_limit": (
                            result["stop_reason"] == "max_new_tokens"
                            if "stop_reason" in result
                            else bool(n >= budget)
                        ),
                    }
                )
                interventions[(model, seed, trial_id)] = item

    expected_source = len(MODELS) * len(SEEDS) * expected_trials
    if source_rows != expected_source:
        raise RuntimeError(f"expected {expected_source} source rows, got {source_rows}")

    expected_vanilla = len(MODELS) * expected_trials
    if len(vanilla) != expected_vanilla:
        raise RuntimeError(
            f"expected {expected_vanilla} unique vanilla rows, got {len(vanilla)}"
        )

    dataset_config.judge_root.mkdir(parents=True, exist_ok=True)
    vanilla_path = dataset_config.judge_root / f"{args.split}_vanilla.jsonl"
    intervention_path = dataset_config.judge_root / f"{args.split}_intervention.jsonl"

    write_jsonl([vanilla[k] for k in sorted(vanilla)], vanilla_path)
    write_jsonl(
        [interventions[k] for k in sorted(interventions)],
        intervention_path,
    )

    print(f"dataset: {dataset_config.name}")
    print(f"source rows: {source_rows}")
    print(f"vanilla: {len(vanilla)} -> {vanilla_path}")
    print(
        f"triggered intervention: {len(interventions)} "
        f"(source triggered={triggered_rows}) -> {intervention_path}"
    )
    print("Non-triggered system outcomes will reuse vanilla judge labels downstream.")


if __name__ == "__main__":
    main()
