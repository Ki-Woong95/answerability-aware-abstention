import argparse
import json
from pathlib import Path

from openai import OpenAI
from src.eval.run_judge import (
    DEFAULT_JUDGE_MODEL,
    REASONING_EFFORT,
    RUBRIC_VERSION,
    judge_one,
    make_key,
    load_completed,
    judge_input_fingerprint,
)
from src.gate.datasets import (
    get_dataset_config,
    validate_artifact_dataset,
    validate_trajectory_rows,
)

MODELS = ("af_next", "audio-reasoner", "moss")
EXPECTED_TRIALS = 780
JUDGE_ROOT = Path("data/ablations/always_idk/umwp/judge")


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def validate_input(rows, path, model, dataset="umwp", split="val"):
    config = get_dataset_config(dataset)
    _, _, expected_trials = config.expected_counts(split)
    validate_trajectory_rows(
        [{**row, "model_name": row.get("model"), "split": split} for row in rows],
        config,
        split,
        model,
    )
    if len(rows) != expected_trials:
        raise RuntimeError(f"{path}: expected {expected_trials} rows, got {len(rows)}")
    ids = [str(r["trial_id"]) for r in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"{path}: duplicate trial_id")
    for row in rows:
        validate_artifact_dataset(row, config, path)
        if row.get("model") != model:
            raise RuntimeError(
                f"{row.get('trial_id')}: model={row.get('model')!r}, expected {model!r}"
            )
        if row.get("ablation") != "always_idk":
            raise RuntimeError(f"{row.get('trial_id')}: missing always_idk provenance")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="umwp", choices=("umwp", "worldsense"))
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument(
        "--output", type=Path, help="Optional fresh judgment output file"
    )
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test:
        raise RuntimeError("Refusing to touch test split without --allow-test")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")

    judge_root = Path("data/ablations/always_idk") / args.dataset / "judge"
    input_path = judge_root / f"{args.split}_{args.model}.jsonl"
    output_path = (
        args.output
        or judge_root / f"{args.split}_{args.model}_judged_{RUBRIC_VERSION}.jsonl"
    )
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    rows = read_jsonl(input_path)
    validate_input(rows, input_path, args.model, args.dataset, args.split)

    completed = load_completed(output_path, rows, args.judge_model)
    client = OpenAI()
    processed = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as fout:
        for row in rows:
            key = make_key(row)
            if key in completed:
                continue
            result, response_id = judge_one(
                client=client, row=row, judge_model=args.judge_model
            )
            output = {
                "dataset": args.dataset,
                "trial_id": row["trial_id"],
                "model": row["model"],
                "condition": row["condition"],
                "ablation": "always_idk",
            }
            for field in (
                "pair_id",
                "source",
                "category",
                "family",
                "question_type",
                "n_entities",
            ):
                if field in row:
                    output[field] = row.get(field)
            output.update(
                {
                    "label": result["label"],
                    "extracted_answer": result["extracted_answer"],
                    "note": result["note"],
                    "hit_token_limit": bool(row.get("hit_token_limit", False)),
                    "judge_model": args.judge_model,
                    "reasoning_effort": REASONING_EFFORT,
                    "rubric_version": RUBRIC_VERSION,
                    "judge_response_id": response_id,
                    "input_fingerprint": judge_input_fingerprint(row, args.judge_model),
                }
            )
            fout.write(json.dumps(output, ensure_ascii=False) + "\n")
            fout.flush()
            completed[key] = output
            processed += 1
            print(
                f"{processed:04d} {row['model']} {row['trial_id']} -> {result['label']}"
            )
            if args.limit is not None and processed >= args.limit:
                break
    print(f"\nProcessed this run: {processed}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
