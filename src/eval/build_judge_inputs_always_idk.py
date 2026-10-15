import argparse
import json
from pathlib import Path
from src.gate.datasets import get_dataset_config, validate_trajectory_rows
from src.capture.run_umwp_always_idk import TASK_PROMPT as EXPECTED_PROMPT
from src.capture.run_worldsense_always_idk import TASK_PROMPT as WORLDSENSE_PROMPT

MODELS = ("af_next", "audio-reasoner", "moss")
EXPECTED_TRIALS = 780
EXPECTED_A = 390
EXPECTED_U = 390
TRAJECTORY_ROOT = Path("data/ablations/always_idk/umwp/trajectories")
JUDGE_ROOT = Path("data/ablations/always_idk/umwp/judge")


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def possible_answers(row):
    for key in ("possible_answers", "answer_options", "candidate_answers", "options"):
        value = row.get(key)
        if isinstance(value, (list, tuple)) and value:
            return list(value)
    return []


def validate_rows(rows, path, model, split, dataset="umwp"):
    config = get_dataset_config(dataset)
    expected_a, expected_u, expected_trials = config.expected_counts(split)
    validate_trajectory_rows(rows, config, split, model)
    if len(rows) != expected_trials:
        raise RuntimeError(f"{path}: expected {expected_trials} rows, got {len(rows)}")
    ids = [str(r["trial_id"]) for r in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"{path}: duplicate trial_id")
    n_a = sum(r.get("condition") == "answerable" for r in rows)
    n_u = sum(r.get("condition") == "unanswerable" for r in rows)
    if (n_a, n_u) != (expected_a, expected_u):
        raise RuntimeError(
            f"{path}: expected {expected_a} A + {expected_u} U, got {n_a} A + {n_u} U"
        )
    trials = {}
    if dataset == "worldsense":
        trials = {
            r["trial_id"]: r
            for r in read_jsonl(
                Path("data/processed/worldsense_trials_openended.jsonl")
            )
        }
    for row in rows:
        if row.get("model_name") != model:
            raise RuntimeError(
                f"{row.get('trial_id')}: model_name={row.get('model_name')!r}, expected {model!r}"
            )
        if row.get("split") != split:
            raise RuntimeError(
                f"{row.get('trial_id')}: split={row.get('split')!r}, expected {split!r}"
            )
        expected_prompt = (
            WORLDSENSE_PROMPT.format(question=trials[row["trial_id"]]["question"])
            if dataset == "worldsense"
            else EXPECTED_PROMPT
        )
        if row.get("prompt_text") != expected_prompt:
            raise RuntimeError(
                f"{row.get('trial_id')}: prompt_text does not match frozen always-IDK prompt"
            )
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise RuntimeError(f"{row.get('trial_id')}: missing response text")
        condition = row.get("condition")
        gold = row.get("gold_answer")
        if condition == "answerable":
            if gold is None or (isinstance(gold, (list, tuple)) and not gold):
                raise RuntimeError(
                    f"{row.get('trial_id')}: answerable row missing gold_answer"
                )
        elif condition == "unanswerable":
            if gold not in (None, [], ()):
                raise RuntimeError(
                    f"{row.get('trial_id')}: unanswerable row unexpectedly has gold_answer={gold!r}"
                )
        else:
            raise RuntimeError(
                f"{row.get('trial_id')}: unknown condition={condition!r}"
            )


def build_item(model, row, dataset="umwp"):
    item = {
        "dataset": dataset,
        "trial_id": str(row["trial_id"]),
        "model": model,
        "condition": row["condition"],
        "answerable_answer": (
            row.get("gold_answer") if row["condition"] == "answerable" else None
        ),
        "possible_answers": possible_answers(row),
        "response_text": row["text"],
        "hit_token_limit": bool(row.get("stop_reason") == "max_new_tokens"),
        "ablation": "always_idk",
        "input_prompt_text": row["prompt_text"],
    }
    for key in (
        "pair_id",
        "source",
        "category",
        "family",
        "question_type",
        "n_entities",
    ):
        if row.get(key) is not None:
            item[key] = row[key]
    return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="umwp", choices=("umwp", "worldsense"))
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args()

    if args.split == "test" and not args.allow_test:
        raise RuntimeError("Refusing to touch test split without --allow-test")
    root = Path("data/ablations/always_idk") / args.dataset
    path = root / "trajectories" / args.model / args.split / "metadata.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    rows = read_jsonl(path)
    validate_rows(rows, path, args.model, args.split, args.dataset)
    items = [build_item(args.model, row, args.dataset) for row in rows]
    output = root / "judge" / f"{args.split}_{args.model}.jsonl"
    write_jsonl(items, output)
    print(f"model: {args.model}")
    print(f"rows: {len(items)}")
    print(f"output: {output}")
    print("prompt provenance: always_idk")


if __name__ == "__main__":
    main()
