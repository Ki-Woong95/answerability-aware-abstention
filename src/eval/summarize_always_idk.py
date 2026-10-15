import argparse
import json
from collections import Counter
from pathlib import Path
from src.gate.datasets import get_dataset_config, validate_artifact_dataset

MODELS = ("af_next", "audio-reasoner", "moss")
EXPECTED_TRIALS = 780
EXPECTED_A = 390
EXPECTED_U = 390
JUDGE_ROOT = Path("data/ablations/always_idk/umwp/judge")


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pct(n, d):
    return 100.0 * n / d if d else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="umwp", choices=("umwp", "worldsense"))
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--rubric", default="v2")
    args = parser.parse_args()

    config = get_dataset_config(args.dataset)
    expected_a, expected_u, expected_trials = config.expected_counts(args.split)
    judge_root = Path("data/ablations/always_idk") / args.dataset / "judge"
    path = judge_root / f"{args.split}_{args.model}_judged_{args.rubric}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    rows = read_jsonl(path)
    for row in rows:
        validate_artifact_dataset(row, config, path)
        if row.get("model") != args.model or row.get("ablation") != "always_idk":
            raise RuntimeError(f"{path}: model/ablation mismatch")
    ids = [str(r["trial_id"]) for r in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"{path}: duplicate trial_id")
    a = [r for r in rows if r["condition"] == "answerable"]
    u = [r for r in rows if r["condition"] == "unanswerable"]
    ac = Counter(r["label"] for r in a)
    uc = Counter(r["label"] for r in u)

    print(f"Always-IDK | {args.model} | {args.split} | rubric={args.rubric}")
    print(f"judged: {len(rows)}/{expected_trials}")
    print("\nAnswerable")
    print(
        f"  correct:   {ac['correct']:3d}/{len(a):3d} = {pct(ac['correct'], len(a)):6.2f}%"
    )
    print(
        f"  incorrect: {ac['incorrect']:3d}/{len(a):3d} = {pct(ac['incorrect'], len(a)):6.2f}%"
    )
    print(
        f"  abstain:   {ac['abstain']:3d}/{len(a):3d} = {pct(ac['abstain'], len(a)):6.2f}%"
    )
    print(
        f"  nonanswer: {ac['nonanswer']:3d}/{len(a):3d} = {pct(ac['nonanswer'], len(a)):6.2f}%"
    )
    print("\nUnanswerable")
    print(
        f"  abstain:         {uc['abstain']:3d}/{len(u):3d} = {pct(uc['abstain'], len(u)):6.2f}%"
    )
    print(
        f"  specific_answer: {uc['specific_answer']:3d}/{len(u):3d} = {pct(uc['specific_answer'], len(u)):6.2f}%"
    )
    print(
        f"  nonanswer:       {uc['nonanswer']:3d}/{len(u):3d} = {pct(uc['nonanswer'], len(u)):6.2f}%"
    )

    if len(rows) == expected_trials:
        if len(a) != expected_a or len(u) != expected_u:
            raise RuntimeError(
                f"expected {expected_a} A + {expected_u} U, got {len(a)} A + {len(u)} U"
            )
        print("\ncomplete: PASS")
    else:
        print("\ncomplete: partial judgment file")


if __name__ == "__main__":
    main()
