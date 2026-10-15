"""Read-only validation timing analysis; does not read test trajectories."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    x = (len(values) - 1) * q
    lo = int(x)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (x - lo)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test:
        parser.error("Test analysis requires --allow-test")
    folder = ROOT / "data/analysis" / args.split
    with (folder / "trial_level.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    metadata = {}
    for model in sorted({r["model"] for r in rows}):
        path = ROOT / "data/trajectories" / model / args.split / "metadata.jsonl"
        with path.open() as stream:
            for line in stream:
                record = json.loads(line)
                metadata[model, record["trial_id"]] = record
    details = []
    for row in rows:
        if row["triggered"] != "True":
            continue
        record = metadata[row["model"], row["trial_id"]]
        checkpoint = int(float(row["brake_checkpoint"]))
        checkpoints = record["checkpoints"]
        rank = checkpoints.index(checkpoint) + 1
        assert rank == int(float(row["brake_sentence_index"]))
        event = next(
            e
            for e in record["boundary_events"]
            if e["confirmation_token_index"] == checkpoint
        )
        delays = [b["confirmation_delay_tokens"] for b in event["boundaries"]]
        assert checkpoint <= record["num_generated_tokens"]
        details.append(
            {
                "model": row["model"],
                "seed": int(row["seed"]),
                "trial_id": row["trial_id"],
                "condition": row["condition"],
                "vanilla_label": row["vanilla_label"],
                "final_label": row["final_label"],
                "confirmation_token": checkpoint,
                "checkpoint_rank": rank,
                "eligible_checkpoint_count": len(checkpoints),
                "checkpoint_fraction": rank / len(checkpoints),
                "whole_output_fraction": float(row["brake_fraction_of_vanilla"]),
                "min_confirmation_delay_tokens": min(delays),
                "max_confirmation_delay_tokens": max(delays),
                "vanilla_hit_token_limit": row["vanilla_hit_token_limit"] == "True",
            }
        )
    summaries = []
    for model in sorted({r["model"] for r in rows}):
        for seed in sorted({int(r["seed"]) for r in rows}):
            for name, predicate in [
                ("all_U", lambda r: r["condition"] == "unanswerable"),
                (
                    "vanilla_unsupported_U",
                    lambda r: r["condition"] == "unanswerable"
                    and r["vanilla_label"] == "specific_answer",
                ),
                (
                    "vanilla_abstained_U",
                    lambda r: r["condition"] == "unanswerable"
                    and r["vanilla_label"] == "abstain",
                ),
                ("all_A", lambda r: r["condition"] == "answerable"),
                (
                    "vanilla_correct_A",
                    lambda r: r["condition"] == "answerable"
                    and r["vanilla_label"] == "correct",
                ),
            ]:
                population = [
                    r
                    for r in rows
                    if r["model"] == model and int(r["seed"]) == seed and predicate(r)
                ]
                triggered = [
                    r
                    for r in details
                    if r["model"] == model and r["seed"] == seed and predicate(r)
                ]
                n = len(triggered)
                summary = {
                    "model": model,
                    "seed": seed,
                    "subset": name,
                    "population_n": len(population),
                    "triggered_n": n,
                    "trigger_rate": n / len(population) if population else None,
                    "post_abstain_n": sum(
                        r["final_label"] == "abstain" for r in triggered
                    ),
                    "post_correct_n": sum(
                        r["final_label"] == "correct" for r in triggered
                    ),
                    "post_abstain_rate": (
                        mean(r["final_label"] == "abstain" for r in triggered)
                        if n
                        else None
                    ),
                    "first_checkpoint_rate": (
                        mean(r["checkpoint_rank"] == 1 for r in triggered)
                        if n
                        else None
                    ),
                    "before_25pct_output_rate": (
                        mean(r["whole_output_fraction"] <= 0.25 for r in triggered)
                        if n
                        else None
                    ),
                    "token_limit_n": sum(
                        r["vanilla_hit_token_limit"] for r in triggered
                    ),
                }
                for field in [
                    "confirmation_token",
                    "checkpoint_rank",
                    "whole_output_fraction",
                    "checkpoint_fraction",
                    "max_confirmation_delay_tokens",
                ]:
                    values = [r[field] for r in triggered]
                    summary[field + "_mean"] = mean(values) if values else None
                    for label, q in [("q25", 0.25), ("median", 0.5), ("q75", 0.75)]:
                        summary[field + "_" + label] = percentile(values, q)
                summaries.append(summary)
    for name, output_rows in [
        ("intervention_timing_trials.csv", details),
        ("intervention_timing_by_seed.csv", summaries),
    ]:
        with (folder / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
            writer.writeheader()
            writer.writerows(output_rows)
    result = {
        "scope": f"{args.split} only; analysis of saved results, no inference",
        "conventions": [
            "Timing summaries condition on intervention; non-triggered trials remain in coverage denominators.",
            "Token position is the saved boundary-confirmation readout/replay-prefix length.",
            "Whole-output fraction uses completed vanilla length, not reasoning-only length.",
            "Checkpoint fraction is rank / total eligible vanilla checkpoints; retrospective only.",
            "Three seeds reuse the same trials; they are not independent data samples.",
            "Means below average per-seed statistics equally. Mean of medians is not a pooled median.",
        ],
        "seed_average": [],
    }
    for model in sorted({r["model"] for r in summaries}):
        for subset in sorted({r["subset"] for r in summaries}):
            selected = [
                r for r in summaries if r["model"] == model and r["subset"] == subset
            ]
            aggregate = {"model": model, "subset": subset}
            for field in selected[0]:
                if field in ["model", "seed", "subset"]:
                    continue
                values = [r[field] for r in selected if r[field] is not None]
                aggregate[field] = mean(values) if values else None
            result["seed_average"].append(aggregate)
    (folder / "intervention_timing_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
