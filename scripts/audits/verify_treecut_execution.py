"""Check execution fingerprints and complete TreeCut captures/gates."""

import argparse, hashlib, json
from pathlib import Path
from src.data.frozen_treecut import ROOT, verify_dataset
from src.gate.datasets import (
    get_dataset_config,
    validate_trajectory_rows,
    validate_artifact_dataset,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-directory", type=Path, required=True)
    p.add_argument("--model", choices=["af_next", "moss", "audio-reasoner"])
    p.add_argument("--complete", action="store_true")
    args = p.parse_args()
    record = json.loads((args.run_directory / "execution_plan.json").read_text())
    for name, expected in record["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("TreeCut source changed: " + name)
    dataset = verify_dataset()
    cfg = get_dataset_config("treecut")
    if args.complete:
        if not args.model:
            raise ValueError("Complete audit needs model")
        for split in ["gate_train", "calibration", "val"]:
            path = cfg.trajectory_root / args.model / split / "metadata.jsonl"
            with path.open() as f:
                rows = [json.loads(line) for line in f if line.strip()]
            validate_trajectory_rows(rows, cfg, split, args.model)
            print(f"{args.model} {split}: {len(rows)} complete trajectories")
        summary_path = cfg.gate_root / args.model / "all_layers/train_summary.json"
        summary = json.loads(summary_path.read_text())
        validate_artifact_dataset(summary, cfg, summary_path, args.model)
        if summary["epochs"] != 20 or not summary["training_complete"]:
            raise RuntimeError("Incomplete gate fitting")
        for layer in range(1, summary["n_layers"] + 1):
            for seed in [17, 29, 43]:
                path = (
                    cfg.gate_root
                    / args.model
                    / "all_layers"
                    / f"layer_{layer:02d}"
                    / f"seed_{seed}.pt"
                )
                if not path.is_file():
                    raise RuntimeError("Missing gate " + str(path))
        print("State-fitting coverage:", json.dumps(summary["state_fitting_coverage"]))
    print(
        "Frozen TreeCut inputs and execution sources verified; test inference not requested"
    )


if __name__ == "__main__":
    main()
