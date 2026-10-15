"""Read-only checks for authorized test execution with fixed artifacts."""

import argparse
import hashlib
import json
from pathlib import Path

from src.gate.datasets import (
    get_dataset_config,
    load_split_ids,
    validate_artifact_dataset,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--model", choices=["af_next", "moss", "audio-reasoner"])
    parser.add_argument("--complete", action="store_true")
    args = parser.parse_args()
    record = json.loads(args.record.read_text())
    for path, expected in record["protected_sha256"].items():
        assert (
            hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
        ), f"Frozen artifact/source changed: {path}"
    if args.complete:
        assert args.model
        cfg = get_dataset_config("umwp")
        expected = {
            (pid, c)
            for pid in load_split_ids(cfg, "test")
            for c in ["answerable", "unanswerable"]
        }
        for seed in [17, 29, 43]:
            path = cfg.intervention_root / args.model / "test" / f"seed_{seed}.jsonl"
            rows = [
                json.loads(line)
                for line in path.read_text().splitlines()
                if line.strip()
            ]
            identities = [(r["pair_id"], r["condition"]) for r in rows]
            assert (
                len(identities) == len(expected) and set(identities) == expected
            ), f"{path}: incomplete/duplicate results"
            for r in rows:
                validate_artifact_dataset(r, cfg, path, model_name=args.model)
                assert (
                    r.get("model_name") == args.model
                    and r.get("split") == "test"
                    and r.get("seed") == seed
                )
                assert r.get("test_read") is True and not r.get(
                    "intervention_error"
                ), f"{path}: failed intervention"
                assert not r["triggered"] or (
                    r.get("intervention_run")
                    and r.get("intervention_result") is not None
                )
            print(
                f"{args.model} seed={seed}: {len(rows)} complete test results, no intervention errors"
            )
    print("Frozen artifacts and execution sources unchanged")


if __name__ == "__main__":
    main()
