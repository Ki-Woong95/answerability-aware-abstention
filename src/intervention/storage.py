"""Validate intervention resume files before reusing or retrying any result."""

import json
from pathlib import Path

from src.gate.datasets import validate_artifact_dataset


def load_existing(
    path,
    retry_errors,
    planned=None,
    dataset_config=None,
    model_name=None,
    split=None,
    seed=None,
):
    if not path.exists():
        return {}, {}
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected = (
        {row["trial_id"]: row for row in planned} if planned is not None else None
    )
    done, cache = {}, {}
    for row in rows:
        trial_id = str(row["trial_id"])
        if trial_id in done:
            raise RuntimeError(f"{path}: duplicate intervention trial_id {trial_id}")
        done[trial_id] = row
        if expected is not None:
            if trial_id not in expected:
                raise RuntimeError(f"{path}: unknown saved trial_id {trial_id}")
            validate_artifact_dataset(row, dataset_config, path, model_name=model_name)
            if (row.get("model_name"), row.get("split"), row.get("seed")) != (
                model_name,
                split,
                seed,
            ):
                raise RuntimeError(
                    f"{path}: saved model/split/seed mismatch for {trial_id}"
                )
            for field, value in expected[trial_id].items():
                if row.get(field) != value:
                    raise RuntimeError(
                        f"{path}: saved {field} does not match current intervention plan for {trial_id}; "
                        "use a fresh intervention output directory"
                    )
    # Verify all records before removing retryable errors. Failed validation is read-only.
    if retry_errors:
        rows = [row for row in rows if not row.get("intervention_error")]
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        temporary.replace(path)
        done = {str(row["trial_id"]): row for row in rows}
    for row in rows:
        if (
            row.get("triggered")
            and row.get("intervention_result")
            and not row.get("intervention_error")
        ):
            cache[(str(row["trial_id"]), int(row["brake_checkpoint"]))] = row[
                "intervention_result"
            ]
    return done, cache
