"""Explicit pair-level capture failures; never edit the frozen population."""

import json
from pathlib import Path

FAILURE_POLICY = "treecut-pair-generation-failure-v1"


def load_failures(directory, rows, model_name, provenance):
    path = Path(directory) / "generation_failures.jsonl"
    expected = {r["trial_id"]: r for r in rows}
    failures = {}
    if not path.exists():
        return failures
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        row = expected.get(record.get("trial_id"))
        if (
            row is None
            or record.get("pair_id") != row["pair_id"]
            or record.get("condition") != row["condition"]
            or record.get("split") != row["split"]
            or record.get("model_name") != model_name
            or record.get("failure_policy") != FAILURE_POLICY
            or any(record.get(k) != v for k, v in provenance.items())
        ):
            raise RuntimeError("Invalid/stale TreeCut generation failure record")
        if record["pair_id"] in failures:
            raise RuntimeError("Duplicate TreeCut pair failure record")
        failures[record["pair_id"]] = record
    return failures


def write_failure(directory, row, model_name, provenance, exc, model):
    import traceback
    from datetime import datetime, timezone

    decoder = getattr(model, "last_token_text_decoder", None)
    record = dict(
        provenance,
        failure_policy=FAILURE_POLICY,
        dataset="treecut",
        model_name=model_name,
        trial_id=row["trial_id"],
        pair_id=row["pair_id"],
        condition=row["condition"],
        split=row["split"],
        exception_type=type(exc).__name__,
        exception_message=str(exc),
        traceback=traceback.format_exc(),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        observed_generated_token_ids=list(getattr(decoder, "observed_token_ids", [])),
        observed_monitoring_prefix=getattr(decoder, "text", None),
    )
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "generation_failures.jsonl").open("a") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
    return record


def capture_population(rows, directory, model_name, provenance):
    """Return usable rows and auditable original-population failure counts."""
    failures = load_failures(directory, rows, model_name, provenance)
    active = [r for r in rows if r["pair_id"] not in failures]
    conditions = {}
    for condition in ("answerable", "unanswerable"):
        total = sum(r["condition"] == condition for r in rows)
        excluded = sum(
            r["condition"] == condition and r["pair_id"] in failures for r in rows
        )
        conditions[condition] = dict(
            total=total,
            excluded=excluded,
            exclusion_rate=excluded / total if total else 0,
        )
    report = dict(
        failure_policy=FAILURE_POLICY,
        frozen_trial_count=len(rows),
        excluded_pair_ids=sorted(failures),
        excluded_pairs=len(failures),
        conditions=conditions,
        behavioral_evaluation_policy="Retain frozen denominators; report capture failures separately, never treat them as abstention.",
    )
    return active, report


def canonical_population(config, split, model_name):
    from src.data.frozen_treecut import load_trials, artifact_provenance

    rows = [
        r
        for r in load_trials()
        if r["split"] == split
        and (split != "calibration" or r["condition"] == "answerable")
    ]
    return capture_population(
        rows,
        config.trajectory_root / model_name / split,
        model_name,
        artifact_provenance(model_name),
    )


def recoverable_generation_failure(exc):
    """Skip trial-local generation exceptions; resource/system failures halt."""
    import torch

    if isinstance(exc, (MemoryError, OSError, torch.cuda.OutOfMemoryError)):
        return False
    message = str(exc).lower()
    if any(
        s in message
        for s in (
            "cuda",
            "device-side assert",
            "illegal memory access",
            "out of memory",
            "checkpoint",
            "alignment",
            "capture positions",
            "state mismatch",
        )
    ):
        return False
    return isinstance(exc, (UnicodeError, ValueError))
