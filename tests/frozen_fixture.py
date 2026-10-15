"""Temporary frozen-input fixtures for tests that exercise artifact provenance."""

import hashlib
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch


def use_synthetic_umwp(test):
    """Keep real hash validation while replacing the omitted research dataset."""
    from src.data import frozen_umwp
    from src.gate.datasets import DATASETS

    directory = tempfile.TemporaryDirectory()
    test.addCleanup(directory.cleanup)
    split_path = Path(directory.name) / "umwp_final_split.json"
    splits = {
        name: [f"synthetic_{name}"]
        for name in ("gate_train", "calibration", "val", "test")
    }
    split_path.write_text(json.dumps({"splits": splits}))
    record = {
        "sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "counts": {name: len(ids) for name, ids in splits.items()},
    }
    for fixture_patch in (
        patch.object(frozen_umwp, "CANONICAL_SPLIT", split_path),
        patch.object(frozen_umwp, "frozen_split_record", return_value=record),
        patch.dict(
            DATASETS, {"umwp": replace(DATASETS["umwp"], split_path=split_path)}
        ),
    ):
        fixture_patch.start()
        test.addCleanup(fixture_patch.stop)
