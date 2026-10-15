"""Canonical V10 split integrity, shared by capture and downstream gate loaders."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CANONICAL_SPLIT = ROOT / "data/splits/umwp_final_split.json"


def frozen_split_record():
    return json.loads(
        (ROOT / "docs/provenance/phase2b/frozen_umwp_split.json").read_text()
    )


def verify_frozen_split(path):
    path = Path(path)
    from src.data.frozen_treecut import SPLIT_PATH, verify_dataset

    if path.resolve() == SPLIT_PATH.resolve():
        verify_dataset()
    if path.resolve() == CANONICAL_SPLIT.resolve():
        expected = frozen_split_record()["sha256"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError("Frozen UMWP manifest changed; refusing execution")


def v10_artifact_provenance(dataset_name, model_name=None):
    if dataset_name == "treecut":
        from src.data.frozen_treecut import artifact_provenance

        return artifact_provenance(model_name)
    if dataset_name != "umwp":
        return {}
    from src.capture.v10_prompt import PROMPT_VERSION
    from src.models.probing import BOUNDARY_VERSION

    verify_frozen_split(CANONICAL_SPLIT)
    from src.models.audio_reasoner_stop import stop_policy_provenance

    return dict(
        **stop_policy_provenance(model_name),
        frozen_split_sha256=frozen_split_record()["sha256"],
        prompt_version=PROMPT_VERSION,
        boundary_version=BOUNDARY_VERSION,
        boundary_source_sha256=hashlib.sha256(
            (ROOT / "src/models/probing.py").read_bytes()
        ).hexdigest(),
    )
