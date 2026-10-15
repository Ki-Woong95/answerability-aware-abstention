"""TreeCut dataset fingerprints; UMWP prompt and experimental policy unchanged."""

import hashlib, json
from pathlib import Path
from src.capture.v10_prompt import TASK_PROMPT, PROMPT_VERSION

ROOT = Path(__file__).resolve().parents[2]
SPLIT_PATH = ROOT / "data/splits/treecut_final_split.json"
TRIALS_PATH = ROOT / "data/treecut/trials.jsonl"
RECORD_PATH = ROOT / "docs/provenance/treecut/frozen_dataset.json"
_verified = None
_record = None


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_dataset():
    global _verified, _record
    record = json.loads(RECORD_PATH.read_text())
    paths = [RECORD_PATH] + [ROOT / p for p in record["protected_sha256"]]
    signature = tuple(
        (str(p), p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns)
        for p in paths
    )
    if signature != _verified:
        for p, h in record["protected_sha256"].items():
            if sha256(ROOT / p) != h:
                raise RuntimeError("Frozen TreeCut input changed: " + p)
        _verified, _record = signature, record
    return _record


def artifact_provenance(model_name=None):
    from src.models.probing import BOUNDARY_VERSION
    from src.models.audio_reasoner_stop import stop_policy_provenance

    record = verify_dataset()
    return dict(
        **stop_policy_provenance(model_name),
        frozen_split_sha256=record["protected_sha256"][
            "data/splits/treecut_final_split.json"
        ],
        dataset_trials_sha256=record["protected_sha256"]["data/treecut/trials.jsonl"],
        prompt_version=PROMPT_VERSION,
        task_prompt_sha256=hashlib.sha256(TASK_PROMPT.encode()).hexdigest(),
        boundary_version=BOUNDARY_VERSION,
        boundary_source_sha256=sha256(ROOT / "src/models/probing.py"),
    )


def validate_provenance(row, model_name=None):
    if any(
        row.get(k) != v
        for k, v in artifact_provenance(model_name or row.get("model_name")).items()
    ):
        raise RuntimeError(
            "Missing/stale TreeCut split, prompt, boundary or AR stop provenance"
        )


def load_trials():
    verify_dataset()
    return [json.loads(line) for line in TRIALS_PATH.open() if line.strip()]
