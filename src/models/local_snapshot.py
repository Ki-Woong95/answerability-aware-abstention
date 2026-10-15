"""Resolve the audited AR snapshot without consulting a remote model hub."""

import json
from pathlib import Path

AR_MODEL_ID = "zhifeixie/Audio-Reasoner"
ROOT = Path(__file__).resolve().parents[2]


def resolve_audio_reasoner_snapshot(model_id=AR_MODEL_ID):
    if str(model_id) == AR_MODEL_ID:
        record = json.loads(
            (ROOT / "docs/provenance/phase2b/audio-reasoner_tokenizer.json").read_text()
        )
        path = Path(record["tokenizer_path"])
    else:
        path = Path(model_id).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(
            f"Audio-Reasoner local snapshot unavailable: {path}; refusing remote download"
        )
    index = path / "model.safetensors.index.json"
    files = {"config.json", "tokenizer.json"}
    if index.exists():
        files.update(json.loads(index.read_text())["weight_map"].values())
    elif (path / "model.safetensors").is_file():
        files.add("model.safetensors")
    else:
        raise FileNotFoundError(f"Audio-Reasoner local weights missing: {path}")
    missing = sorted(name for name in files if not (path / name).is_file())
    if missing:
        raise FileNotFoundError(
            f"Audio-Reasoner local snapshot incomplete: {missing}; refusing remote download"
        )
    return str(path.resolve())
