"""Common saved-result checks for all capture runners."""

import json
from pathlib import Path

from src.capture.storage import metadata_to_trajectory
from src.models.base import COMPLETE_TOKEN_CAPTURE


def load_completed(metadata_path):
    metadata_path = Path(metadata_path)
    completed = {}
    if not metadata_path.exists():
        return completed
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        metadata = json.loads(line)
        trial_id = str(metadata["trial_id"])
        if trial_id in completed:
            raise RuntimeError(f"Duplicate metadata for trial_id: {trial_id}")
        completed[trial_id] = metadata
    return completed


def validate_completed(metadata, expected, state_path):
    trial_id = expected["trial_id"]
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(
                f"{trial_id}: saved {key} does not match current trial; "
                "refusing to resume stale results"
            )
    if (
        expected["model_name"] == "audio-reasoner"
        and metadata.get("token_capture_method") != COMPLETE_TOKEN_CAPTURE
    ):
        raise RuntimeError(
            f"{trial_id}: saved Audio-Reasoner tokens predate complete token capture. "
            "Recapture into a fresh --output directory; the old state files are preserved."
        )
    from src.capture.v10_prompt import TASK_PROMPT

    if expected.get("prompt_text") == TASK_PROMPT:
        if expected["model_name"] == "audio-reasoner":
            from src.models.audio_reasoner_stop import STOP_POLICY

            if metadata.get("generation_stop_policy") != STOP_POLICY:
                raise RuntimeError(
                    f"{trial_id}: stale AR generation stop policy; refusing resume"
                )
        import hashlib
        from src.models.probing import BOUNDARY_VERSION

        current_hash = hashlib.sha256(
            Path("src/models/probing.py").read_bytes()
        ).hexdigest()
        if (
            metadata.get("boundary_version") != BOUNDARY_VERSION
            or metadata.get("boundary_source_sha256") != current_hash
        ):
            raise RuntimeError(
                f"{trial_id}: saved boundary rules predate V10; refusing stale resume"
            )
    if not state_path.exists():
        raise FileNotFoundError(
            f"Metadata exists but state file is missing: {state_path}"
        )
    if Path(metadata["state_path"]).resolve() != state_path.resolve():
        raise RuntimeError(
            f"{trial_id}: saved state_path does not match output directory"
        )
    # Also verifies token counts, checkpoint bounds and metadata/state agreement.
    try:
        trajectory = metadata_to_trajectory(metadata)
    except (ValueError, TypeError) as exc:
        raise RuntimeError(f"{trial_id}: {exc}") from exc
    if trajectory.checkpoints:
        state = trajectory.generation.all_layers[trajectory.checkpoints[0]]
        if list(state.shape) != [metadata["num_layers"], metadata["hidden_dim"]]:
            raise RuntimeError(
                f"{trial_id}: saved state dimensions do not match metadata"
            )


def validate_trial_ids(rows, completed):
    ids = [str(row["trial_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate input trial_id")
    return set(ids)
