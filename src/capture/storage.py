import json
from pathlib import Path

import torch

from src.models.base import GenerationResult
from src.capture.trajectory import Trajectory


def save_trajectory_states(
    trajectory: Trajectory,
    output_path: str | Path,
) -> None:
    checkpoints = trajectory.checkpoints

    if checkpoints:
        # [T, L, H]
        all_layers = torch.stack(
            [
                trajectory.generation.all_layers[checkpoint]
                for checkpoint in checkpoints
            ],
            dim=0,
        )

        # [T, H]
        final_norm = torch.stack(
            [
                trajectory.generation.final_norm[checkpoint]
                for checkpoint in checkpoints
            ],
            dim=0,
        )
    else:
        all_layers = torch.empty(
            (0, trajectory.generation.num_layers, trajectory.generation.hidden_size),
            dtype=torch.float32,
        )
        final_norm = torch.empty(
            (0, trajectory.generation.hidden_size),
            dtype=torch.float32,
        )

    payload = {
        "checkpoints": checkpoints,
        "fixed_checkpoints": (trajectory.fixed_checkpoints),
        "sentence_checkpoints": (trajectory.sentence_checkpoints),
        "all_layers": (all_layers.detach().cpu()),
        "final_norm": (final_norm.detach().cpu()),
    }

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    torch.save(
        payload,
        tmp_path,
    )

    tmp_path.replace(output_path)


def trajectory_to_metadata(trajectory: Trajectory, state_path: str | Path) -> dict:
    meta = {}
    meta["trial_id"] = trajectory.trial_id
    meta["pair_id"] = trajectory.pair_id
    meta["condition"] = trajectory.condition
    meta["split"] = trajectory.split
    meta["source"] = trajectory.source
    meta["category"] = trajectory.category

    meta["voice"] = trajectory.voice
    meta["audio_path"] = trajectory.audio_path
    meta["raw_text"] = trajectory.raw_text
    meta["tts_text"] = trajectory.tts_text
    meta["prompt_text"] = trajectory.prompt_text
    meta["gold_answer"] = list(trajectory.gold_answer)

    meta["model_name"] = trajectory.model_name
    meta["text"] = trajectory.generation.text
    meta["num_generated_tokens"] = trajectory.num_generated_tokens
    meta["stop_reason"] = trajectory.stop_reason
    if trajectory.generation.generation_stop_policy is not None:
        meta["generation_stop_policy"] = trajectory.generation.generation_stop_policy
        meta["generation_stop_marker"] = trajectory.generation.generation_stop_marker
    meta["token_ids"] = trajectory.generation.token_ids
    if trajectory.generation.token_capture_method is not None:
        meta["token_capture_method"] = trajectory.generation.token_capture_method

    meta["checkpoints"] = list(trajectory.checkpoints)
    meta["fixed_checkpoints"] = list(trajectory.fixed_checkpoints)
    meta["sentence_checkpoints"] = list(trajectory.sentence_checkpoints)

    if trajectory.checkpoints:
        first_state = trajectory.generation.all_layers[trajectory.checkpoints[0]]
        meta["num_layers"] = int(first_state.shape[0])
        meta["hidden_dim"] = int(first_state.shape[1])
    else:
        meta["num_layers"] = trajectory.generation.num_layers
        meta["hidden_dim"] = trajectory.generation.hidden_size

    from src.models.probing import BOUNDARY_VERSION
    from src.eval.response_format import inspect_response_format

    meta["boundary_version"] = trajectory.generation.boundary_version
    import hashlib

    meta["boundary_events"] = [
        dict(
            event,
            prefix_token_ids_sha256=hashlib.sha256(
                json.dumps(
                    meta["token_ids"][: event["confirmation_token_index"]],
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        )
        for event in trajectory.generation.boundary_events
    ]
    meta["monitoring_stop"] = trajectory.generation.monitoring_stop
    meta["boundary_exclusions"] = trajectory.generation.boundary_exclusions
    meta["pending_boundary_at_termination"] = trajectory.generation.pending_boundary
    meta["consumed_generated_tokens"] = trajectory.generation.consumed_generated_tokens
    meta["tokenizer_fingerprint"] = trajectory.generation.tokenizer_fingerprint
    meta["eligible_checkpoint_count"] = len(trajectory.sentence_checkpoints)
    meta["response_format"] = inspect_response_format(meta["text"], meta["model_name"])
    if not trajectory.sentence_checkpoints:
        from src.capture.v10_eligibility import zero_checkpoint_record

        meta = zero_checkpoint_record(meta)
    else:
        meta["monitoring_status"] = "eligible_reasoning_checkpoints"
    if meta["boundary_version"] == BOUNDARY_VERSION:
        import hashlib

        meta["boundary_source_sha256"] = hashlib.sha256(
            Path("src/models/probing.py").read_bytes()
        ).hexdigest()
        from src.data.frozen_umwp import frozen_split_record

        if meta["pair_id"].startswith("umwp_"):
            meta["frozen_split_sha256"] = frozen_split_record()["sha256"]
    meta["state_path"] = str(state_path)
    return meta


def append_metadata(
    metadata: dict,
    metadata_path: str | Path,
) -> None:
    metadata_path = Path(metadata_path)
    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with metadata_path.open(
        "a",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
        )
        f.write("\n")


def save_trajectory(
    trajectory: Trajectory,
    state_path: str | Path,
    metadata_path: str | Path,
) -> None:
    save_trajectory_states(
        trajectory=trajectory,
        output_path=state_path,
    )

    metadata = trajectory_to_metadata(
        trajectory=trajectory,
        state_path=state_path,
    )

    append_metadata(
        metadata=metadata,
        metadata_path=metadata_path,
    )


def load_trajectory_states(
    state_path: str | Path,
) -> dict:
    state_path = Path(state_path)

    payload = torch.load(
        state_path,
        map_location="cpu",
    )

    required_keys = {
        "checkpoints",
        "fixed_checkpoints",
        "sentence_checkpoints",
        "all_layers",
        "final_norm",
    }

    missing_keys = required_keys - payload.keys()

    if missing_keys:
        raise ValueError(
            "state file is missing required keys: " f"{sorted(missing_keys)}"
        )

    all_layers = payload["all_layers"]
    final_norm = payload["final_norm"]

    if not isinstance(
        all_layers,
        torch.Tensor,
    ):
        raise TypeError("all_layers must be a torch.Tensor")

    if not isinstance(
        final_norm,
        torch.Tensor,
    ):
        raise TypeError("final_norm must be a torch.Tensor")

    if all_layers.ndim != 3:
        raise ValueError("all_layers must have shape [T, L, H]")

    if final_norm.ndim != 2:
        raise ValueError("final_norm must have shape [T, H]")

    if all_layers.shape[0] != final_norm.shape[0]:
        raise ValueError(
            "all_layers and final_norm must " "have the same number of checkpoints"
        )

    if all_layers.shape[2] != final_norm.shape[1]:
        raise ValueError("all_layers and final_norm hidden " "dimensions must match")

    checkpoints = tuple(payload["checkpoints"])

    fixed_checkpoints = tuple(payload["fixed_checkpoints"])

    sentence_checkpoints = tuple(payload["sentence_checkpoints"])

    for name, positions in (
        ("checkpoints", checkpoints),
        ("fixed_checkpoints", fixed_checkpoints),
        ("sentence_checkpoints", sentence_checkpoints),
    ):
        GenerationResult._validate_checkpoint_tuple(positions, name)
    if not torch.isfinite(all_layers).all() or not torch.isfinite(final_norm).all():
        raise ValueError("state file contains non-finite hidden states")

    if len(checkpoints) != all_layers.shape[0]:
        raise ValueError(
            "number of checkpoints must match " "the first dimension of all_layers"
        )

    if any(checkpoint <= 0 for checkpoint in checkpoints):
        raise ValueError("checkpoints must use positive " "1-based token indices")

    if any(
        current <= previous
        for previous, current in zip(
            checkpoints,
            checkpoints[1:],
        )
    ):
        raise ValueError("checkpoints must be strictly increasing")

    if set(checkpoints) != (set(fixed_checkpoints) | set(sentence_checkpoints)):
        raise ValueError(
            "checkpoints must equal the union of "
            "fixed_checkpoints and "
            "sentence_checkpoints"
        )

    payload["checkpoints"] = checkpoints
    payload["fixed_checkpoints"] = fixed_checkpoints
    payload["sentence_checkpoints"] = sentence_checkpoints

    return payload


def load_metadata(
    metadata_path: str | Path,
) -> list[dict]:
    metadata_path = Path(metadata_path)

    records = []
    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            records.append(json.loads(line))

    return records


def metadata_to_trajectory(
    metadata: dict,
) -> Trajectory:
    states = load_trajectory_states(metadata["state_path"])

    for name in ("checkpoints", "fixed_checkpoints", "sentence_checkpoints"):
        if name in metadata and metadata[name] != list(states[name]):
            raise ValueError(f"metadata {name} do not match state file {name}")

    if metadata.get("boundary_version"):
        if tuple(states["all_layers"].shape[1:]) != (
            metadata.get("num_layers"),
            metadata.get("hidden_dim"),
        ):
            raise ValueError("Saved state dimensions disagree with metadata")
        if metadata.get("eligible_checkpoint_count") != len(
            states["sentence_checkpoints"]
        ):
            raise ValueError("Eligible checkpoint count disagrees with saved positions")
    checkpoints = states["checkpoints"]

    all_layers = {
        checkpoint: states["all_layers"][i] for i, checkpoint in enumerate(checkpoints)
    }
    final_norm = {
        checkpoint: states["final_norm"][i] for i, checkpoint in enumerate(checkpoints)
    }

    fixed_checkpoints = tuple(
        metadata.get(
            "fixed_checkpoints",
            states["fixed_checkpoints"],
        )
    )
    sentence_checkpoints = tuple(
        metadata.get(
            "sentence_checkpoints",
            states["sentence_checkpoints"],
        )
    )

    result = GenerationResult(
        model_name=metadata["model_name"],
        token_ids=metadata["token_ids"],
        text=metadata["text"],
        num_generated_tokens=(metadata["num_generated_tokens"]),
        stop_reason=metadata["stop_reason"],
        all_layers=all_layers,
        final_norm=final_norm,
        fixed_checkpoints=(fixed_checkpoints),
        sentence_checkpoints=(sentence_checkpoints),
        token_capture_method=metadata.get("token_capture_method"),
        num_layers=int(states["all_layers"].shape[1]),
        hidden_size=int(states["all_layers"].shape[2]),
        boundary_version=metadata.get("boundary_version"),
        boundary_events=tuple(metadata.get("boundary_events", [])),
        monitoring_stop=metadata.get("monitoring_stop"),
        boundary_exclusions=metadata.get("boundary_exclusions", {}),
        pending_boundary=metadata.get("pending_boundary_at_termination"),
        consumed_generated_tokens=metadata.get("consumed_generated_tokens"),
        tokenizer_fingerprint=metadata.get("tokenizer_fingerprint"),
        generation_stop_policy=metadata.get("generation_stop_policy"),
        generation_stop_marker=metadata.get("generation_stop_marker"),
    )

    trajectory = Trajectory(
        trial_id=metadata["trial_id"],
        pair_id=metadata["pair_id"],
        condition=metadata["condition"],
        split=metadata["split"],
        source=metadata["source"],
        category=(
            int(metadata["category"]) if metadata["category"] is not None else None
        ),
        voice=metadata["voice"],
        audio_path=metadata["audio_path"],
        raw_text=metadata["raw_text"],
        tts_text=metadata["tts_text"],
        prompt_text=metadata["prompt_text"],
        gold_answer=tuple(metadata["gold_answer"]),
        generation=result,
    )

    return trajectory
