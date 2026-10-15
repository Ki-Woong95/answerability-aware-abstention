import argparse
import json
from pathlib import Path

from src.capture.storage import save_trajectory
from src.capture.trajectory import Trajectory
from src.capture.worldsense_validation import (
    gold_answer_for_trial,
    load_completed,
    preflight_trials,
)
from src.config import MAX_NEW_TOKENS


TASK_PROMPT = """Listen to the audio and answer the question below. Reason step by step using only the information provided in the audio.

Question:
{question}

Assess whether the audio contains sufficient information to answer the question.

Use only information explicitly provided in the audio and do not infer missing facts.

If the answer cannot be uniquely determined from the audio, conclude with:
FINAL: I don't know

Otherwise, conclude with:
FINAL: <answer>"""

DEFAULT_TRIALS = Path("data/processed/worldsense_trials_openended.jsonl")
DEFAULT_TTS_MANIFEST = Path("data/tts/worldsense_manifest.jsonl")
DEFAULT_SPLITS = Path("data/splits/worldsense_openended_split.json")
DEFAULT_OUTPUT = Path("data/ablations/always_idk/worldsense/trajectories")


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_split_map(path):
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)

    data = data["splits"]

    split_map = {}
    for split in ("gate_train", "calibration", "val", "test"):
        if split not in data:
            raise ValueError(f"Missing split: {split}")

        for source_id in data[split]:
            source_id = str(source_id)

            if source_id in split_map:
                raise ValueError(f"Duplicate source_id across splits: {source_id}")

            split_map[source_id] = split

    return split_map


def load_tts_map(path):
    rows = read_jsonl(path)
    out = {}

    for row in rows:
        source_id = str(row["source_id"])

        if source_id in out:
            raise RuntimeError(f"Duplicate source_id in TTS manifest: {source_id}")

        out[source_id] = row

    return out


def build_model(name):
    if name == "af_next":
        from src.models.af_next import AFNextModel

        return AFNextModel(
            model_id="nvidia/audio-flamingo-next-think-hf",
            prompt_template="{question}",
        )

    if name == "audio-reasoner":
        from src.models.audio_reasoner import AudioReasonerModel

        return AudioReasonerModel()

    if name == "moss":
        from src.models.moss import MOSSModel

        return MOSSModel()

    raise ValueError(f"Unknown model: {name}")


def validate_result(result, model):
    if result.fixed_checkpoints:
        raise RuntimeError(
            f"Expected no fixed checkpoints, got {result.fixed_checkpoints}"
        )

    if not result.sentence_checkpoints:
        raise RuntimeError("No sentence checkpoints captured")

    sentence = set(result.sentence_checkpoints)

    if set(result.all_layers) != sentence:
        raise RuntimeError("all_layers positions do not match sentence checkpoints")

    if set(result.final_norm) != sentence:
        raise RuntimeError("final_norm positions do not match sentence checkpoints")

    for checkpoint in result.sentence_checkpoints:
        state = result.all_layers[checkpoint]

        if state.ndim != 2:
            raise RuntimeError(
                f"Checkpoint {checkpoint}: expected [L,H], " f"got {tuple(state.shape)}"
            )

        if state.shape[1] != model.hidden_dim:
            raise RuntimeError(
                f"Checkpoint {checkpoint}: hidden dim "
                f"{state.shape[1]} != {model.hidden_dim}"
            )


def select_rows(trials, split_map, split, condition, limit_pairs):
    if limit_pairs is not None and limit_pairs <= 0:
        raise ValueError("--limit-pairs must be positive")
    rows = []

    for row in trials:
        source_id = str(row["source_id"])

        if source_id not in split_map:
            raise RuntimeError(f"{source_id}: missing from frozen split")

        if row["split"] != split_map[source_id]:
            raise RuntimeError(
                f"{source_id}: trial split={row['split']} "
                f"but frozen split={split_map[source_id]}"
            )

        if split_map[source_id] != split:
            continue

        if condition != "all" and row["condition"] != condition:
            continue

        rows.append(row)

    if limit_pairs is None:
        return rows

    selected = []
    seen = set()

    for row in rows:
        source_id = str(row["source_id"])

        if source_id not in seen:
            seen.add(source_id)
            selected.append(source_id)

        if len(selected) >= limit_pairs:
            break

    selected = set(selected)

    return [row for row in rows if str(row["source_id"]) in selected]


def run(args):
    if args.split == "test" and not args.allow_test:
        raise RuntimeError("Refusing to touch test split without --allow-test")

    trials = read_jsonl(args.trials)
    split_map = load_split_map(args.splits)
    tts_map = load_tts_map(args.tts_manifest)

    condition = args.condition

    if args.split == "calibration" and condition == "all":
        condition = "answerable"

    rows = select_rows(
        trials=trials,
        split_map=split_map,
        split=args.split,
        condition=condition,
        limit_pairs=args.limit_pairs,
    )

    if not rows:
        raise RuntimeError("No matching trials found")

    output_dir = args.output / args.model / args.split
    state_dir = output_dir / "states"
    metadata_path = output_dir / "metadata.jsonl"

    state_dir.mkdir(parents=True, exist_ok=True)

    if metadata_path.exists() and not args.resume:
        raise FileExistsError(f"{metadata_path} already exists; rerun with --resume")
    completed = load_completed(metadata_path) if args.resume else {}
    preflight_trials(rows, tts_map, completed, TASK_PROMPT, args.model, state_dir)
    if all(str(row["trial_id"]) in completed for row in rows):
        print(f"All {len(rows)} selected trajectories are already complete")
        return

    model = build_model(args.model)

    print(f"Loading {args.model}...")
    model.load()

    print(f"Hidden dim: {model.hidden_dim}")
    print(f"Trials selected: {len(rows)}")

    for i, row in enumerate(rows, start=1):
        trial_id = str(row["trial_id"])
        pair_id = str(row["source_id"])

        tts = tts_map[pair_id]

        state_path = state_dir / f"{trial_id}.pt"

        if args.resume:
            if trial_id in completed:
                print(f"[{i}/{len(rows)}] skip {trial_id}")
                continue

            if state_path.exists():
                print(f"[{i}/{len(rows)}] removing orphan state " f"{trial_id}")
                state_path.unlink()

        elif state_path.exists():
            raise FileExistsError(f"{state_path} already exists; rerun with --resume")

        prompt_text = TASK_PROMPT.format(question=row["question"])

        print(f"[{i}/{len(rows)}] {trial_id} | " f"{row['condition']}")
        print(f"  question={row['question']}")

        result = model.generate_full(
            audio_path=row["audio_path"],
            prompt_text=prompt_text,
            max_new_tokens=MAX_NEW_TOKENS,
            checkpoints=(),
            capture_sentence_boundaries=True,
        )

        validate_result(result, model)

        gold_answer = gold_answer_for_trial(row)

        trajectory = Trajectory(
            trial_id=trial_id,
            pair_id=pair_id,
            condition=row["condition"],
            split=args.split,
            source="worldsense",
            category=None,
            voice=str(tts["voice"]),
            audio_path=str(row["audio_path"]),
            raw_text=str(row["audio_text"]),
            tts_text=str(row["audio_text"]),
            prompt_text=prompt_text,
            gold_answer=gold_answer,
            generation=result,
        )

        save_trajectory(
            trajectory=trajectory,
            state_path=state_path,
            metadata_path=metadata_path,
        )

        first_cp = result.sentence_checkpoints[0]
        shape = tuple(result.all_layers[first_cp].shape)

        print(
            f"  tokens={result.num_generated_tokens} "
            f"sentences={len(result.sentence_checkpoints)} "
            f"state_shape={shape} "
            f"stop={result.stop_reason}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=["af_next", "audio-reasoner", "moss"],
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=["gate_train", "calibration", "val", "test"],
    )
    parser.add_argument(
        "--condition",
        choices=["all", "answerable", "unanswerable"],
        default="all",
    )

    parser.add_argument("--limit-pairs", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-test", action="store_true")

    parser.add_argument(
        "--trials",
        type=Path,
        default=DEFAULT_TRIALS,
    )
    parser.add_argument(
        "--tts-manifest",
        type=Path,
        default=DEFAULT_TTS_MANIFEST,
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=DEFAULT_SPLITS,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()
