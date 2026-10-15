import argparse
import json
from pathlib import Path

from src.capture.storage import save_trajectory
from src.capture.trajectory import Trajectory
from src.capture.umwp_validation import (
    load_completed,
    preflight_trials,
    validate_manifest,
)
from src.config import MAX_NEW_TOKENS


from src.capture.v10_prompt import TASK_PROMPT

DEFAULT_MANIFEST = Path("data/tts/umwp_manifest.jsonl")
DEFAULT_PAIRS = Path("data/processed/umwp_pairs.jsonl")
DEFAULT_SPLITS = Path("data/splits/umwp_final_split.json")
DEFAULT_OUTPUT = Path("data/trajectories")


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def extract_pair_ids(values):
    pair_ids = []
    for value in values:
        if isinstance(value, str):
            pair_ids.append(value)
        elif isinstance(value, dict) and "pair_id" in value:
            pair_ids.append(str(value["pair_id"]))
        else:
            raise ValueError(f"Cannot extract pair_id from split entry: {value}")
    return pair_ids


def load_split_map(path):
    from src.data.frozen_umwp import verify_frozen_split

    verify_frozen_split(path)
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)

    if "splits" in data:
        data = data["splits"]

    split_map = {}
    for split in ("gate_train", "calibration", "val", "test"):
        if split not in data:
            raise ValueError(f"Missing split: {split}")
        for pair_id in extract_pair_ids(data[split]):
            if pair_id in split_map:
                raise ValueError(f"Duplicate pair_id across splits: {pair_id}")
            split_map[pair_id] = split

    return split_map


def load_pair_answers(path):
    answers = {}
    for row in read_jsonl(path):
        pair_id = str(row["pair_id"])
        answers[pair_id] = tuple(row["answerable"]["answer"])
    return answers


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

    sentence = set(result.sentence_checkpoints)

    if set(result.all_layers) != sentence:
        raise RuntimeError("all_layers positions do not match sentence checkpoints")

    if set(result.final_norm) != sentence:
        raise RuntimeError("final_norm positions do not match sentence checkpoints")

    for checkpoint in result.sentence_checkpoints:
        state = result.all_layers[checkpoint]
        if state.ndim != 2:
            raise RuntimeError(
                f"Checkpoint {checkpoint}: expected [L,H], got {tuple(state.shape)}"
            )
        if state.shape[1] != model.hidden_dim:
            raise RuntimeError(
                f"Checkpoint {checkpoint}: hidden dim {state.shape[1]} != {model.hidden_dim}"
            )


def select_rows(manifest, split_map, split, condition, limit_pairs):
    if limit_pairs is not None and limit_pairs <= 0:
        raise ValueError("--limit-pairs must be positive")
    rows = []

    for row in manifest:
        pair_id = str(row["pair_id"])
        if split_map.get(pair_id) != split:
            continue
        if condition != "all" and row["condition"] != condition:
            continue
        rows.append(row)

    if limit_pairs is None:
        return rows

    selected_pairs = []
    seen = set()

    for row in rows:
        pair_id = str(row["pair_id"])
        if pair_id not in seen:
            seen.add(pair_id)
            selected_pairs.append(pair_id)
        if len(selected_pairs) >= limit_pairs:
            break

    selected_pairs = set(selected_pairs)
    return [row for row in rows if str(row["pair_id"]) in selected_pairs]


def run(args, task_prompt=TASK_PROMPT, model_factory=None):
    if args.split == "test" and not args.allow_test:
        raise RuntimeError("Refusing to touch test split without --allow-test")

    manifest = read_jsonl(args.manifest)
    split_map = load_split_map(args.splits)
    pair_answers = load_pair_answers(args.pairs)
    manifest = [row for row in manifest if str(row["pair_id"]) in split_map]
    validate_manifest(manifest, read_jsonl(args.pairs), split_map)

    condition = args.condition
    if args.split == "calibration" and condition == "all":
        condition = "answerable"

    rows = select_rows(
        manifest=manifest,
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
    preflight_trials(rows, pair_answers, completed, task_prompt, args.model, state_dir)
    if all(str(row["trial_id"]) in completed for row in rows):
        print(f"All {len(rows)} selected trajectories are already complete")
        return

    model = (model_factory or build_model)(args.model)
    print(f"Loading {args.model}...")
    model.load()
    print(f"Hidden dim: {model.hidden_dim}")
    print(f"Trials selected: {len(rows)}")

    for i, row in enumerate(rows, start=1):
        trial_id = str(row["trial_id"])
        pair_id = str(row["pair_id"])

        state_path = state_dir / f"{trial_id}.pt"

        if args.resume:
            if trial_id in completed:
                print(f"[{i}/{len(rows)}] skip {trial_id}")
                continue

            if state_path.exists():
                print(f"[{i}/{len(rows)}] removing orphan state {trial_id}")
                state_path.unlink()

        elif state_path.exists():
            raise FileExistsError(f"{state_path} already exists; rerun with --resume")

        print(
            f"[{i}/{len(rows)}] {trial_id} | "
            f"{row['condition']} | {row['source']} | cat{row['category']}"
        )

        result = model.generate_full(
            audio_path=row["audio_path"],
            prompt_text=task_prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            checkpoints=(),
            capture_sentence_boundaries=True,
        )

        validate_result(result, model)

        gold_answer = pair_answers[pair_id] if row["condition"] == "answerable" else ()

        trajectory = Trajectory(
            trial_id=trial_id,
            pair_id=pair_id,
            condition=row["condition"],
            split=args.split,
            source=row["source"],
            category=int(row["category"]),
            voice=row["voice"],
            audio_path=row["audio_path"],
            raw_text=row["raw_text"],
            tts_text=row["tts_text"],
            prompt_text=task_prompt,
            gold_answer=gold_answer,
            generation=result,
        )

        save_trajectory(
            trajectory=trajectory,
            state_path=state_path,
            metadata_path=metadata_path,
        )

        shape = (
            (result.num_layers, result.hidden_size)
            if not result.sentence_checkpoints
            else tuple(result.all_layers[result.sentence_checkpoints[0]].shape)
        )

        print(
            f"  tokens={result.num_generated_tokens} "
            f"sentences={len(result.sentence_checkpoints)} "
            f"state_shape={shape} "
            f"stop={result.stop_reason}"
        )


def main(default_output=DEFAULT_OUTPUT, task_prompt=TASK_PROMPT, model_factory=None):
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

    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--output", type=Path, default=default_output)

    args = parser.parse_args()

    run(args, task_prompt=task_prompt, model_factory=model_factory)


if __name__ == "__main__":
    main()
