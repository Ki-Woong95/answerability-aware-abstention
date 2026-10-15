"""Validate UMWP manifests against pairs and frozen splits before model loading."""

from src.capture.validation import (
    load_completed,
    validate_completed,
    validate_trial_ids,
)
from src.data.tts_text import normalize_tts_text


def validate_manifest(manifest, pairs, split_map):
    pair_map = {}
    for pair in pairs:
        pair_id = str(pair["pair_id"])
        if pair_id in pair_map:
            raise RuntimeError(f"Duplicate input pair_id: {pair_id}")
        pair_map[pair_id] = pair
    validate_trial_ids(manifest, {})
    seen = set()
    for row in manifest:
        pair_id = str(row["pair_id"])
        if pair_id not in pair_map or pair_id not in split_map:
            raise RuntimeError(f"{pair_id}: missing from pairs or frozen split")
        condition = row["condition"]
        if condition not in ("answerable", "unanswerable"):
            raise RuntimeError(f"{row['trial_id']}: unknown condition {condition!r}")
        key = (pair_id, condition)
        if key in seen:
            raise RuntimeError(f"{pair_id}: duplicate {condition} manifest entry")
        seen.add(key)
        pair = pair_map[pair_id]
        raw_text = pair[condition]["question"]
        expected = {
            "trial_id": pair_id + ("_A" if condition == "answerable" else "_U"),
            "split": split_map[pair_id],
            "source": pair["source"],
            "category": pair["category"],
            "raw_text": raw_text,
            "tts_text": normalize_tts_text(raw_text),
        }
        for field, value in expected.items():
            if row.get(field) != value:
                raise RuntimeError(
                    f"{row['trial_id']}: manifest {field} does not match current pair/split"
                )

    required = {
        (pair_id, condition)
        for pair_id, split in split_map.items()
        for condition in (
            ("answerable",)
            if split == "calibration"
            else ("answerable", "unanswerable")
        )
    }
    missing = required - seen
    if missing:
        raise RuntimeError(
            f"Incomplete retained UMWP manifest: {len(missing)} required recordings missing"
        )


def preflight_trials(rows, pair_answers, completed, task_prompt, model_name, state_dir):
    validate_trial_ids(rows, completed)
    for row in rows:
        trial_id, pair_id = str(row["trial_id"]), str(row["pair_id"])
        gold = list(pair_answers[pair_id]) if row["condition"] == "answerable" else []
        if row["condition"] == "answerable" and not gold:
            raise RuntimeError(f"{trial_id}: missing answerable gold")
        if trial_id in completed:
            expected = {
                key: row[key]
                for key in (
                    "condition",
                    "source",
                    "category",
                    "voice",
                    "audio_path",
                    "raw_text",
                    "tts_text",
                )
            }
            expected.update(
                trial_id=trial_id,
                pair_id=pair_id,
                split=row["split"],
                prompt_text=task_prompt,
                model_name=model_name,
                gold_answer=gold,
            )
            validate_completed(
                completed[trial_id], expected, state_dir / f"{trial_id}.pt"
            )
