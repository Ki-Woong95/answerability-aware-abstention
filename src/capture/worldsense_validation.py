"""Input and resume checks shared by the WorldSense capture runners."""

from src.capture.validation import load_completed, validate_completed


def gold_answer_for_trial(row):
    condition = row["condition"]
    gold = row["gold_answer"]
    if condition == "answerable":
        if gold is None or not str(gold).strip():
            raise RuntimeError(f"{row['trial_id']}: invalid answerable gold {gold!r}")
        return (str(gold),)
    if condition == "unanswerable":
        if gold is not None:
            raise RuntimeError(f"{row['trial_id']}: U gold_answer should be None")
        return ()
    raise RuntimeError(f"{row['trial_id']}: unknown condition {condition!r}")


def validate_trial_audio(row, tts):
    for key in ("audio_path", "audio_text", "split"):
        if str(tts[key]) != str(row[key]):
            raise RuntimeError(
                f"{row['trial_id']}: {key} mismatch between trial and TTS manifest"
            )
    if tts.get("dataset") != "worldsense":
        raise RuntimeError(f"{row['trial_id']}: TTS manifest is not WorldSense")


def validate_completed_trial(metadata, row, tts, prompt_text, model_name, state_path):
    expected = {
        "trial_id": str(row["trial_id"]),
        "pair_id": str(row["source_id"]),
        "condition": row["condition"],
        "split": row["split"],
        "source": "worldsense",
        "model_name": model_name,
        "voice": str(tts["voice"]),
        "audio_path": str(row["audio_path"]),
        "raw_text": str(row["audio_text"]),
        "tts_text": str(row["audio_text"]),
        "prompt_text": prompt_text,
        "gold_answer": list(gold_answer_for_trial(row)),
    }
    validate_completed(metadata, expected, state_path)


def preflight_trials(rows, tts_map, completed, task_prompt, model_name, state_dir):
    seen = set()
    for row in rows:
        trial_id = str(row["trial_id"])
        if trial_id in seen:
            raise RuntimeError(f"Duplicate input trial_id: {trial_id}")
        seen.add(trial_id)
        source_id = str(row["source_id"])
        if source_id not in tts_map:
            raise RuntimeError(f"{trial_id}: no TTS manifest entry for {source_id}")
        tts = tts_map[source_id]
        validate_trial_audio(row, tts)
        gold_answer_for_trial(row)
        if trial_id in completed:
            validate_completed_trial(
                completed[trial_id],
                row,
                tts,
                task_prompt.format(question=row["question"]),
                model_name,
                state_dir / f"{trial_id}.pt",
            )
