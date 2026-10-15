import argparse, json
from pathlib import Path

import numpy as np
import soundfile as sf
from src.data.tts_text import normalize_tts_text

VOICE = "af_heart"
SPEED = 1.0
SR = 24000


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(x) for x in stream if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")

    pairs = load_jsonl("data/processed/umwp_pairs.jsonl")
    with Path("data/splits/umwp_final_split.json").open(encoding="utf-8") as stream:
        split_data = json.load(stream)
    pair_to_split = {
        pid: split
        for split in ["gate_train", "calibration", "val", "test"]
        for pid in split_data[split]
    }

    out_root = Path("data/tts/umwp")
    manifest_path = Path("data/tts/umwp_manifest.jsonl")
    error_path = Path("logs/tts_umwp_errors.jsonl")
    out_root.mkdir(parents=True, exist_ok=True)
    error_path.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if manifest_path.exists():
        for row in load_jsonl(manifest_path):
            trial_id = row["trial_id"]
            if trial_id in existing:
                raise RuntimeError(f"Duplicate TTS manifest trial_id: {trial_id}")
            existing[trial_id] = row
    pipeline = None
    manifest, errors = dict(existing), []
    trials = []

    for pair in pairs:
        for condition, suffix in [("answerable", "A"), ("unanswerable", "U")]:
            trials.append((pair, condition, suffix))

    if args.limit is not None:
        trials = trials[: args.limit]

    for i, (pair, condition, suffix) in enumerate(trials, 1):
        pid = pair["pair_id"]
        split = pair_to_split[pid]
        item = pair[condition]
        trial_id = f"{pid}_{suffix}"
        audio_dir = out_root / split
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path = audio_dir / f"{trial_id}.wav"
        raw_text = item["question"]
        tts_text = normalize_tts_text(raw_text)

        if audio_path.exists() and not args.overwrite:
            old = existing.get(trial_id)
            if old is None:
                raise RuntimeError(
                    "Existing WAV has no manifest provenance; use --overwrite to regenerate"
                )
            for field, value in {
                "raw_text": raw_text,
                "tts_text": tts_text,
                "audio_path": str(audio_path),
                "voice": VOICE,
                "speed": SPEED,
                "split": split,
            }.items():
                if old.get(field) != value:
                    raise RuntimeError(
                        f"Existing WAV/manifest has different {field}; use --overwrite"
                    )
        try:
            if args.overwrite or not audio_path.exists():
                if pipeline is None:
                    from kokoro import KPipeline

                    pipeline = KPipeline(lang_code="a")
                parts = [
                    audio
                    for _, _, audio in pipeline(tts_text, voice=VOICE, speed=SPEED)
                ]
                if not parts:
                    raise RuntimeError("Kokoro returned no audio")
                audio = np.concatenate(parts)
                temporary_audio = audio_path.with_suffix(".wav.tmp")
                sf.write(temporary_audio, audio, SR, format="WAV")
                temporary_audio.replace(audio_path)

            info = sf.info(audio_path)
            row = {
                "trial_id": trial_id,
                "pair_id": pid,
                "condition": condition,
                "split": split,
                "source": pair["source"],
                "category": pair["category"],
                "raw_text": raw_text,
                "tts_text": tts_text,
                "audio_path": str(audio_path),
                "sample_rate": info.samplerate,
                "duration": info.duration,
                "voice": VOICE,
                "speed": SPEED,
            }
            manifest[trial_id] = row
            print(f"[{i}/{len(trials)}] {trial_id} {info.duration:.2f}s")
        except Exception as e:
            manifest.pop(trial_id, None)
            err = {"trial_id": trial_id, "error": repr(e)}
            errors.append(err)
            print("ERROR", trial_id, repr(e))

    tmp = manifest_path.with_suffix(".tmp")
    with tmp.open("w") as f:
        for row in manifest.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(manifest_path)

    with error_path.open("w") as f:
        for row in errors:
            f.write(json.dumps(row) + "\n")

    print("\nGenerated/verified:", len(manifest))
    print("Errors:", len(errors))
    print("Manifest:", manifest_path)
    print("Error log:", error_path)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
