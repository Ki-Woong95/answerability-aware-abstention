import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

PAIR_PATH = Path("data/processed/worldsense_pairs_openended.jsonl")
MANIFEST_PATH = Path("data/tts/worldsense_manifest.jsonl")

VOICE = "af_heart"
SPEED = 1.0
SAMPLE_RATE = 24000


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def synthesize(pipeline, text):
    chunks = []

    for result in pipeline(text, voice=VOICE, speed=SPEED):
        if hasattr(result, "audio"):
            audio = result.audio
        elif isinstance(result, (tuple, list)):
            audio = result[-1]
        else:
            raise TypeError(f"Unexpected Kokoro output: {type(result)}")

        if audio is None:
            continue

        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)

        if len(audio):
            chunks.append(audio)

    if not chunks:
        raise RuntimeError(f"Kokoro returned no audio for: {text!r}")

    return np.concatenate(chunks)


def load_manifest():
    if not MANIFEST_PATH.exists():
        return {}

    out = {}

    for row in read_jsonl(MANIFEST_PATH):
        source_id = row["source_id"]

        if source_id in out:
            raise RuntimeError(f"Duplicate source_id in manifest: {source_id}")

        out[source_id] = row

    return out


def write_manifest(rows):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST_PATH.with_suffix(".jsonl.tmp")

    with tmp.open("w", encoding="utf-8") as f:
        for row in sorted(rows.values(), key=lambda x: x["source_id"]):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    tmp.replace(MANIFEST_PATH)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--split",
        choices=["gate_train", "calibration", "val", "test"],
        default=None,
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate selected WAVs from current text",
    )
    args = ap.parse_args()

    pairs = read_jsonl(PAIR_PATH)

    if args.split is not None:
        pairs = [x for x in pairs if x["split"] == args.split]

    print("pairs to check:", len(pairs))

    manifest = load_manifest()
    pipeline = None

    done = 0
    generated = 0

    for i, row in enumerate(pairs, 1):
        source_id = row["source_id"]
        audio_text = row["audio_text"]
        wav_path = Path(row["audio_path"])

        wav_path.parent.mkdir(parents=True, exist_ok=True)

        if wav_path.exists() and not args.overwrite:
            if source_id not in manifest:
                raise RuntimeError(
                    f"{source_id}: existing WAV has no manifest provenance; use --overwrite"
                )
            old = manifest[source_id]
            for key, value in {
                "audio_text": audio_text,
                "audio_path": str(wav_path),
                "voice": VOICE,
                "tts_speed": SPEED,
                "split": row["split"],
            }.items():
                if old.get(key) != value:
                    raise RuntimeError(
                        f"{source_id}: existing WAV/manifest has different {key}; use --overwrite"
                    )

            info = sf.info(wav_path)
            duration = info.frames / info.samplerate
            sr = info.samplerate
            done += 1

        else:
            print(f"[{i}/{len(pairs)}] {source_id}")
            if pipeline is None:
                from kokoro import KPipeline

                pipeline = KPipeline(lang_code="a")

            audio = synthesize(pipeline, audio_text)

            temporary_audio = wav_path.with_suffix(".wav.tmp")
            sf.write(
                temporary_audio,
                audio,
                SAMPLE_RATE,
                subtype="PCM_16",
                format="WAV",
            )
            temporary_audio.replace(wav_path)

            duration = len(audio) / SAMPLE_RATE
            sr = SAMPLE_RATE
            generated += 1

        if sr != SAMPLE_RATE:
            raise RuntimeError(f"{wav_path}: expected {SAMPLE_RATE} Hz, got {sr}")

        manifest[source_id] = {
            "source_id": source_id,
            "dataset": "worldsense",
            "variant": "openended_relation_completion",
            "split": row["split"],
            "audio_text": audio_text,
            "audio_path": str(wav_path),
            "sample_rate": sr,
            "duration_sec": round(duration, 6),
            "voice": VOICE,
            "tts_speed": SPEED,
        }

        if i % 25 == 0:
            write_manifest(manifest)

    write_manifest(manifest)

    print("\nexisting:", done)
    print("generated:", generated)
    print("manifest:", MANIFEST_PATH)


if __name__ == "__main__":
    main()
