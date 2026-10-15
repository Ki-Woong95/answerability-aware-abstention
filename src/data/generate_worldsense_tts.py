"""Compatibility entrypoint for the current open-ended WorldSense TTS builder."""

from src.data.build_worldsense_tts import (
    PAIR_PATH,
    MANIFEST_PATH,
    VOICE,
    SPEED,
    SAMPLE_RATE,
    read_jsonl,
    synthesize,
    load_manifest,
    write_manifest,
    main,
)


if __name__ == "__main__":
    main()
