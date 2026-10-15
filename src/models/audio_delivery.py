"""Fail before a backend can silently truncate a recording."""

import math
import soundfile as sf


def validate_audio_delivery(audio_path, model_name):
    info = sf.info(str(audio_path))
    if info.frames <= 0 or info.samplerate <= 0:
        raise ValueError("Empty or invalid audio recording")
    if model_name == "audio-reasoner":
        # librosa.load(sr=16000) uses fix=True: ceil of resampled length.
        expected_samples = math.ceil(info.frames * 16000 / info.samplerate)
        if expected_samples > 480000:
            raise ValueError(
                "Audio-Reasoner input exceeds 480000 samples; refusing silent 30-second truncation"
            )
    elif model_name == "af_next":
        if info.frames > info.samplerate * 1800:
            raise ValueError("AF-Next input exceeds audited full-audio limit")
    elif model_name != "moss":
        raise ValueError(model_name)
    return info
