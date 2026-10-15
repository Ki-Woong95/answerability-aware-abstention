"""Text normalization shared by UMWP TTS generation and capture validation."""

import re


def normalize_tts_text(text):
    def existing_dollars(match):
        return f"{match.group(1)} {match.group(2)}"

    def bare_dollars(match):
        number = match.group(1)
        try:
            singular = float(number.replace(",", "")) == 1
        except ValueError:
            singular = False
        return f"{number} {'dollar' if singular else 'dollars'}"

    text = re.sub(
        r"\$\s*([0-9][0-9,]*(?:\.\d+)?)\s+(dollars?)\b",
        existing_dollars,
        text,
        flags=re.I,
    )
    text = re.sub(r"\$\s*([0-9][0-9,]*(?:\.\d+)?)", bare_dollars, text)
    text = re.sub(r"([.!?])(?=[A-Za-z])", r"\1 ", text)
    return re.sub(r"\s+", " ", text).strip()
