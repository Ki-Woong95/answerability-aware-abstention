"""Formatting diagnostics only. Raw responses remain authoritative judge inputs."""

import re


def inspect_response_format(text, model_name):
    cleaned = text
    closers = ["<\\|im_end\\|>", "<\\|endoftext\\|>"]
    if model_name == "audio-reasoner":
        closers += ["</RESPONSE>"]
    # Only remove known trailing wrapper delimiters, not arbitrary angle brackets.
    pattern = r"(?:\s*(?:" + "|".join(closers) + r"))+\s*$"
    cleaned = re.sub(pattern, "", cleaned, flags=re.I)
    finals = list(re.finditer(r"(?im)^[ \t]*FINAL[ \t]*:[ \t]*([^\r\n]*)", cleaned))
    reasons = list(re.finditer(r"(?im)^[ \t]*REASON[ \t]*:[ \t]*([^\r\n]+)", cleaned))
    answer = finals[-1].group(1).strip() if finals else None
    answer = answer or None
    reason_present = bool(
        finals and any(r.end() <= finals[-1].start() for r in reasons)
    )
    abstains = bool(answer and re.fullmatch(r"I don't know[.!]?", answer, re.I))
    terminal = bool(finals and not cleaned[finals[-1].end() :].strip())
    return dict(
        final_text=answer,
        reason_present=reason_present,
        format_ok=bool(
            len(finals) == 1
            and answer
            and terminal
            and (not abstains or reason_present)
        ),
    )
