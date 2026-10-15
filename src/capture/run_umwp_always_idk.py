"""UMWP prompt-only abstention ablation using the shared capture pipeline."""

from pathlib import Path
from src.capture import run as capture
from src.capture.run import (
    read_jsonl,
    extract_pair_ids,
    load_split_map,
    load_pair_answers,
    build_model,
    load_completed,
    validate_result,
    select_rows,
)

TASK_PROMPT = """Listen to the audio and reason step by step using only the information provided in the audio.

Assess whether the audio contains sufficient information to answer the question.

Use only information explicitly provided in the audio and do not infer missing facts.

If the answer cannot be uniquely determined from the audio, conclude with:
FINAL: I don't know

Otherwise, conclude with:
FINAL: <answer>"""

DEFAULT_MANIFEST = capture.DEFAULT_MANIFEST
DEFAULT_PAIRS = capture.DEFAULT_PAIRS
DEFAULT_SPLITS = capture.DEFAULT_SPLITS
DEFAULT_OUTPUT = Path("data/ablations/always_idk/umwp/trajectories")


def run(args):
    return capture.run(args, task_prompt=TASK_PROMPT, model_factory=build_model)


def main():
    return capture.main(
        default_output=DEFAULT_OUTPUT,
        task_prompt=TASK_PROMPT,
        model_factory=build_model,
    )


if __name__ == "__main__":
    main()
