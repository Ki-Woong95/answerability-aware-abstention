from dataclasses import dataclass
from typing import Literal

from src.models.base import GenerationResult


@dataclass
class Trajectory:
    trial_id: str
    pair_id: str
    condition: Literal["answerable", "unanswerable"]
    split: Literal["gate_train", "calibration", "val", "test"]

    source: str
    category: int | None

    voice: str
    audio_path: str
    raw_text: str
    tts_text: str
    prompt_text: str

    gold_answer: tuple[str | int | float, ...]
    generation: GenerationResult

    @property
    def model_name(self) -> str:
        return self.generation.model_name

    @property
    def checkpoints(self) -> tuple[int, ...]:
        return tuple(sorted(self.generation.all_layers.keys()))

    @property
    def fixed_checkpoints(self) -> tuple[int, ...]:
        return self.generation.fixed_checkpoints

    @property
    def sentence_checkpoints(self) -> tuple[int, ...]:
        return self.generation.sentence_checkpoints

    @property
    def num_generated_tokens(self) -> int:
        return self.generation.num_generated_tokens

    @property
    def stop_reason(self) -> str:
        return self.generation.stop_reason

    def __post_init__(self):
        if not isinstance(self.trial_id, str) or not self.trial_id.strip():
            raise ValueError("trial_id must be a non-empty string")

        if not isinstance(self.pair_id, str) or not self.pair_id.strip():
            raise ValueError("pair_id must be a non-empty string")

        if self.condition not in ("answerable", "unanswerable"):
            raise ValueError("condition must be 'answerable' or 'unanswerable'")

        if self.split not in ("gate_train", "calibration", "val", "test"):
            raise ValueError("split must be gate_train, calibration, val, or test")

        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a non-empty string")

        if self.category is not None:
            if not isinstance(self.category, int) or self.category not in (
                1,
                2,
                3,
                4,
                5,
            ):
                raise ValueError("category must be None or an integer from 1 to 5")

        if not isinstance(self.voice, str) or not self.voice.strip():
            raise ValueError("voice must be a non-empty string")

        if not isinstance(self.audio_path, str) or not self.audio_path.strip():
            raise ValueError("audio_path must be a non-empty string")

        if not isinstance(self.raw_text, str) or not self.raw_text.strip():
            raise ValueError("raw_text must be a non-empty string")

        if not isinstance(self.tts_text, str) or not self.tts_text.strip():
            raise ValueError("tts_text must be a non-empty string")

        if not isinstance(self.prompt_text, str) or not self.prompt_text.strip():
            raise ValueError("prompt_text must be a non-empty string")

        if not isinstance(self.gold_answer, tuple):
            raise ValueError("gold_answer must be a tuple")
