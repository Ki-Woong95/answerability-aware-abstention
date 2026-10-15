from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch


COMPLETE_TOKEN_CAPTURE = "generated_stream_v1"


def generation_stop_reason(token_ids, budget, eos_token_ids, stopped_reason="eos"):
    """EOS takes precedence when it is emitted on the last allowed step."""
    eos_ids = (
        eos_token_ids
        if isinstance(eos_token_ids, (list, tuple, set))
        else [eos_token_ids]
    )
    if token_ids and token_ids[-1] in eos_ids:
        return "eos"
    return "max_new_tokens" if len(token_ids) >= budget else stopped_reason


@dataclass
class GenerationResult:
    model_name: str
    token_ids: list[int]
    text: str
    num_generated_tokens: int
    stop_reason: str

    # checkpoint -> [num_layers, hidden_dim]
    all_layers: dict[int, torch.Tensor]

    # checkpoint -> [hidden_dim]
    final_norm: dict[int, torch.Tensor]

    fixed_checkpoints: tuple[int, ...] = field(default_factory=tuple)
    sentence_checkpoints: tuple[int, ...] = field(default_factory=tuple)
    token_capture_method: str | None = None
    num_layers: int = 0
    hidden_size: int = 0
    boundary_events: tuple = field(default_factory=tuple)
    monitoring_stop: dict | None = None
    boundary_version: str | None = None
    tokenizer_fingerprint: str | None = None
    boundary_exclusions: dict = field(default_factory=dict)
    pending_boundary: dict | None = None
    consumed_generated_tokens: int | None = None
    generation_stop_policy: str | None = None
    generation_stop_marker: str | None = None

    @staticmethod
    def _validate_checkpoint_tuple(
        checkpoints: tuple[int, ...],
        name: str,
    ) -> None:

        if any(checkpoint <= 0 for checkpoint in checkpoints):
            raise ValueError(f"{name} must use positive " "1-based token indices")

        if any(
            current <= previous
            for previous, current in zip(
                checkpoints,
                checkpoints[1:],
            )
        ):
            raise ValueError(f"{name} must be strictly increasing")

    def __post_init__(self):

        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must be a non-empty string")

        if self.num_generated_tokens != len(self.token_ids):
            raise ValueError("num_generated_tokens must equal " "len(token_ids)")

        if self.all_layers.keys() != self.final_norm.keys():
            raise ValueError(
                "all_layers and final_norm must " "contain the same checkpoints"
            )

        if self.boundary_version:
            if self.num_layers <= 0 or self.hidden_size <= 0:
                raise ValueError(
                    "V10 capture requires known layer and hidden dimensions even with zero states"
                )
            if (
                tuple(e["confirmation_token_index"] for e in self.boundary_events)
                != self.sentence_checkpoints
            ):
                raise ValueError(
                    "Boundary events must align exactly with saved confirmation states"
                )
        num_layers = None

        for checkpoint, states in self.all_layers.items():

            if checkpoint <= 0:
                raise ValueError(
                    "checkpoints must use positive " "1-based token indices"
                )

            if checkpoint > self.num_generated_tokens:
                raise ValueError("checkpoint cannot exceed " "num_generated_tokens")

            if states.ndim != 2:
                raise ValueError(
                    "all_layers state at checkpoint " f"{checkpoint} must be 2D [L, H]"
                )

            if self.boundary_version and tuple(states.shape) != (
                self.num_layers,
                self.hidden_size,
            ):
                raise ValueError(
                    "V10 hidden-state dimensions disagree with capture configuration"
                )
            if states.shape[0] <= 0:
                raise ValueError(
                    "all_layers must contain at least " "one decoder layer"
                )

            if num_layers is None:
                num_layers = states.shape[0]
            elif states.shape[0] != num_layers:
                raise ValueError(
                    "number of decoder layers must " "be consistent across checkpoints"
                )

            final_state = self.final_norm[checkpoint]

            if final_state.ndim != 1:
                raise ValueError(
                    "final_norm state at checkpoint " f"{checkpoint} must be 1D"
                )

            if states.shape[1] != final_state.shape[0]:
                raise ValueError(
                    "all_layers hidden dimension and "
                    "final_norm dimension must match "
                    f"at checkpoint {checkpoint}"
                )

        self._validate_checkpoint_tuple(
            self.fixed_checkpoints,
            "fixed_checkpoints",
        )

        self._validate_checkpoint_tuple(
            self.sentence_checkpoints,
            "sentence_checkpoints",
        )

        state_positions = set(self.all_layers)

        probe_positions = set(self.fixed_checkpoints) | set(self.sentence_checkpoints)

        if probe_positions and probe_positions != state_positions:
            raise ValueError(
                "stored state checkpoints must equal "
                "the union of fixed_checkpoints and "
                "sentence_checkpoints"
            )

        if any(
            checkpoint > self.num_generated_tokens for checkpoint in probe_positions
        ):
            raise ValueError("probe checkpoint cannot exceed " "num_generated_tokens")


class SpeechReasoningModel(ABC):

    def __init__(self):
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @property
    @abstractmethod
    def hidden_dim(self) -> int:
        pass

    @abstractmethod
    def _load(self) -> None:
        pass

    def load(self) -> None:

        if self._is_loaded:
            return

        self._load()
        self._is_loaded = True

    def _validate_result(
        self,
        result: GenerationResult,
    ) -> None:

        if result.model_name != self.model_name:
            raise ValueError(
                f"result model_name "
                f"'{result.model_name}' does not "
                f"match backend model_name "
                f"'{self.model_name}'"
            )

        for checkpoint, states in result.all_layers.items():
            if states.shape[1] != self.hidden_dim:
                raise ValueError(
                    "all_layers state at checkpoint "
                    f"{checkpoint} has hidden "
                    f"dimension {states.shape[1]}, "
                    f"expected {self.hidden_dim}"
                )

        for checkpoint, state in result.final_norm.items():
            if state.shape[0] != self.hidden_dim:
                raise ValueError(
                    "final_norm state at checkpoint "
                    f"{checkpoint} has dimension "
                    f"{state.shape[0]}, expected "
                    f"{self.hidden_dim}"
                )

    @abstractmethod
    def _generate_full(
        self,
        audio_path: str,
        prompt_text: str,
        max_new_tokens: int,
        checkpoints: tuple[int, ...],
        capture_sentence_boundaries: bool = False,
    ) -> GenerationResult:
        """
        Checkpoint k refers to the hidden state aligned
        with the k-th generated token, using 1-based
        generated-token indexing.
        """
        pass

    def generate_full(
        self,
        audio_path: str,
        prompt_text: str,
        max_new_tokens: int,
        checkpoints: tuple[int, ...],
        capture_sentence_boundaries: bool = False,
    ) -> GenerationResult:

        if not self._is_loaded:
            raise RuntimeError(f"{self.model_name} must be loaded " "before generation")

        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")

        if any(checkpoint <= 0 for checkpoint in checkpoints):
            raise ValueError("checkpoints must use positive " "1-based token indices")

        if any(
            current <= previous
            for previous, current in zip(
                checkpoints,
                checkpoints[1:],
            )
        ):
            raise ValueError("checkpoints must be strictly increasing")

        result = self._generate_full(
            audio_path=audio_path,
            prompt_text=prompt_text,
            max_new_tokens=max_new_tokens,
            checkpoints=checkpoints,
            capture_sentence_boundaries=(capture_sentence_boundaries),
        )

        self._validate_result(result)

        return result
