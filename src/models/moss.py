from src.models.token_text import TokenTextDecoder
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from src.models.base import (
    GenerationResult,
    SpeechReasoningModel,
    generation_stop_reason,
)
from src.models.probing import BoundaryConfirmationTracker, BOUNDARY_VERSION


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOSS_SRC = PROJECT_ROOT / "MOSS-Audio" / "src"

if not (MOSS_SRC / "modeling_moss_audio.py").exists():
    raise FileNotFoundError(f"MOSS source not found: {MOSS_SRC}")

if str(MOSS_SRC) not in sys.path:
    sys.path.insert(0, str(MOSS_SRC))

from modeling_moss_audio import MossAudioModel
from processing_moss_audio import MossAudioProcessor

MODEL_ID = "OpenMOSS-Team/MOSS-Audio-8B-Thinking"


class MOSSModel(SpeechReasoningModel):
    def __init__(
        self,
        model_id: str = MODEL_ID,
    ):
        super().__init__()

        self.model_id = model_id
        self.processor = None
        self.model = None
        self.decoder_layers = None
        self.final_norm = None
        self._hidden_dim = None
        self._tokenizer = None
        self._embed_token = None

    @property
    def model_name(self) -> str:
        return "moss"

    @property
    def hidden_dim(self) -> int:
        if self._hidden_dim is None:
            raise RuntimeError("MOSS must be loaded before accessing hidden_dim")
        return self._hidden_dim

    def _load(self) -> None:
        self.processor = MossAudioProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            enable_time_marker=True,
        )

        self.model = MossAudioModel.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map="cuda:0",
        ).eval()

        self.model.requires_grad_(False)

        self.decoder_layers = self.model.language_model.layers
        self.embed_tokens = self.model.language_model.embed_tokens
        self.final_norm = self.model.language_model.norm
        self._hidden_dim = int(self.model.language_model.config.hidden_size)

        self._tokenizer = getattr(
            self.processor,
            "tokenizer",
            None,
        )

    def _load_audio(
        self,
        audio_path: str,
    ) -> np.ndarray:
        audio, sample_rate = sf.read(
            audio_path,
            dtype="float32",
            always_2d=False,
        )

        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        expected_sr = int(self.processor.config.mel_sr)

        if sample_rate != expected_sr:
            audio = torchaudio.functional.resample(
                torch.from_numpy(np.asarray(audio, dtype=np.float32)),
                orig_freq=sample_rate,
                new_freq=expected_sr,
            ).numpy()

        return np.asarray(
            audio,
            dtype=np.float32,
        )

    def _prepare_inputs(
        self,
        audio_path: str,
        prompt_text: str,
    ):
        from src.models.audio_delivery import validate_audio_delivery

        validate_audio_delivery(audio_path, "moss")
        audio = self._load_audio(audio_path)

        inputs = self.processor(
            text=prompt_text,
            audios=[audio],
            return_tensors="pt",
        )

        input_device = next(self.model.language_model.parameters()).device

        inputs = inputs.to(input_device)

        if inputs.get("audio_data") is not None:
            inputs["audio_data"] = inputs["audio_data"].to(
                device=input_device,
                dtype=self.model.dtype,
            )

        inputs["audio_input_mask"] = (
            inputs["input_ids"] == self.processor.audio_token_id
        )

        return inputs

    def _decode_token_piece(
        self,
        token_id: int,
    ) -> str:
        if self._tokenizer is not None:
            return self._tokenizer.decode(
                [int(token_id)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

        # Fallback for custom MOSS processors that expose batch_decode
        # but not a public tokenizer attribute.
        token = torch.tensor(
            [[int(token_id)]],
            dtype=torch.long,
        )
        return self.processor.batch_decode(
            token,
            skip_special_tokens=False,
        )[0]

    def _generate_full(
        self,
        audio_path: str,
        prompt_text: str,
        max_new_tokens: int,
        checkpoints: tuple[int, ...],
        capture_sentence_boundaries: bool = False,
    ) -> GenerationResult:

        inputs = self._prepare_inputs(
            audio_path=audio_path,
            prompt_text=prompt_text,
        )

        input_len = int(inputs["input_ids"].shape[1])

        token_decoder = TokenTextDecoder(
            self.tokenizer if self.model_name == "audio-reasoner" else self._tokenizer
        )
        capture = AllLayerSelectiveCapture(
            embed_tokens=self.embed_tokens,
            decoder_layers=self.decoder_layers,
            final_norm=self.final_norm,
            hidden_dim=self.hidden_dim,
            fixed_checkpoints=checkpoints,
            decode_token=token_decoder,
            capture_sentence_boundaries=(capture_sentence_boundaries),
        )

        with capture:
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

        completion = generated[
            :,
            input_len:,
        ]

        token_ids = completion[0].tolist()
        observed_token_ids = capture.decode_token_ids

        if token_ids[: len(observed_token_ids)] != observed_token_ids:
            raise RuntimeError(
                "MOSS embed-token capture is not aligned " "with generated token IDs"
            )
        text = self.processor.batch_decode(
            completion,
            skip_special_tokens=True,
        )[0]

        num_generated_tokens = len(token_ids)

        if set(capture.all_layers) != set(capture.final_norm_states):
            raise RuntimeError(
                "MOSS all_layers/final_norm " "capture positions diverged"
            )

        stored_positions = set(capture.all_layers)

        expected_positions = set(capture.fixed_positions) | set(
            capture.sentence_positions
        )

        if stored_positions != expected_positions:
            raise RuntimeError(
                "MOSS stored positions do not " "match fixed/sentence checkpoint union"
            )

        stop_reason = generation_stop_reason(
            token_ids,
            max_new_tokens,
            self.model.generation_config.eos_token_id,
        )

        return GenerationResult(
            num_layers=capture.num_layers,
            hidden_size=self.hidden_dim,
            consumed_generated_tokens=capture.decode_step,
            pending_boundary=(
                dict(capture.boundary_tracker.pending)
                if capture_sentence_boundaries and capture.boundary_tracker.pending
                else None
            ),
            tokenizer_fingerprint=token_decoder.fingerprint,
            boundary_version=BOUNDARY_VERSION if capture_sentence_boundaries else None,
            boundary_events=(
                tuple(capture.boundary_tracker.events)
                if capture_sentence_boundaries
                else ()
            ),
            monitoring_stop=(
                capture.boundary_tracker.stop_event
                if capture_sentence_boundaries
                else None
            ),
            boundary_exclusions=(
                capture.boundary_tracker.exclusion_counts
                if capture_sentence_boundaries
                else {}
            ),
            model_name=self.model_name,
            token_ids=token_ids,
            text=text,
            num_generated_tokens=num_generated_tokens,
            stop_reason=stop_reason,
            all_layers=capture.all_layers,
            final_norm=capture.final_norm_states,
            fixed_checkpoints=tuple(capture.fixed_positions),
            sentence_checkpoints=tuple(capture.sentence_positions),
        )


def tensor_from_output(output):
    if torch.is_tensor(output):
        return output

    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item) and item.ndim >= 2:
                return item

    if hasattr(output, "last_hidden_state"):
        hidden = output.last_hidden_state

        if torch.is_tensor(hidden):
            return hidden

    return None


class AllLayerSelectiveCapture:
    """
    Capture all decoder-layer hidden states only at requested
    fixed-token or sentence-boundary positions.

    Prompt prefill is ignored. Generated-token positions are
    1-based and tracked from [1, 1] embed_tokens calls.
    """

    def __init__(
        self,
        embed_tokens,
        decoder_layers,
        final_norm,
        hidden_dim: int,
        fixed_checkpoints: tuple[int, ...],
        decode_token,
        capture_sentence_boundaries: bool,
    ):
        self.embed_tokens = embed_tokens
        self.decoder_layers = decoder_layers
        self.final_norm = final_norm

        self.hidden_dim = hidden_dim
        self.num_layers = len(decoder_layers)

        self.fixed_checkpoint_set = set(fixed_checkpoints)

        self.boundary_tracker = (
            BoundaryConfirmationTracker(
                decode_token=decode_token,
                native_stop_token_ids=getattr(
                    decode_token, "native_stop_token_ids", {}
                ),
                model_name="moss",
            )
            if capture_sentence_boundaries
            else None
        )

        self.decode_step = 0
        self.current_position = None
        self.capture_this_step = False

        # Useful for verifying token alignment against generate().
        self.decode_token_ids = []

        self.fixed_positions = []
        self.sentence_positions = []

        # position -> [L, H]
        self.all_layers = {}

        # position -> [H]
        self.final_norm_states = {}

        # Temporary GPU references for one decoding step.
        self.current_layer_states = {}

        self.handles = []

    def _embed_pre_hook(
        self,
        module,
        args,
    ):
        self.current_position = None
        self.capture_this_step = False
        self.current_layer_states.clear()

        if not args or not torch.is_tensor(args[0]):
            return

        token_ids = args[0]

        # Prompt prefill is [1, T].
        # Autoregressive decode steps are [1, 1].
        if not (token_ids.ndim == 2 and tuple(token_ids.shape) == (1, 1)):
            return

        self.decode_step += 1
        self.current_position = self.decode_step

        token_id = int(token_ids[0, 0].item())

        self.decode_token_ids.append(token_id)

        is_fixed = self.current_position in self.fixed_checkpoint_set

        is_sentence = False

        if self.boundary_tracker is not None:
            is_sentence = self.boundary_tracker.observe(token_id)

        if is_fixed:
            self.fixed_positions.append(self.current_position)

        if is_sentence:
            self.sentence_positions.append(self.current_position)

        self.capture_this_step = is_fixed or is_sentence

    def _extract_state(
        self,
        output,
    ) -> torch.Tensor:
        hidden = tensor_from_output(output)

        if hidden is None:
            raise RuntimeError("Could not extract hidden state")

        if hidden.ndim == 3:
            hidden = hidden[0, -1, :]

        elif hidden.ndim == 2:
            hidden = hidden[-1, :]

        else:
            raise RuntimeError("Unexpected hidden shape: " f"{tuple(hidden.shape)}")

        if hidden.ndim != 1 or hidden.shape[0] != self.hidden_dim:
            raise RuntimeError(
                "Unexpected MOSS hidden-state shape: " f"{tuple(hidden.shape)}"
            )

        return hidden.detach()

    def _make_layer_hook(
        self,
        layer_index: int,
    ):
        def hook(
            module,
            inputs,
            output,
        ):
            if self.current_position is None or not self.capture_this_step:
                return

            if layer_index in self.current_layer_states:
                return

            self.current_layer_states[layer_index] = self._extract_state(output)

        return hook

    def _norm_hook(
        self,
        module,
        inputs,
        output,
    ):
        if self.current_position is None or not self.capture_this_step:
            return

        if self.current_position in self.final_norm_states:
            return

        if len(self.current_layer_states) != self.num_layers:
            missing = sorted(
                set(range(self.num_layers)) - set(self.current_layer_states)
            )

            raise RuntimeError(
                "MOSS did not capture every "
                "decoder layer at position "
                f"{self.current_position}; "
                f"missing={missing}"
            )

        # [L, H]
        stacked = torch.stack(
            [
                self.current_layer_states[layer_index]
                for layer_index in range(self.num_layers)
            ],
            dim=0,
        )

        self.all_layers[self.current_position] = stacked.float().cpu()

        self.final_norm_states[self.current_position] = (
            self._extract_state(output).float().cpu()
        )

    def __enter__(self):
        self.handles = [
            self.embed_tokens.register_forward_pre_hook(self._embed_pre_hook)
        ]

        self.handles.extend(
            layer.register_forward_hook(self._make_layer_hook(layer_index))
            for layer_index, layer in enumerate(self.decoder_layers)
        )

        self.handles.append(self.final_norm.register_forward_hook(self._norm_hook))

        return self

    def __exit__(
        self,
        exc_type,
        exc,
        tb,
    ):
        for handle in self.handles:
            handle.remove()

        self.handles = []
