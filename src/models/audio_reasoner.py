from src.models.token_text import TokenTextDecoder
import torch

from src.models.base import (
    COMPLETE_TOKEN_CAPTURE,
    SpeechReasoningModel,
    GenerationResult,
)
from src.models.probing import BoundaryConfirmationTracker, BOUNDARY_VERSION
from src.models.audio_reasoner_stop import (
    STOP_POLICY,
    classify_stop,
    request_stop_sequences,
)


AR_SYSTEM = (
    "You are an audio deep-thinking model. Upon receiving a question, "
    "please respond in two parts: <THINK> and <RESPONSE>. "
    "The <THINK> section should be further divided into four parts: "
    "<PLANNING>, <CAPTION>, <REASONING>, and <SUMMARY>."
)


def tensor_from_output(output):
    if torch.is_tensor(output):
        return output

    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item) and item.ndim >= 2:
                return item

    if hasattr(output, "last_hidden_state"):
        item = output.last_hidden_state

        if torch.is_tensor(item):
            return item

    return None


def resolve_ptengine_tokenizer(engine):
    candidates = []

    tokenizer = getattr(engine, "tokenizer", None)
    if tokenizer is not None:
        candidates.append(tokenizer)

    processor = getattr(engine, "processor", None)
    if processor is not None and getattr(processor, "tokenizer", None) is not None:
        candidates.append(processor.tokenizer)

    template = getattr(engine, "template", None)
    if template is not None:
        tokenizer = getattr(template, "tokenizer", None)

        if tokenizer is not None:
            candidates.append(tokenizer)

        processor = getattr(template, "processor", None)

        if processor is not None and getattr(processor, "tokenizer", None) is not None:
            candidates.append(processor.tokenizer)

    for tokenizer in candidates:
        if hasattr(tokenizer, "decode") and hasattr(tokenizer, "encode"):
            return tokenizer

    raise RuntimeError("Could not resolve tokenizer from PtEngine")


class GeneratedTokenCapture:
    """Record the exact HF output tokens while preserving PtEngine's streamer.

    HF streams the prompt first, then every emitted token, including the last
    token that is never fed back through embed_tokens. PtEngine runs generate
    in a worker thread, so recording happens before forwarding to its queue.
    """

    def __init__(self, model):
        self.model = model
        self.token_ids = []
        self.streamer = None
        self.prompt_seen = False

    def put(self, value):
        if self.prompt_seen:
            self.token_ids.extend(value.reshape(-1).tolist())
        else:
            self.prompt_seen = True
        self.streamer.put(value)

    def end(self):
        self.streamer.end()

    def __enter__(self):
        self.original_generate = self.model.generate
        # Restore the original instance attribute exactly, including an absent
        # attribute when generate is normally a class method.
        self.instance_generate = self.model.__dict__.get("generate")

        def generate(*args, **kwargs):
            self.streamer = kwargs.get("streamer")
            if self.streamer is None:
                raise RuntimeError(
                    "Audio-Reasoner capture requires streaming generation"
                )
            kwargs["streamer"] = self
            return self.original_generate(*args, **kwargs)

        self.model.generate = generate
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.instance_generate is None:
            del self.model.generate
        else:
            self.model.generate = self.instance_generate


class AllLayerSelectiveCapture:
    """
    Capture all decoder-layer hidden states only at requested
    fixed-token or sentence-boundary positions.

    Generated-token positions are 1-based and are tracked from
    [1, 1] embed_tokens calls. Prompt prefill is ignored.
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

        self.capture_sentence_boundaries = capture_sentence_boundaries

        self.boundary_tracker = (
            BoundaryConfirmationTracker(
                decode_token=decode_token,
                native_stop_token_ids=getattr(
                    decode_token, "native_stop_token_ids", {}
                ),
                model_name="audio-reasoner",
            )
            if capture_sentence_boundaries
            else None
        )

        self.decode_step = 0
        self.current_position = None
        self.decode_token_ids = []

        self.capture_this_step = False

        self.fixed_positions = []
        self.sentence_positions = []

        # position -> [L, H]
        self.all_layers = {}

        # position -> [H]
        self.final_norm_states = {}

        # Temporary GPU references for one decode step.
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
        # Generated-token forward is [1, 1].
        if not (token_ids.ndim == 2 and tuple(token_ids.shape) == (1, 1)):
            return

        self.decode_step += 1

        token_id = int(token_ids[0, 0].item())

        self.decode_token_ids.append(token_id)
        self.current_position = self.decode_step

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
                "Unexpected Audio-Reasoner "
                "hidden-state shape: "
                f"{tuple(hidden.shape)}"
            )

        # Keep on GPU temporarily.
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

            # Defensive: do not overwrite if the same
            # layer hook is unexpectedly called twice.
            if layer_index in self.current_layer_states:
                return

            self.current_layer_states[layer_index] = self._extract_state(output)

        return hook

    def _final_norm_hook(
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
                "Audio-Reasoner did not capture "
                "every decoder layer at position "
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

        # One GPU -> CPU transfer for all layers.
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

        self.handles.append(
            self.final_norm.register_forward_hook(self._final_norm_hook)
        )

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


class AudioReasonerModel(SpeechReasoningModel):
    def __init__(
        self,
        model_id: str = "zhifeixie/Audio-Reasoner",
        generation_seed: int = 20260921,
    ):
        super().__init__()

        self.model_id = model_id
        self.generation_seed = generation_seed

        self.engine = None
        self.decoder = None
        self.decoder_layers = None
        self.embed_tokens = None
        self.final_norm = None
        self.tokenizer = None

        self._hidden_dim = None

        self.InferRequest = None
        self.RequestConfig = None

    @property
    def model_name(self) -> str:
        return "audio-reasoner"

    @property
    def hidden_dim(self) -> int:
        if self._hidden_dim is None:
            raise RuntimeError(
                "Audio-reasoner must be loaded before accessing hidden dim"
            )

        return self._hidden_dim

    def _load(self) -> None:
        from src.models.local_snapshot import resolve_audio_reasoner_snapshot

        self.model_id = resolve_audio_reasoner_snapshot(self.model_id)
        from swift.llm import InferRequest, PtEngine, RequestConfig

        self.InferRequest = InferRequest
        self.RequestConfig = RequestConfig
        self.engine = PtEngine(
            self.model_id,
            max_batch_size=1,
            model_type="qwen2_audio",
        )

        self.decoder = self.engine.model.language_model.model
        self.decoder_layers = self.decoder.layers
        self.embed_tokens = self.decoder.embed_tokens
        self.final_norm = self.decoder.norm

        self._hidden_dim = int(self.engine.model.language_model.config.hidden_size)
        self.tokenizer = resolve_ptengine_tokenizer(self.engine)

    def _messages(
        self,
        audio_path: str,
        prompt_text: str,
    ):
        from src.models.audio_delivery import validate_audio_delivery

        validate_audio_delivery(audio_path, "audio-reasoner")
        return [
            {
                "role": "system",
                "content": AR_SYSTEM,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio": str(audio_path),
                    },
                    {
                        "type": "text",
                        "text": prompt_text,
                    },
                ],
            },
        ]

    def _decode_token_piece(
        self,
        token_id: int,
    ) -> str:
        return self.tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def _generate_full(
        self,
        audio_path: str,
        prompt_text: str,
        max_new_tokens: int,
        checkpoints: tuple[int, ...],
        capture_sentence_boundaries: bool = False,
    ) -> GenerationResult:

        request = self.InferRequest(
            messages=self._messages(
                audio_path=audio_path,
                prompt_text=prompt_text,
            )
        )
        config = self.RequestConfig(
            max_tokens=max_new_tokens,
            temperature=0,
            stream=True,
            seed=self.generation_seed,
            stop=request_stop_sequences(),
        )

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

        text = ""

        with capture, GeneratedTokenCapture(self.engine.model) as generated_tokens:
            generation = self.engine.infer(
                [request],
                config,
            )

            for response_list in generation:
                if response_list[0] is None:
                    continue

                delta = response_list[0].choices[0].delta.content

                if delta:
                    text += delta

        token_ids = list(generated_tokens.token_ids)
        observed_token_ids = capture.decode_token_ids
        if not token_ids:
            raise RuntimeError("Audio-Reasoner generated no tokens")
        if token_ids[: len(observed_token_ids)] != observed_token_ids:
            raise RuntimeError(
                "Audio-Reasoner embed-token capture is not aligned with generated token IDs"
            )
        num_generated_tokens = len(token_ids)

        if set(capture.all_layers) != set(capture.final_norm_states):
            raise RuntimeError(
                "Audio-Reasoner all_layers/final_norm " "capture positions diverged"
            )

        stored_positions = set(capture.all_layers)

        expected_positions = set(capture.fixed_positions) | set(
            capture.sentence_positions
        )

        if stored_positions != expected_positions:
            raise RuntimeError(
                "Audio-Reasoner stored positions do not "
                "match fixed/sentence checkpoint union"
            )

        eos_ids = getattr(self.engine.model.generation_config, "eos_token_id", None)
        if eos_ids is None:
            eos_ids = getattr(self.tokenizer, "eos_token_id", None)
        stop_reason, stop_marker = classify_stop(
            token_ids, text, max_new_tokens, eos_ids, self.tokenizer
        )

        return GenerationResult(
            generation_stop_policy=STOP_POLICY,
            generation_stop_marker=stop_marker,
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
            token_capture_method=COMPLETE_TOKEN_CAPTURE,
        )
