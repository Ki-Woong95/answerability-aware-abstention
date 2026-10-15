from src.models.token_text import TokenTextDecoder
import torch

from src.models.base import (
    GenerationResult,
    SpeechReasoningModel,
    generation_stop_reason,
)
from src.models.probing import BoundaryConfirmationTracker, BOUNDARY_VERSION

from transformers import AutoModelForSeq2SeqLM, AutoProcessor


class AFNextModel(SpeechReasoningModel):
    def __init__(
        self,
        model_id: str,
        prompt_template: str,
    ):
        super().__init__()
        self.model_id = model_id
        self.prompt_template = prompt_template

        self.model = None
        self.processor = None
        self._hidden_dim = None
        self._decoder_layers = None
        self._final_norm = None
        self._tokenizer = None
        self._embed_tokens = None

    def _prepare_batch(
        self,
        audio_path: str,
        prompt_text: str,
    ):
        from src.models.audio_delivery import validate_audio_delivery

        validate_audio_delivery(audio_path, "af_next")
        prompt = self.prompt_template.format(question=prompt_text)
        conversation = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "audio",
                            "path": str(audio_path),
                        },
                    ],
                }
            ]
        ]
        batch = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )
        batch = batch.to(self.model.device)

        if "input_features" in batch:
            batch["input_features"] = batch["input_features"].to(self.model.dtype)
        return batch

    @property
    def model_name(self) -> str:
        return "af_next"

    @property
    def hidden_dim(self) -> int:
        if self._hidden_dim is None:
            raise RuntimeError("AF-Next must be loaded before accessing hidden_dim")
        return self._hidden_dim

    def _load(self) -> None:
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            device_map="auto",
        ).eval()

        self.model.requires_grad_(False)

        language_model = self.model.model.language_model

        self._hidden_dim = self.model.config.text_config.hidden_size
        self._decoder_layers = language_model.layers
        self._final_norm = language_model.norm
        self._embed_tokens = language_model.embed_tokens

        self._tokenizer = getattr(
            self.processor,
            "tokenizer",
            None,
        )
        if self._tokenizer is None:
            raise RuntimeError("Could not resolve AF-Next tokenizer from processor")

    def _decode_token_piece(
        self,
        token_id: int,
    ) -> str:
        return self._tokenizer.decode(
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

        batch = self._prepare_batch(
            audio_path=audio_path,
            prompt_text=prompt_text,
        )

        input_len = int(batch["input_ids"].shape[1])

        fixed_checkpoint_set = set(checkpoints)

        token_decoder = (
            TokenTextDecoder(self._tokenizer, errors="replace")
            if getattr(self, "token_text_errors", "strict") == "replace"
            else TokenTextDecoder(self._tokenizer)
        )
        self.last_token_text_decoder = token_decoder
        boundary_tracker = BoundaryConfirmationTracker(
            decode_token=token_decoder, model_name=self.model_name
        )

        num_layers = len(self._decoder_layers)

        # Only probe positions are stored.
        # all_layers_states[position] -> [L, H]
        all_layers_states: dict[int, torch.Tensor] = {}
        final_norm_states: dict[int, torch.Tensor] = {}

        fixed_positions: list[int] = []
        sentence_positions: list[int] = []

        decode_step = 0
        consumed_token_ids = []
        current_position: int | None = None
        capture_this_step = False

        # Temporary GPU references for ONE decoding step only.
        current_layer_states: dict[int, torch.Tensor] = {}

        def embed_pre_hook(
            module,
            inputs,
        ):
            nonlocal decode_step
            nonlocal current_position
            nonlocal capture_this_step

            # Start a fresh decoding step.
            current_layer_states.clear()
            capture_this_step = False
            current_position = None

            if not inputs:
                return

            token_ids = inputs[0]

            if not isinstance(token_ids, torch.Tensor) or token_ids.numel() != 1:
                # Prompt prefill.
                return

            # This token is a previously generated token being
            # fed back into the decoder. Its resulting hidden state
            # is aligned with this generated-token position.
            decode_step += 1
            current_position = decode_step

            token_id = int(token_ids.reshape(-1)[0].item())

            consumed_token_ids.append(token_id)

            is_fixed = current_position in fixed_checkpoint_set

            is_sentence = False

            if capture_sentence_boundaries:
                is_sentence = boundary_tracker.observe(token_id)

            if is_fixed:
                fixed_positions.append(current_position)

            if is_sentence:
                sentence_positions.append(current_position)

            capture_this_step = is_fixed or is_sentence

        def make_layer_hook(
            layer_index: int,
        ):
            def layer_hook(
                module,
                inputs,
                output,
            ):
                if not capture_this_step:
                    return

                if current_position is None:
                    raise RuntimeError(
                        "AF-Next capture step has " "no current position"
                    )

                current_layer_states[layer_index] = output[0, -1, :].detach()

            return layer_hook

        def final_norm_hook(
            module,
            inputs,
            output,
        ):
            if not capture_this_step:
                return

            if current_position is None:
                raise RuntimeError(
                    "AF-Next final_norm capture has " "no current position"
                )

            if len(current_layer_states) != num_layers:
                missing = sorted(set(range(num_layers)) - set(current_layer_states))

                raise RuntimeError(
                    "AF-Next did not capture every "
                    "decoder layer at position "
                    f"{current_position}; "
                    f"missing={missing}"
                )

            # [L, H]
            stacked = torch.stack(
                [
                    current_layer_states[layer_index]
                    for layer_index in range(num_layers)
                ],
                dim=0,
            )

            # One GPU -> CPU transfer for all decoder layers.
            all_layers_states[current_position] = stacked.float().cpu()

            final_norm_states[current_position] = (
                output[0, -1, :].detach().float().cpu()
            )

        handles = [self._embed_tokens.register_forward_pre_hook(embed_pre_hook)]

        handles.extend(
            layer.register_forward_hook(make_layer_hook(layer_index))
            for layer_index, layer in enumerate(self._decoder_layers)
        )

        handles.append(self._final_norm.register_forward_hook(final_norm_hook))

        try:
            with torch.inference_mode():
                generated = self.model.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=1.2,
                    do_sample=False,
                )

        finally:
            for handle in handles:
                handle.remove()

        completion = generated[
            :,
            input_len:,
        ]

        token_ids = completion[0].tolist()
        if token_ids[: len(consumed_token_ids)] != consumed_token_ids:
            raise RuntimeError("AF-Next consumed/generated token prefixes disagree")

        text = self.processor.batch_decode(
            completion,
            skip_special_tokens=True,
        )[0]

        num_generated_tokens = len(token_ids)

        if set(all_layers_states) != set(final_norm_states):
            raise RuntimeError(
                "AF-Next all_layers/final_norm " "capture positions diverged"
            )

        stored_positions = set(all_layers_states)

        expected_positions = set(fixed_positions) | set(sentence_positions)

        if stored_positions != expected_positions:
            raise RuntimeError(
                "AF-Next stored positions do not "
                "match fixed/sentence checkpoint union"
            )

        stop_reason = generation_stop_reason(
            token_ids,
            max_new_tokens,
            self.model.generation_config.eos_token_id,
        )

        return GenerationResult(
            num_layers=num_layers,
            hidden_size=self.hidden_dim,
            consumed_generated_tokens=decode_step,
            pending_boundary=(
                dict(boundary_tracker.pending)
                if capture_sentence_boundaries and boundary_tracker.pending
                else None
            ),
            tokenizer_fingerprint=token_decoder.fingerprint,
            boundary_version=BOUNDARY_VERSION if capture_sentence_boundaries else None,
            boundary_events=(
                tuple(boundary_tracker.events) if capture_sentence_boundaries else ()
            ),
            monitoring_stop=(
                boundary_tracker.stop_event if capture_sentence_boundaries else None
            ),
            boundary_exclusions=(
                boundary_tracker.exclusion_counts if capture_sentence_boundaries else {}
            ),
            model_name=self.model_name,
            token_ids=token_ids,
            text=text,
            num_generated_tokens=(num_generated_tokens),
            stop_reason=stop_reason,
            all_layers=all_layers_states,
            final_norm=final_norm_states,
            fixed_checkpoints=tuple(fixed_positions),
            sentence_checkpoints=tuple(sentence_positions),
        )
