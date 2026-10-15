from swift.llm import InferRequest, PtEngine, RequestConfig
from swift.plugin import InferStats
from src.models.audio_reasoner import AR_SYSTEM, GeneratedTokenCapture
from src.models.base import COMPLETE_TOKEN_CAPTURE, generation_stop_reason

MODEL_ID = "zhifeixie/Audio-Reasoner"
MAX_CONTINUATION_TOKENS = 2048


from src.capture.v10_prompt import INTERVENTION_PROMPT, intervention_prompt_provenance
from src.models.audio_reasoner_stop import (
    STOP_POLICY,
    request_stop_sequences,
    classify_stop,
)


class ExactInputTemplate:
    """
    Delegate all normal ms-swift template behavior to the original
    template, but replace the final encoded text input_ids with the
    exact token sequence supplied by us.
    """

    def __init__(
        self,
        base_template,
        exact_input_ids,
    ):
        object.__setattr__(
            self,
            "_base_template",
            base_template,
        )
        object.__setattr__(
            self,
            "_exact_input_ids",
            list(exact_input_ids),
        )

    def __getattr__(self, name):
        return getattr(
            self._base_template,
            name,
        )

    def __setattr__(self, name, value):
        if name in {
            "_base_template",
            "_exact_input_ids",
        }:
            object.__setattr__(
                self,
                name,
                value,
            )
        else:
            setattr(
                self._base_template,
                name,
                value,
            )

    def encode(
        self,
        infer_request,
        *args,
        **kwargs,
    ):
        encoded = self._base_template.encode(
            infer_request,
            *args,
            **kwargs,
        )

        encoded["input_ids"] = list(self._exact_input_ids)

        return encoded


class AudioReasonerInterventionRunner:
    def __init__(self):
        from src.models.local_snapshot import resolve_audio_reasoner_snapshot

        model_path = resolve_audio_reasoner_snapshot(MODEL_ID)
        print(f"Loading Audio-Reasoner: {MODEL_ID}")

        self.engine = PtEngine(
            model_path,
            max_batch_size=1,
            model_type="qwen2_audio",
        )

        self.tokenizer = self._resolve_tokenizer()

    def _resolve_tokenizer(self):
        candidates = [
            getattr(
                self.engine,
                "tokenizer",
                None,
            ),
            getattr(
                getattr(
                    self.engine,
                    "processor",
                    None,
                ),
                "tokenizer",
                None,
            ),
            getattr(
                getattr(
                    self.engine,
                    "template",
                    None,
                ),
                "tokenizer",
                None,
            ),
            getattr(
                getattr(
                    getattr(
                        self.engine,
                        "template",
                        None,
                    ),
                    "processor",
                    None,
                ),
                "tokenizer",
                None,
            ),
        ]

        for tokenizer in candidates:
            if (
                tokenizer is not None
                and hasattr(tokenizer, "encode")
                and hasattr(tokenizer, "decode")
            ):
                return tokenizer

        raise RuntimeError("Could not resolve PtEngine tokenizer.")

    def _messages(
        self,
        audio_path,
        prompt_text,
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

    def continue_from_prefix(
        self,
        audio_path,
        prompt_text,
        vanilla_token_ids,
        brake_checkpoint,
    ):
        if brake_checkpoint <= 0 or brake_checkpoint > len(vanilla_token_ids):
            raise ValueError(
                f"Invalid brake checkpoint "
                f"{brake_checkpoint} for "
                f"{len(vanilla_token_ids)} "
                f"vanilla tokens."
            )
        # Exact saved vanilla assistant prefix.
        prefix_ids = list(vanilla_token_ids[:brake_checkpoint])

        prefix_text = self.tokenizer.decode(
            prefix_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        # Diagnostic only.
        # This may differ at the token-ID level even when the
        # decoded text is exactly preserved.
        recovered_ids = self.tokenizer.encode(
            prefix_text,
            add_special_tokens=False,
        )

        recovered_text = self.tokenizer.decode(
            recovered_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        if recovered_text != prefix_text:
            raise RuntimeError(
                "Audio-Reasoner prefix text changed after "
                "decode -> encode -> decode."
            )

        prefix_roundtrip_exact = recovered_ids == prefix_ids
        # Same-assistant-turn intervention text.
        intervention_text = INTERVENTION_PROMPT.strip()

        intervention_ids = self.tokenizer.encode(
            intervention_text,
            add_special_tokens=False,
        )

        intervention_context_text = "\n\n" + intervention_text + "\n\n"

        conditioned_prefix_text = prefix_text + intervention_context_text
        # Build normal PtEngine request.
        request = InferRequest(
            messages=self._messages(
                audio_path=audio_path,
                prompt_text=prompt_text,
            )
        )

        request.messages.append(
            {
                "role": "assistant",
                "content": conditioned_prefix_text,
            }
        )

        assert request.messages[-1]["role"] == "assistant"
        assert request.messages[-1]["content"] == conditioned_prefix_text
        # Get the official base multimodal context.
        #
        # This contains:
        #   system
        #   user audio/question
        #   assistant generation header
        #
        # but NO generated assistant tokens.
        base_request = InferRequest(
            messages=self._messages(
                audio_path=audio_path,
                prompt_text=prompt_text,
            )
        )

        template = self.engine.default_template

        if template.use_model:
            template.model = self.engine.model

        template.set_mode("pt")

        base_encoded = template.encode(
            base_request,
            return_template_inputs=True,
        )

        base_ids = list(base_encoded["input_ids"])
        # Tokenize the intervention explicitly.
        #
        # Do NOT infer this boundary from template.encode().
        intervention_context_text = "\n\n" + intervention_text + "\n\n"

        intervention_context_ids = self.tokenizer.encode(
            intervention_context_text,
            add_special_tokens=False,
        )

        # Hard text QC for the intervention itself.
        decoded_intervention_context = self.tokenizer.decode(
            intervention_context_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        if decoded_intervention_context != intervention_context_text:
            raise RuntimeError(
                "Audio-Reasoner intervention text "
                "failed encode -> decode exact recovery."
            )
        # Construct the ACTUAL exact-token context.
        #
        # Crucially, prefix_ids are the ORIGINAL saved
        # vanilla generation IDs. No re-tokenization.
        exact_input_ids = base_ids + prefix_ids + intervention_context_ids

        # Verify that the exact token sequence still decodes
        # to the intended assistant content.
        exact_assistant_ids = prefix_ids + intervention_context_ids

        exact_assistant_text = self.tokenizer.decode(
            exact_assistant_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        if exact_assistant_text != conditioned_prefix_text:
            raise RuntimeError(
                "Audio-Reasoner exact assistant token "
                "sequence does not decode to the "
                "intended prefix + intervention text."
            )
        # Let ms-swift do all normal multimodal work,
        # but replace only input_ids.
        exact_template = ExactInputTemplate(
            base_template=template,
            exact_input_ids=exact_input_ids,
        )
        # Continue generation.
        config = RequestConfig(
            max_tokens=MAX_CONTINUATION_TOKENS,
            temperature=0,
            stream=True,
            seed=12345,
            stop=request_stop_sequences(),
        )

        metric = InferStats()

        continuation_text = ""
        with GeneratedTokenCapture(self.engine.model) as captured:
            generation = self.engine.infer(
                [request],
                config,
                metrics=[metric],
                template=exact_template,
            )
            for response_list in generation:
                if response_list[0] is None:
                    continue
                delta = response_list[0].choices[0].delta.content
                if delta:
                    continuation_text += delta
        continuation_ids = list(captured.token_ids)
        if not continuation_ids:
            raise RuntimeError("Audio-Reasoner intervention generated no tokens")

        metrics = metric.compute()
        stop_reason, stop_marker = classify_stop(
            continuation_ids,
            continuation_text,
            MAX_CONTINUATION_TOKENS,
            self.engine.model.generation_config.eos_token_id,
            self.tokenizer,
        )

        return {
            **intervention_prompt_provenance(),
            "generation_stop_policy": STOP_POLICY,
            "generation_stop_marker": stop_marker,
            "prefix_token_ids": prefix_ids,
            "prefix_text": prefix_text,
            "prefix_roundtrip_exact": prefix_roundtrip_exact,
            "exact_token_replay": True,
            "same_turn_intervention": True,
            "base_context_token_count": len(base_ids),
            "context_len": len(exact_input_ids),
            "intervention_text": intervention_text,
            "intervention_token_ids": intervention_ids,
            "intervention_token_count": len(intervention_ids),
            "intervention_context_token_ids": intervention_context_ids,
            "intervention_context_token_count": len(intervention_context_ids),
            "continuation_token_ids": continuation_ids,
            "continuation_text": continuation_text,
            "num_continuation_tokens": len(continuation_ids),
            "token_capture_method": COMPLETE_TOKEN_CAPTURE,
            "continuation_budget": MAX_CONTINUATION_TOKENS,
            "stop_reason": stop_reason,
            "engine_generated_tokens": metrics.get("num_generated_tokens"),
        }
