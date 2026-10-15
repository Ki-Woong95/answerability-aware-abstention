import torch
from transformers import AutoModelForSeq2SeqLM, AutoProcessor
from src.models.base import generation_stop_reason
from src.capture.v10_prompt import INTERVENTION_PROMPT, intervention_prompt_provenance

MODEL_ID = "nvidia/audio-flamingo-next-think-hf"
MAX_CONTINUATION_TOKENS = 2048


class AFNextInterventionRunner:
    def __init__(self):
        print(f"Loading AF-Next: {MODEL_ID}")

        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.tokenizer = getattr(
            self.processor,
            "tokenizer",
            None,
        )

        if (
            self.tokenizer is None
            or not hasattr(self.tokenizer, "encode")
            or not hasattr(self.tokenizer, "decode")
        ):
            raise RuntimeError("Could not resolve AF-Next tokenizer.")

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            device_map="auto",
        ).eval()

        self.model.requires_grad_(False)

    def _apply_chat_template(self, conversation):
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

    def prepare_base_batch(self, audio_path, prompt_text):
        from src.models.audio_delivery import validate_audio_delivery

        validate_audio_delivery(audio_path, "af_next")

        conversation = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_text,
                        },
                        {
                            "type": "audio",
                            "path": str(audio_path),
                        },
                    ],
                }
            ]
        ]

        return self._apply_chat_template(conversation)

    def continue_from_prefix(
        self,
        audio_path,
        prompt_text,
        vanilla_token_ids,
        brake_checkpoint,
    ):
        if brake_checkpoint <= 0 or brake_checkpoint > len(vanilla_token_ids):
            raise ValueError(
                f"Invalid brake checkpoint {brake_checkpoint} "
                f"for {len(vanilla_token_ids)} vanilla tokens."
            )
        # Reconstruct the exact original user/audio -> assistant
        #    generation context.
        batch = self.prepare_base_batch(
            audio_path=audio_path,
            prompt_text=prompt_text,
        )

        base_ids = batch["input_ids"][0]
        base_len = int(base_ids.shape[0])

        # Before modifying input_ids, make sure there are no other
        # unknown fields aligned one-to-one with the text sequence.
        for key, value in batch.items():
            if key in {
                "input_ids",
                "attention_mask",
            }:
                continue

            if (
                torch.is_tensor(value)
                and value.ndim == 2
                and value.shape[1] == base_len
            ):
                raise RuntimeError(
                    f"Unexpected sequence-aligned field {key!r}. "
                    "Inspect it before modifying the AF-Next "
                    "same-turn intervention context."
                )
        # Exact saved vanilla assistant prefix.
        prefix_ids = torch.tensor(
            vanilla_token_ids[:brake_checkpoint],
            dtype=torch.long,
            device=base_ids.device,
        )

        prefix_text = self.tokenizer.decode(
            prefix_ids.tolist(),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        # Assistant-side intervention.
        #
        # No assistant end token.
        # No new user turn.
        # No new assistant header.
        #
        # The model sees:
        #
        # original context
        # -> assistant header
        # -> exact saved vanilla prefix
        # -> injected reassessment instruction
        # -> continuation
        intervention_text = INTERVENTION_PROMPT.strip()

        intervention_context_text = "\n\n" + intervention_text + "\n\n"

        intervention_id_list = self.tokenizer.encode(
            intervention_context_text,
            add_special_tokens=False,
        )

        intervention_ids = torch.tensor(
            intervention_id_list,
            dtype=torch.long,
            device=base_ids.device,
        )
        # Build same-turn context.
        context_ids = torch.cat(
            [
                base_ids,
                prefix_ids,
                intervention_ids,
            ],
            dim=0,
        ).unsqueeze(0)

        batch["input_ids"] = context_ids

        if "attention_mask" in batch:
            old_mask = batch["attention_mask"][0]

            if int(old_mask.shape[0]) != base_len:
                raise RuntimeError(
                    "AF-Next attention mask length does not "
                    "match the original text context."
                )

            added_mask = torch.ones(
                len(prefix_ids) + len(intervention_ids),
                dtype=old_mask.dtype,
                device=old_mask.device,
            )

            batch["attention_mask"] = torch.cat(
                [
                    old_mask,
                    added_mask,
                ],
                dim=0,
            ).unsqueeze(0)

        context_len = int(batch["input_ids"].shape[1])

        # AF-Next outer config also contains an audio-position limit.
        # For text generation, use the Qwen2 text backbone limit.
        text_config = getattr(
            self.model.config,
            "text_config",
            None,
        )

        if text_config is not None and hasattr(
            text_config,
            "max_position_embeddings",
        ):
            max_position_embeddings = int(text_config.max_position_embeddings)
        else:
            max_position_embeddings = int(self.model.config.max_position_embeddings)

        available_tokens = max_position_embeddings - context_len

        if available_tokens <= 0:
            raise RuntimeError(
                "AF-Next intervention context exceeds "
                "the model text-context limit: "
                f"context_len={context_len}, "
                f"max_position_embeddings="
                f"{max_position_embeddings}"
            )

        max_new_tokens = min(
            MAX_CONTINUATION_TOKENS,
            available_tokens,
        )
        # Continue generation from the intervention context.
        with torch.inference_mode():
            generated = self.model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                repetition_penalty=1.2,
                do_sample=False,
            )

        if generated.shape[1] < context_len:
            raise RuntimeError(
                "AF-Next generate() returned fewer tokens "
                "than the supplied same-turn context."
            )

        continuation = generated[:, context_len:]

        continuation_ids = continuation[0].tolist()

        continuation_text = self.processor.batch_decode(
            continuation,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return {
            **intervention_prompt_provenance(),
            "prefix_token_ids": prefix_ids.tolist(),
            "prefix_text": prefix_text,
            "intervention_text": intervention_text,
            "intervention_token_ids": intervention_ids.tolist(),
            "intervention_token_count": len(intervention_ids),
            "same_turn_intervention": True,
            "context_len": context_len,
            "max_position_embeddings": max_position_embeddings,
            "continuation_budget": max_new_tokens,
            "continuation_token_ids": continuation_ids,
            "continuation_text": continuation_text,
            "num_continuation_tokens": len(continuation_ids),
            "stop_reason": generation_stop_reason(
                continuation_ids,
                max_new_tokens,
                self.model.generation_config.eos_token_id,
            ),
        }
