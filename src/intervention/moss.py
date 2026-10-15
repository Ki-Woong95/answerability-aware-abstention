import sys
from pathlib import Path
import torch
from src.models.base import generation_stop_reason
import torchaudio
import numpy as np
import soundfile as sf

MODEL_ID = "OpenMOSS-Team/MOSS-Audio-8B-Thinking"
MAX_CONTINUATION_TOKENS = 2048
from src.capture.v10_prompt import INTERVENTION_PROMPT, intervention_prompt_provenance

# MOSS-Audio source used by v9_final.
MOSS_SRC = Path(__file__).resolve().parents[2] / "MOSS-Audio" / "src"
if str(MOSS_SRC) not in sys.path:
    sys.path.insert(
        0,
        str(MOSS_SRC),
    )
from modeling_moss_audio import MossAudioModel
from processing_moss_audio import MossAudioProcessor


class MOSSInterventionRunner:
    def __init__(self):
        print(f"Loading MOSS-Audio: {MODEL_ID}")
        self.processor = MossAudioProcessor.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            enable_time_marker=True,
        )
        self.model = MossAudioModel.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map="cuda:0",
        ).eval()
        self.model.requires_grad_(False)
        self.tokenizer = self.processor.tokenizer

    def _load_audio(self, audio_path):
        audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        expected_sr = int(self.processor.config.mel_sr)
        if sample_rate != expected_sr:
            waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(
                0
            )
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sample_rate, new_freq=expected_sr
            )
            audio = waveform.squeeze(0).cpu().numpy()
        return np.ascontiguousarray(audio, dtype=np.float32)

    def prepare_base_batch(
        self,
        audio_path,
        prompt_text,
    ):
        from src.models.audio_delivery import validate_audio_delivery

        validate_audio_delivery(audio_path, "moss")
        audio = self._load_audio(audio_path)
        batch = self.processor(
            text=prompt_text,
            audios=[audio],
            return_tensors="pt",
        )
        input_device = next(self.model.language_model.parameters()).device
        batch = batch.to(input_device)
        if batch.get("audio_data") is not None:
            batch["audio_data"] = batch["audio_data"].to(self.model.dtype)
        return batch

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
        # Exact original audio/question context.
        #
        # MOSS processor ends this context with:
        # <|im_start|>assistant\n
        batch = self.prepare_base_batch(
            audio_path=audio_path,
            prompt_text=prompt_text,
        )
        base_ids = batch["input_ids"][0]
        base_len = int(base_ids.shape[0])
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
        # Exact round-trip QC.
        recovered_ids = self.tokenizer.encode(
            prefix_text,
            add_special_tokens=False,
        )
        prefix_rountrip_exact = recovered_ids == prefix_ids.tolist()
        # Same-assistant-turn intervention.
        #
        # NO <|im_end|>
        # NO new user turn
        # NO new assistant header
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
        # Build exact same-turn context.
        context_ids = torch.cat(
            [
                base_ids,
                prefix_ids,
                intervention_ids,
            ],
            dim=0,
        ).unsqueeze(0)
        batch["input_ids"] = context_ids
        batch["attention_mask"] = torch.ones_like(context_ids)
        # Important for MOSS:
        # recreate AFTER inserting prefix/intervention,
        # so the mask stays aligned with input_ids.
        batch["audio_input_mask"] = context_ids == self.processor.audio_token_id
        context_len = int(context_ids.shape[1])
        language_config = getattr(
            self.model.config,
            "language_config",
            None,
        )
        if language_config is not None and hasattr(
            language_config,
            "max_position_embeddings",
        ):
            max_position_embeddings = int(language_config.max_position_embeddings)
        else:
            max_position_embeddings = 40960
        available_tokens = max_position_embeddings - context_len
        if available_tokens <= 0:
            raise RuntimeError(
                "MOSS intervention context "
                "exceeds text context limit: "
                f"context_len={context_len}, "
                f"max_position_embeddings="
                f"{max_position_embeddings}"
            )
        max_new_tokens = min(
            MAX_CONTINUATION_TOKENS,
            available_tokens,
        )
        # Continue same assistant response.
        with torch.inference_mode():
            generated = self.model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
            )
        if generated.shape[1] < context_len:
            raise RuntimeError(
                "MOSS generate() returned "
                "fewer tokens than supplied "
                "same-turn context."
            )
        continuation = generated[:, context_len:]
        continuation_ids = continuation[0].tolist()
        continuation_text = self.processor.decode(
            continuation[0],
            skip_special_tokens=True,
        )
        return {
            **intervention_prompt_provenance(),
            "prefix_token_ids": prefix_ids.tolist(),
            "prefix_text": prefix_text,
            "prefix_roundtrip_exact": prefix_rountrip_exact,
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
