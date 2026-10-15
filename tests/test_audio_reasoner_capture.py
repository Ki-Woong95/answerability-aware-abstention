import queue
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch, Mock

import torch
from transformers import (
    LogitsProcessor,
    Qwen2Config,
    Qwen2ForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)

from src.models.audio_reasoner import AudioReasonerModel, GeneratedTokenCapture
from src.models.base import COMPLETE_TOKEN_CAPTURE


class ForcedTokens(LogitsProcessor):
    def __init__(self, token_ids):
        self.token_ids = token_ids

    def __call__(self, input_ids, scores):
        token_id = self.token_ids[input_ids.shape[1] - 2]
        scores.fill_(-float("inf"))
        scores[:, token_id] = 0
        return scores


class QueueStreamer:
    def __init__(self):
        self.queue = queue.Queue()

    def put(self, value):
        self.queue.put(value)

    def end(self):
        self.queue.put(None)


class TinyStreamingEngine:
    """Exercise real HF generation in a worker thread, as PtEngine does."""

    def __init__(self, token_ids, token_texts=None):
        config = Qwen2Config(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
            eos_token_id=15,
            pad_token_id=0,
        )
        self.model = Qwen2ForCausalLM(config).eval()
        self.token_ids = token_ids
        self.token_texts = token_texts

    def infer(self, requests, config, **kwargs):
        self.request_config = config
        streamer = QueueStreamer()
        errors = []
        engine = self

        class NativeEnd(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                text = "".join(
                    engine.token_texts.get(int(t), "") for t in input_ids[0, 2:]
                )
                return any(text.endswith(stop) for stop in config.stop)

        def generate():
            try:
                with torch.inference_mode():
                    output = self.model.generate(
                        input_ids=torch.tensor([[1, 2]]),
                        attention_mask=torch.ones((1, 2), dtype=torch.long),
                        streamer=streamer,
                        max_new_tokens=config.max_tokens,
                        do_sample=False,
                        logits_processor=[ForcedTokens(self.token_ids)],
                        stopping_criteria=(
                            StoppingCriteriaList([NativeEnd()])
                            if self.token_texts
                            else None
                        ),
                    )
                self.generated_ids = output[0, 2:].tolist()
            except Exception as exc:
                errors.append(exc)
                streamer.end()

        worker = threading.Thread(target=generate)
        worker.start()
        prompt = True
        while True:
            value = streamer.queue.get(timeout=20)
            if value is None:
                break
            if prompt:
                prompt = False
                continue
            text = (
                self.token_texts.get(int(value[0]), "")
                if self.token_texts
                else ("" if int(value[0]) == 15 else ".")
            )
            yield [
                SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content=text))]
                )
            ]
        worker.join(timeout=20)
        if errors:
            raise errors[0]


class AudioReasonerCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def capture(self, emitted, budget, token_texts=None):
        engine = TinyStreamingEngine(emitted, token_texts)
        model = AudioReasonerModel()
        model.engine = engine
        model._hidden_dim = 8
        model.embed_tokens = engine.model.model.embed_tokens
        model.decoder_layers = engine.model.model.layers
        model.final_norm = engine.model.model.norm
        model.tokenizer = SimpleNamespace(eos_token_id=15)
        if token_texts:
            model.tokenizer.decode = lambda ids, **kwargs: "".join(
                token_texts.get(int(t), "") for t in ids
            )
        model.InferRequest = SimpleNamespace
        model.RequestConfig = SimpleNamespace
        model._messages = (
            lambda **kwargs: []
        )  # synthetic engine does not preprocess audio
        decoder = Mock(
            side_effect=lambda token_id: (
                token_texts.get(token_id, "")
                if token_texts
                else ("Done. " if token_id == 3 else "<eos>")
            )
        )
        decoder.fingerprint = "synthetic"
        decoder.native_stop_token_ids = {}
        with patch("src.models.audio_reasoner.TokenTextDecoder", return_value=decoder):
            result = model._generate_full("unused.wav", "test", budget, (), True)
        self.assertEqual(result.token_ids, engine.generated_ids)
        self.assertEqual(result.num_generated_tokens, len(engine.generated_ids))
        self.assertEqual(result.token_capture_method, COMPLETE_TOKEN_CAPTURE)
        self.assertEqual(engine.request_config.stop, ["</RESPONSE>"])
        self.assertNotIn("generate", engine.model.__dict__)
        self.assertEqual(len(model.embed_tokens._forward_pre_hooks), 0)
        return result

    def test_native_stop_keeps_terminal_ids_and_aligned_states(self):
        pieces = {
            3: "Done. ",
            4: "</THINK>",
            5: "<RESPONSE>42",
            6: "</",
            7: "RESPONSE",
            8: ">",
            9: "Unwanted repeat",
        }
        result = self.capture([3, 4, 5, 6, 7, 8, 9, 15], 9, pieces)
        self.assertEqual(result.token_ids, [3, 4, 5, 6, 7, 8])
        self.assertEqual(result.stop_reason, "stop_sequence")
        self.assertEqual(result.generation_stop_marker, "</RESPONSE>")
        self.assertEqual(result.consumed_generated_tokens, 5)
        self.assertEqual(result.sentence_checkpoints, (1,))
        self.assertNotIn("Unwanted repeat", result.text)

    def test_native_stop_exactly_at_budget_is_not_max_new_tokens(self):
        pieces = {
            3: "Done. ",
            4: "</THINK>",
            5: "<RESPONSE>42",
            6: "</",
            7: "RESPONSE",
            8: ">",
        }
        result = self.capture([3, 4, 5, 6, 7, 8], 6, pieces)
        self.assertEqual(result.stop_reason, "stop_sequence")

    def test_terminal_token_at_length_limit_is_retained(self):
        result = self.capture([3, 3, 3, 3], 4)
        self.assertEqual(result.token_ids, [3, 3, 3, 3])
        self.assertEqual(result.stop_reason, "max_new_tokens")
        # Hidden states remain aligned to tokens already fed back to the model.
        self.assertEqual(result.sentence_checkpoints, (1, 2, 3))

    def test_eos_token_and_stop_reason_are_retained(self):
        result = self.capture([3, 3, 15], 4)
        self.assertEqual(result.token_ids, [3, 3, 15])
        self.assertEqual(result.stop_reason, "eos")
        self.assertEqual(result.sentence_checkpoints, (1, 2))

    def test_eos_on_last_allowed_token_is_normal_termination(self):
        self.assertEqual(self.capture([3, 3, 3, 15], 4).stop_reason, "eos")

    def test_generate_is_restored_after_failure(self):
        def fail(**kwargs):
            raise ValueError("generation failed")

        model = SimpleNamespace(generate=fail)
        with self.assertRaisesRegex(ValueError, "generation failed"):
            with GeneratedTokenCapture(model):
                model.generate(streamer=QueueStreamer())
        self.assertIs(model.generate, fail)

    def test_intervention_uses_emitted_ids_instead_of_retokenizing_text(self):
        from src.intervention import audio_reasoner as intervention

        engine = TinyStreamingEngine([3, 3, 15])
        engine.default_template = SimpleNamespace(
            use_model=False,
            set_mode=lambda mode: None,
            encode=lambda *args, **kwargs: {"input_ids": [1, 2]},
        )
        encoded = {}

        def encode(text, **kwargs):
            # Same text can legitimately have a different tokenization.
            ids = [9] if text == "." else [100 + ord(char) for char in text]
            encoded[tuple(ids)] = text
            return ids

        def decode(ids, **kwargs):
            return "".join(
                "" if token == 15 else ("." if token in (3, 9) else chr(token - 100))
                for token in ids
            )

        runner = intervention.AudioReasonerInterventionRunner.__new__(
            intervention.AudioReasonerInterventionRunner
        )
        runner.engine = engine
        runner._messages = lambda **kwargs: []  # raw prefix test; no audio processing
        runner.tokenizer = SimpleNamespace(encode=encode, decode=decode)
        metric = SimpleNamespace(compute=lambda: {"num_generated_tokens": 2})
        with (
            patch.object(intervention, "MAX_CONTINUATION_TOKENS", 4),
            patch.object(intervention, "InferStats", return_value=metric),
        ):
            result = runner.continue_from_prefix("unused.wav", "test", [3, 3, 15], 1)
        from src.capture.v10_prompt import (
            INTERVENTION_PROMPT,
            intervention_prompt_provenance,
        )

        self.assertEqual(result["intervention_text"], INTERVENTION_PROMPT)
        self.assertEqual(
            decode(result["intervention_context_token_ids"]),
            "\n\n" + INTERVENTION_PROMPT + "\n\n",
        )
        for key, value in intervention_prompt_provenance().items():
            self.assertEqual(result[key], value)
        self.assertEqual(result["prefix_token_ids"], [3])
        self.assertFalse(result["prefix_roundtrip_exact"])
        self.assertEqual(result["continuation_text"], "..")
        self.assertEqual(result["continuation_token_ids"], [3, 3, 15])
        self.assertEqual(result["num_continuation_tokens"], 3)
        self.assertEqual(result["token_capture_method"], COMPLETE_TOKEN_CAPTURE)
        self.assertEqual(result["stop_reason"], "eos")


if __name__ == "__main__":
    unittest.main()
