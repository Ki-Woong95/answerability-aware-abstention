from frozen_fixture import use_synthetic_umwp
import unittest
from types import SimpleNamespace
from src.models.audio_reasoner_stop import (
    classify_stop,
    request_stop_sequences,
    STOP_POLICY,
)


class ARStopTests(unittest.TestCase):
    def setUp(self):
        use_synthetic_umwp(self)

    def classify(self, text, n=3, budget=10, eos=99):
        return classify_stop(
            list(range(n)),
            "stream may strip the stop string",
            budget,
            eos,
            SimpleNamespace(decode=lambda *a, **k: text),
        )

    def test_shared_native_stop(self):
        self.assertEqual(request_stop_sequences(), ["</RESPONSE>"])

    def test_native_stop_detected_from_ids_when_stream_strips_it(self):
        self.assertEqual(
            self.classify("<RESPONSE>42</RESPONSE>"), ("stop_sequence", "</RESPONSE>")
        )

    def test_native_stop_on_last_allowed_token_is_not_truncation(self):
        self.assertEqual(
            self.classify("42</RESPONSE>", n=10, budget=10),
            ("stop_sequence", "</RESPONSE>"),
        )

    def test_reasoning_end_is_not_generation_end(self):
        self.assertEqual(
            self.classify("</THINK>", n=10, budget=10), ("max_new_tokens", None)
        )

    def test_reasoning_repetition_retains_budget(self):
        self.assertEqual(
            self.classify("Repeat. Repeat. Repeat.", n=10, budget=10),
            ("max_new_tokens", None),
        )

    def test_eos_at_budget_is_normal_termination(self):
        self.assertEqual(self.classify("Answer", n=10, budget=10, eos=9), ("eos", None))

    def test_unknown_early_stop_is_not_mislabeled_eos(self):
        self.assertEqual(self.classify("Unfinished"), ("unknown", None))

    def test_policy_is_ar_specific(self):
        from src.models.audio_reasoner_stop import stop_policy_provenance

        self.assertEqual(
            stop_policy_provenance("audio-reasoner"),
            {"generation_stop_policy": STOP_POLICY},
        )
        self.assertEqual(stop_policy_provenance("moss"), {})

    def test_old_stop_policy_cannot_resume(self):
        from pathlib import Path
        from src.capture.validation import validate_completed
        from src.capture.v10_prompt import TASK_PROMPT
        from src.models.base import COMPLETE_TOKEN_CAPTURE

        expected = dict(
            trial_id="synthetic", model_name="audio-reasoner", prompt_text=TASK_PROMPT
        )
        saved = dict(expected, token_capture_method=COMPLETE_TOKEN_CAPTURE)
        with self.assertRaisesRegex(RuntimeError, "stale AR generation stop policy"):
            validate_completed(saved, expected, Path("unused.pt"))

    def test_old_stop_policy_cannot_enter_gate_population(self):
        from src.gate.datasets import get_dataset_config, validate_trajectory_rows

        with self.assertRaisesRegex(RuntimeError, "stale AR generation stop policy"):
            validate_trajectory_rows(
                [dict(model_name="audio-reasoner")],
                get_dataset_config("umwp"),
                "gate_train",
                require_complete=False,
            )


if __name__ == "__main__":
    unittest.main()
