"""Prompt and injected-context regressions without pretrained model inference."""

import ast
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.capture.v10_prompt import (
    TASK_PROMPT,
    INTERVENTION_PROMPT,
    SHARED_REASONING_INSTRUCTION,
    intervention_prompt_provenance,
)
from src.intervention.storage import load_existing
from src.models.base import generation_stop_reason

ROOT = Path(__file__).resolve().parents[1]


class InterventionPromptTests(unittest.TestCase):
    def test_approved_initial_prompt_unchanged(self):
        self.assertEqual(
            hashlib.sha256(TASK_PROMPT.encode()).hexdigest(),
            "ed28851d84ecb5d4dc60354178fbca971c0937d53161ff94c001bc05896a2850",
        )

    def test_reassessment_preserves_all_approved_instructions(self):
        self.assertEqual(
            INTERVENTION_PROMPT.split("\n\n", 1)[1], SHARED_REASONING_INSTRUCTION
        )
        for instruction in [
            "Let's think step by step",
            "Do not introduce any additional assumptions.",
            "explain why",
            "REASON: <your explanation here>",
            "FINAL: I don't know.",
            "Place REASON: before FINAL: and output exactly one FINAL: line.",
        ]:
            self.assertIn(instruction, INTERVENTION_PROMPT)
        self.assertIn("If you believe", INTERVENTION_PROMPT)
        self.assertNotIn("closing tag", INTERVENTION_PROMPT)

    def test_all_backends_import_the_same_prompt(self):
        for model in ["af_next", "audio_reasoner", "moss"]:
            with self.subTest(model=model):
                tree = ast.parse((ROOT / f"src/intervention/{model}.py").read_text())
                self.assertTrue(
                    any(
                        isinstance(n, ast.ImportFrom)
                        and n.module == "src.capture.v10_prompt"
                        and any(a.name == "INTERVENTION_PROMPT" for a in n.names)
                        for n in tree.body
                    )
                )
                self.assertFalse(
                    any(
                        isinstance(n, ast.Assign)
                        and any(
                            isinstance(t, ast.Name) and t.id == "INTERVENTION_PROMPT"
                            for t in n.targets
                        )
                        for n in tree.body
                    )
                )

    def check_native_runner(self, model, class_name):
        tree = ast.parse((ROOT / f"src/intervention/{model}.py").read_text())
        cls = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
        )
        method = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "continue_from_prefix"
        )
        scope = dict(
            torch=torch,
            INTERVENTION_PROMPT=INTERVENTION_PROMPT,
            intervention_prompt_provenance=intervention_prompt_provenance,
            MAX_CONTINUATION_TOKENS=2048,
            generation_stop_reason=generation_stop_reason,
        )
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                "runner_fixture",
                "exec",
            ),
            scope,
        )
        encode = lambda text, **kwargs: [100 + ord(c) for c in text]
        decode = lambda ids, **kwargs: "".join(chr(int(i) - 100) for i in ids)
        base = torch.tensor([[1, 2]])
        captured = {}

        def generate(**kwargs):
            captured.update(kwargs)
            return torch.cat([kwargs["input_ids"], torch.tensor([[999]])], dim=1)

        runner = SimpleNamespace(
            prepare_base_batch=lambda **kwargs: dict(
                input_ids=base.clone(), attention_mask=torch.ones_like(base)
            ),
            tokenizer=SimpleNamespace(encode=encode, decode=decode),
            processor=SimpleNamespace(
                audio_token_id=1,
                batch_decode=lambda *a, **k: ["FINAL: 42"],
                decode=lambda *a, **k: "FINAL: 42",
            ),
            model=SimpleNamespace(
                generate=generate,
                generation_config=SimpleNamespace(eos_token_id=999),
                config=SimpleNamespace(
                    text_config=SimpleNamespace(max_position_embeddings=8192),
                    language_config=SimpleNamespace(max_position_embeddings=8192),
                ),
            ),
        )
        prefix = encode("Observed reasoning. ")
        result = scope["continue_from_prefix"](
            runner, "unused.wav", TASK_PROMPT, prefix + [999], len(prefix)
        )
        expected = [1, 2] + prefix + encode("\n\n" + INTERVENTION_PROMPT + "\n\n")
        self.assertEqual(captured["input_ids"][0].tolist(), expected)
        self.assertEqual(result["prefix_token_ids"], prefix)
        self.assertEqual(result["intervention_text"], INTERVENTION_PROMPT)
        self.assertEqual(result["continuation_token_ids"], [999])
        for key, value in intervention_prompt_provenance().items():
            self.assertEqual(result[key], value)
        if model == "moss":
            self.assertEqual(
                captured["audio_input_mask"].tolist(),
                (captured["input_ids"] == 1).tolist(),
            )

    def test_af_actual_context_injection(self):
        self.check_native_runner("af_next", "AFNextInterventionRunner")

    def test_moss_actual_context_injection(self):
        self.check_native_runner("moss", "MOSSInterventionRunner")

    def test_stale_prompt_resume_rejected_before_retry_rewrite(self):
        for field in intervention_prompt_provenance():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "results.jsonl"
                planned = dict(
                    trial_id="synthetic",
                    model_name="af_next",
                    split="val",
                    seed=17,
                    **intervention_prompt_provenance(),
                )
                stale = dict(planned, intervention_error="retryable")
                stale[field] = "old"
                path.write_text(json.dumps(stale) + "\n")
                before = path.read_bytes()
                with (
                    patch("src.intervention.storage.validate_artifact_dataset"),
                    self.assertRaisesRegex(RuntimeError, field),
                ):
                    load_existing(path, True, [planned], None, "af_next", "val", 17)
                self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
