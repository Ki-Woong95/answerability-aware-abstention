from frozen_fixture import use_synthetic_umwp
import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from src.capture import run as umwp, run_umwp_always_idk as idk
from src.capture.storage import load_trajectory_states
from src.data import generate_umwp_tts as tts
from src.eval.run_judge import judge_input_fingerprint, load_completed as load_judgments
from src.eval import (
    build_judge_inputs_always_idk as idk_inputs,
    run_judge_always_idk as idk_judge,
)
from src.capture.run_worldsense_always_idk import TASK_PROMPT as WS_IDK_PROMPT
from src.gate.datasets import (
    get_dataset_config,
    validate_trajectory_rows,
    validate_artifact_dataset,
)
from src.gate.select_layer import load_sentence_states
from src.gate.train_all_layers import sentence_only_states
from src.intervention.storage import load_existing
from src.models.base import COMPLETE_TOKEN_CAPTURE, generation_stop_reason
from test_worldsense_pipeline import DummyModel as LegacyDummyModel
from src.models.probing import BOUNDARY_VERSION


class DummyModel(LegacyDummyModel):
    def generate_full(self, **kwargs):
        result = super().generate_full(**kwargs)
        result.boundary_version = BOUNDARY_VERSION
        result.num_layers = self.layers
        result.hidden_size = self.hidden_dim
        result.boundary_events = (
            {
                "confirmation_token_index": 1,
                "readout_kind": "boundary_confirmation_state",
                "boundaries": [],
            },
        )
        return result


class UMWPCaptureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.pairs = [
            {
                "pair_id": "umwp_1",
                "source": "SVAMP",
                "category": 1,
                "answerable": {
                    "question": "One apple. How many apples?",
                    "answer": [1.0],
                },
                "unanswerable": {"question": "How many apples?"},
            }
        ]
        self.manifest = []
        for condition, suffix in (("answerable", "_A"), ("unanswerable", "_U")):
            raw = self.pairs[0][condition]["question"]
            self.manifest.append(
                dict(
                    trial_id="umwp_1" + suffix,
                    pair_id="umwp_1",
                    condition=condition,
                    split="gate_train",
                    source="SVAMP",
                    category=1,
                    voice="af_heart",
                    raw_text=raw,
                    tts_text=raw,
                    audio_path=str(self.root / (suffix + ".wav")),
                )
            )
        self.manifest_path, self.pairs_path = (
            self.root / "manifest.jsonl",
            self.root / "pairs.jsonl",
        )
        self.write(self.manifest_path, self.manifest)
        self.write(self.pairs_path, self.pairs)
        self.split_path = self.root / "splits.json"
        self.split_path.write_text(
            json.dumps(dict(gate_train=["umwp_1"], calibration=[], val=[], test=[]))
        )
        self.args = SimpleNamespace(
            model="af_next",
            split="gate_train",
            condition="all",
            allow_test=False,
            limit_pairs=None,
            resume=False,
            output=self.root / "output",
            manifest=self.manifest_path,
            pairs=self.pairs_path,
            splits=self.split_path,
        )

    @staticmethod
    def write(path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    @property
    def metadata(self):
        return self.args.output / self.args.model / "gate_train/metadata.jsonl"

    def capture(self, runner, marker=None):
        with (
            contextlib.redirect_stdout(io.StringIO()),
            patch.object(
                runner, "build_model", return_value=DummyModel(self.args.model, marker)
            ) as build,
        ):
            runner.run(self.args)
        return build

    def test_both_prompts_capture_and_resume_without_loading_model(self):
        for runner in (umwp, idk):
            with self.subTest(runner=runner.__name__):
                self.args.output = self.root / runner.__name__
                self.args.resume = False
                self.capture(runner)
                saved = umwp.read_jsonl(self.metadata)
                self.assertEqual(saved[0]["prompt_text"], runner.TASK_PROMPT)
                before = self.metadata.read_bytes()
                self.args.resume = True
                self.capture(runner).assert_not_called()
                self.assertEqual(self.metadata.read_bytes(), before)

    def test_both_runners_reject_stale_prompts_and_legacy_ar(self):
        for runner in (umwp, idk):
            self.args.output = self.root / runner.__name__
            self.args.resume = False
            self.capture(runner)
            saved = umwp.read_jsonl(self.metadata)
            saved[0]["prompt_text"] = "old prompt"
            self.write(self.metadata, saved)
            self.args.resume = True
            with self.assertRaisesRegex(RuntimeError, "prompt_text"):
                self.capture(runner)
            self.args.output = self.root / (runner.__name__ + "_ar")
            self.args.model, self.args.resume = "audio-reasoner", False
            self.capture(runner)
            self.args.resume = True
            before = self.metadata.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "predate complete token capture"):
                self.capture(runner)
            self.assertEqual(self.metadata.read_bytes(), before)
            self.args.model = "af_next"

    def test_manifest_mismatch_and_duplicates_fail_before_model_load(self):
        for field, value in (
            ("split", "test"),
            ("raw_text", "old question"),
            ("tts_text", "old audio"),
        ):
            modified = [dict(row) for row in self.manifest]
            modified[0][field] = value
            self.write(self.manifest_path, modified)
            with patch.object(umwp, "build_model") as build:
                with self.assertRaisesRegex(RuntimeError, field):
                    umwp.run(self.args)
                build.assert_not_called()
        self.write(self.manifest_path, self.manifest + [self.manifest[0]])
        with self.assertRaisesRegex(RuntimeError, "Duplicate input trial_id"):
            umwp.run(self.args)

    def test_metadata_without_resume_and_zero_limit_are_rejected(self):
        self.capture(umwp)
        with self.assertRaises(FileExistsError):
            self.capture(umwp)
        with self.assertRaisesRegex(ValueError, "positive"):
            umwp.select_rows([], {}, "gate_train", "all", 0)


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        use_synthetic_umwp(self)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_checkpoint_disagreement_is_rejected_by_both_gate_loaders(self):
        path = self.root / "state.pt"
        torch.save(dict(all_layers=torch.zeros(2, 28, 2), checkpoints=[2, 4]), path)
        row = dict(
            trial_id="a",
            state_path=str(path),
            checkpoints=[1, 4],
            sentence_checkpoints=[1, 4],
        )
        for loader in (load_sentence_states, sentence_only_states):
            with self.assertRaisesRegex(RuntimeError, "metadata checkpoints"):
                loader(row, self.root / "metadata.jsonl", 28)
        row.update(checkpoints=[2, 4], sentence_checkpoints=[4, 2])
        with self.assertRaisesRegex(RuntimeError, "strictly increasing"):
            load_sentence_states(row, self.root / "metadata.jsonl", 28)

    def test_non_finite_state_is_rejected(self):
        path = self.root / "state.pt"
        torch.save(
            dict(
                all_layers=torch.full((1, 1, 1), float("nan")),
                final_norm=torch.zeros(1, 1),
                checkpoints=[1],
                sentence_checkpoints=[1],
                fixed_checkpoints=[],
            ),
            path,
        )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            load_trajectory_states(path)

    def test_validation_checks_membership_and_complete_coverage(self):
        split = self.root / "split.json"
        split.write_text(json.dumps(dict(val=["one", "two"], test=["test"])))
        config = replace(get_dataset_config("umwp"), split_path=split)
        rows = [
            dict(
                trial_id="one" + suffix,
                pair_id="one",
                model_name="af_next",
                split="val",
                condition=condition,
            )
            for suffix, condition in (("A", "answerable"), ("U", "unanswerable"))
        ]
        with self.assertRaisesRegex(RuntimeError, "incomplete val"):
            validate_trajectory_rows(rows, config, "val", "af_next")
        rows[0]["pair_id"] = "test"
        with self.assertRaisesRegex(RuntimeError, "outside frozen val"):
            validate_trajectory_rows(rows, config, "val", "af_next")
        with self.assertRaisesRegex(RuntimeError, "partial smoke-run"):
            validate_artifact_dataset(
                dict(dataset="umwp", training_complete=False),
                config,
                self.root / "gate.pt",
            )

    def test_intervention_resume_validates_before_retry_rewrite(self):
        path = self.root / "seed_17.jsonl"
        plan = dict(
            trial_id="one",
            triggered=True,
            brake_checkpoint=2,
            tau5=0.5,
            input_fingerprint="new",
        )
        from src.data.frozen_umwp import v10_artifact_provenance

        saved = dict(
            plan,
            **v10_artifact_provenance("umwp"),
            model_name="af_next",
            split="val",
            seed=17,
            dataset="umwp",
            intervention_error="failure",
        )
        path.write_text(json.dumps(dict(saved, input_fingerprint="old")) + "\n")
        before = path.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "input_fingerprint"):
            load_existing(
                path, True, [plan], get_dataset_config("umwp"), "af_next", "val", 17
            )
        self.assertEqual(path.read_bytes(), before)
        path.write_text(json.dumps(saved) + "\n")
        self.assertEqual(
            load_existing(
                path, True, [plan], get_dataset_config("umwp"), "af_next", "val", 17
            ),
            ({}, {}),
        )
        self.assertEqual(path.read_text(), "")

    def test_judgment_resume_rejects_changed_response_rubric_and_duplicates(self):
        path = self.root / "judged.jsonl"
        row = dict(
            model="af_next",
            trial_id="one",
            condition="answerable",
            response_text="FINAL: 1",
            answerable_answer=[1],
        )
        saved = dict(row, input_fingerprint=judge_input_fingerprint(row, "judge"))
        path.write_text(json.dumps(saved) + "\n")
        self.assertEqual(len(load_judgments(path, [row], "judge")), 1)
        for changed in (
            dict(row, response_text="FINAL: 2"),
            dict(row, answerable_answer=[2]),
        ):
            with self.assertRaisesRegex(RuntimeError, "provenance"):
                load_judgments(path, [changed], "judge")
        with self.assertRaisesRegex(RuntimeError, "provenance"):
            load_judgments(path, [row], "other-judge")
        path.write_text((json.dumps(saved) + "\n") * 2)
        with self.assertRaisesRegex(RuntimeError, "duplicate judged"):
            load_judgments(path, [row], "judge")

    def test_eos_takes_precedence_at_budget(self):
        self.assertEqual(generation_stop_reason([1, 15], 2, 15), "eos")
        self.assertEqual(generation_stop_reason([1, 2], 2, [15]), "max_new_tokens")

    def test_worldsense_always_idk_uses_its_frozen_size_and_semantic_gold(self):
        split = self.root / "split.json"
        split.write_text(
            json.dumps(dict(val=["worldsense_one"], test=["worldsense_two"]))
        )
        config = replace(get_dataset_config("worldsense"), split_path=split)
        question = "Alice [RELATION] Bob"
        rows = [
            dict(
                trial_id="worldsense_one" + suffix,
                pair_id="worldsense_one",
                split="val",
                condition=condition,
                model_name="af_next",
                gold_answer=gold,
                text="FINAL: older",
                prompt_text=WS_IDK_PROMPT.format(question=question),
            )
            for suffix, condition, gold in (
                ("_A", "answerable", ["older"]),
                ("_U", "unanswerable", []),
            )
        ]
        trials = [dict(trial_id=row["trial_id"], question=question) for row in rows]
        with (
            patch.object(idk_inputs, "get_dataset_config", return_value=config),
            patch.object(idk_inputs, "read_jsonl", return_value=trials),
        ):
            idk_inputs.validate_rows(
                rows, self.root / "meta.jsonl", "af_next", "val", "worldsense"
            )
        items = [idk_inputs.build_item("af_next", row, "worldsense") for row in rows]
        self.assertEqual(items[0]["answerable_answer"], ["older"])
        self.assertEqual(items[0]["dataset"], "worldsense")
        with patch.object(idk_judge, "get_dataset_config", return_value=config):
            idk_judge.validate_input(
                items, self.root / "input.jsonl", "af_next", "worldsense", "val"
            )

    def test_partial_tts_verification_preserves_unselected_manifest_rows(self):
        (self.root / "data/processed").mkdir(parents=True)
        (self.root / "data/splits").mkdir(parents=True)
        (self.root / "data/tts/umwp/val").mkdir(parents=True)
        pairs = [
            {
                "pair_id": "umwp_1",
                "source": "SVAMP",
                "category": 1,
                "answerable": {"question": "One apple."},
                "unanswerable": {"question": "How many apples?"},
            }
        ]
        (self.root / "data/processed/umwp_pairs.jsonl").write_text(
            json.dumps(pairs[0]) + "\n"
        )
        (self.root / "data/splits/umwp_final_split.json").write_text(
            json.dumps(dict(gate_train=[], calibration=[], val=["umwp_1"], test=[]))
        )
        manifest = []
        for condition, suffix in (("answerable", "A"), ("unanswerable", "U")):
            path = Path(f"data/tts/umwp/val/umwp_1_{suffix}.wav")
            sf.write(self.root / path, np.zeros(240), 24000)
            manifest.append(
                dict(
                    trial_id=f"umwp_1_{suffix}",
                    audio_path=str(path),
                    split="val",
                    raw_text=pairs[0][condition]["question"],
                    tts_text=pairs[0][condition]["question"],
                    voice="af_heart",
                    speed=1.0,
                )
            )
        manifest_path = self.root / "data/tts/umwp_manifest.jsonl"
        manifest_path.write_text("".join(json.dumps(row) + "\n" for row in manifest))
        import os

        old_cwd = Path.cwd()
        try:
            os.chdir(self.root)
            with (
                patch("sys.argv", ["tts", "--limit", "1"]),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                tts.main()
            saved = [
                json.loads(line) for line in manifest_path.read_text().splitlines()
            ]
            self.assertEqual(len(saved), 2)
            self.assertEqual(saved[1], manifest[1])
            before = manifest_path.read_bytes()
            pairs[0]["answerable"]["question"] = "A changed question."
            (self.root / "data/processed/umwp_pairs.jsonl").write_text(
                json.dumps(pairs[0]) + "\n"
            )
            with (
                patch("sys.argv", ["tts", "--limit", "1"]),
                self.assertRaisesRegex(RuntimeError, "different raw_text"),
            ):
                tts.main()
            self.assertEqual(manifest_path.read_bytes(), before)
        finally:
            os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
