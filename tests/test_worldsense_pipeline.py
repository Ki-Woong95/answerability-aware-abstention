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
import torch

from src.capture import run_worldsense as normal, run_worldsense_always_idk as ablation
from src.capture.storage import metadata_to_trajectory
from src.gate.calibrate_tau5 import calibrate_tau5_ltt
from src.gate.datasets import get_dataset_config
from src.gate.diagnose_test_operating_point import first_crossing as test_crossing
from src.gate.diagnose_val_operating_point import first_crossing as val_crossing
from src.gate.train_all_layers import train_model
from src.intervention.run_test_intervention import (
    first_crossing_details as test_details,
)
from src.intervention.run_val_intervention import first_crossing_details as val_details
from src.models.base import COMPLETE_TOKEN_CAPTURE, GenerationResult


class DummyModel:
    hidden_dim = 2

    def __init__(self, name="af_next", marker=None, layers=1):
        self.name, self.marker, self.layers = name, marker, layers

    def load(self):
        pass

    def generate_full(self, **kwargs):
        return GenerationResult(
            model_name=self.name,
            token_ids=[5],
            text="FINAL: older.",
            num_generated_tokens=1,
            stop_reason="eos",
            all_layers={1: torch.randn(self.layers, 2)},
            final_norm={1: torch.zeros(2)},
            sentence_checkpoints=(1,),
            token_capture_method=self.marker,
        )


class WorldSenseCaptureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        source_id = "worldsense_gate_train_0000"
        self.rows = []
        for suffix, condition, gold in [
            ("_A", "answerable", "older"),
            ("_U", "unanswerable", None),
        ]:
            self.rows.append(
                dict(
                    trial_id=source_id + suffix,
                    source_id=source_id,
                    split="gate_train",
                    condition=condition,
                    gold_answer=gold,
                    audio_path=str(self.root / "audio.wav"),
                    audio_text="Alice is older than Bob.",
                    question="Alice [RELATION] Bob",
                )
            )
        self.manifest = [
            {
                **{
                    key: self.rows[0][key]
                    for key in ("source_id", "split", "audio_path", "audio_text")
                },
                "dataset": "worldsense",
                "voice": "af_heart",
            }
        ]
        self.trials_path = self.root / "trials.jsonl"
        self.manifest_path = self.root / "manifest.jsonl"
        self.split_path = self.root / "splits.json"
        self.write_jsonl(self.trials_path, self.rows)
        self.write_jsonl(self.manifest_path, self.manifest)
        self.split_path.write_text(
            json.dumps(
                {
                    "splits": {
                        "gate_train": [source_id],
                        "calibration": [],
                        "val": [],
                        "test": [],
                    }
                }
            )
        )
        self.args = SimpleNamespace(
            model="af_next",
            split="gate_train",
            allow_test=False,
            condition="all",
            trials=self.trials_path,
            tts_manifest=self.manifest_path,
            splits=self.split_path,
            output=self.root / "output",
            resume=False,
            limit_pairs=None,
        )

    @staticmethod
    def write_jsonl(path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    @property
    def metadata_path(self):
        return self.args.output / self.args.model / "gate_train/metadata.jsonl"

    def run_capture(self, runner=normal, marker=None, layers=1):
        with (
            contextlib.redirect_stdout(io.StringIO()),
            patch.object(
                runner,
                "build_model",
                return_value=DummyModel(self.args.model, marker, layers),
            ) as build,
        ):
            runner.run(self.args)
        return build

    def test_always_idk_captures_openended_gold_and_resumes(self):
        self.run_capture(ablation)
        saved = ablation.read_jsonl(self.metadata_path)
        self.assertEqual(saved[0]["gold_answer"], ["older"])
        self.assertEqual(saved[1]["gold_answer"], [])
        self.assertEqual(metadata_to_trajectory(saved[0]).category, None)
        self.args.resume = True
        self.run_capture(ablation).assert_not_called()
        self.assertEqual(ablation.DEFAULT_TRIALS, normal.DEFAULT_TRIALS)
        self.assertEqual(ablation.DEFAULT_SPLITS, normal.DEFAULT_SPLITS)

    def test_wrong_audio_context_is_rejected_before_model_load(self):
        self.manifest[0]["audio_text"] = "A completely different context."
        self.write_jsonl(self.manifest_path, self.manifest)
        with patch.object(normal, "build_model") as build:
            with self.assertRaisesRegex(RuntimeError, "audio_text mismatch"):
                normal.run(self.args)
            build.assert_not_called()

    def test_changed_inputs_are_not_silently_resumed(self):
        self.run_capture()
        original = normal.read_jsonl(self.metadata_path)
        self.args.resume = True
        for field, value in [
            ("prompt_text", "old question"),
            ("raw_text", "old context"),
            ("gold_answer", ["1"]),
            ("voice", "old_voice"),
            ("model_name", "moss"),
            ("sentence_checkpoints", [2]),
        ]:
            with self.subTest(field=field):
                modified = [dict(row) for row in original]
                modified[0][field] = value
                self.write_jsonl(self.metadata_path, modified)
                with patch.object(normal, "build_model") as build:
                    with self.assertRaisesRegex(RuntimeError, field):
                        normal.run(self.args)
                    build.assert_not_called()

    def test_matching_legacy_af_next_remains_resumable(self):
        self.run_capture()
        before = self.metadata_path.read_bytes()
        self.args.resume = True
        self.run_capture().assert_not_called()
        self.assertEqual(self.metadata_path.read_bytes(), before)

    def test_legacy_audio_reasoner_requires_recapture(self):
        self.args.model = "audio-reasoner"
        self.run_capture()
        before = self.metadata_path.read_bytes()
        self.args.resume = True
        with self.assertRaisesRegex(RuntimeError, "predate complete token capture"):
            self.run_capture()
        self.assertEqual(self.metadata_path.read_bytes(), before)

    def test_complete_audio_reasoner_is_resumable(self):
        self.args.model = "audio-reasoner"
        self.run_capture(marker=COMPLETE_TOKEN_CAPTURE)
        saved = normal.read_jsonl(self.metadata_path)[0]
        self.assertEqual(
            metadata_to_trajectory(saved).generation.token_capture_method,
            COMPLETE_TOKEN_CAPTURE,
        )
        self.args.resume = True
        self.run_capture(marker=COMPLETE_TOKEN_CAPTURE).assert_not_called()

    def test_incomplete_training_is_rejected_before_writing_gates(self):
        self.run_capture(layers=28)
        self.split_path.write_text(
            json.dumps(
                {
                    "splits": {
                        "gate_train": [
                            self.rows[0]["source_id"],
                            "worldsense_gate_train_0001",
                        ],
                    }
                }
            )
        )
        config = replace(
            get_dataset_config("worldsense"),
            split_path=self.split_path,
            trajectory_root=self.args.output,
            gate_root=self.root / "gates",
        )
        with self.assertRaisesRegex(RuntimeError, "incomplete gate_train capture"):
            train_model("af_next", torch.device("cpu"), 1, None, False, config)
        self.assertFalse(config.gate_root.exists())
        # An explicitly requested partial smoke run still works.
        torch.set_num_threads(1)
        with contextlib.redirect_stdout(io.StringIO()):
            train_model("af_next", torch.device("cpu"), 1, 1, False, config)
        summary = json.loads(
            (config.gate_root / "af_next/all_layers/train_summary.json").read_text()
        )
        self.assertEqual(summary["n_pairs"], 1)

    def test_training_rejects_pairs_outside_frozen_split(self):
        self.run_capture()
        self.split_path.write_text(
            json.dumps({"splits": {"gate_train": ["another_pair"]}})
        )
        config = replace(
            get_dataset_config("worldsense"),
            split_path=self.split_path,
            trajectory_root=self.args.output,
            gate_root=self.root / "gates",
        )
        with self.assertRaisesRegex(RuntimeError, "outside frozen gate_train split"):
            train_model("af_next", torch.device("cpu"), 1, 1, False, config)


class ThresholdPrecisionTests(unittest.TestCase):
    def test_all_online_comparisons_preserve_frozen_threshold_precision(self):
        states, mean, std = torch.zeros((1, 1, 1)), torch.zeros(1), torch.ones(1)
        gate = torch.nn.Identity()
        threshold = float(np.nextafter(0.5, np.inf))
        for crossing in (val_crossing, test_crossing):
            self.assertIsNone(
                crossing(states, 1, gate, mean, std, threshold, torch.device("cpu"))
            )
            self.assertEqual(
                crossing(states, 1, gate, mean, std, 0.5, torch.device("cpu")), 1
            )
        for details in (val_details, test_details):
            self.assertIsNone(details(states, 1, gate, mean, std, threshold))
            self.assertEqual(
                details(states, 1, gate, mean, std, 0.5)["sentence_index_1based"], 1
            )

    def test_calibration_count_matches_online_interventions(self):
        states = torch.logit(torch.linspace(0.01, 0.99, 400)).reshape(400, 1, 1)
        scores = torch.sigmoid(states).flatten().numpy()
        calibrated = calibrate_tau5_ltt(scores)
        gate, mean, std = torch.nn.Identity(), torch.zeros(1), torch.ones(1)
        observed = sum(
            val_details(state[None], 1, gate, mean, std, calibrated["tau5"]) is not None
            for state in states
        )
        self.assertEqual(observed, calibrated["observed_false_brakes"])
        self.assertEqual(observed, 12)


if __name__ == "__main__":
    unittest.main()
