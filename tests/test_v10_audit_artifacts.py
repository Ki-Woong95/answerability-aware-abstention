from frozen_fixture import use_synthetic_umwp

"""Synthetic CPU audits and artifact loaders; no generation or fitting."""

import ast
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import numpy as np

from src import audit_pipeline as audit
from src.capture.storage import save_trajectory
from src.capture.trajectory import Trajectory
from src.capture.v10_prompt import TASK_PROMPT
from src.data.frozen_umwp import v10_artifact_provenance
from src.data.tts_text import normalize_tts_text
from src.gate.calibrate_tau5 import load_gate, load_selected_layer
from src.gate.datasets import get_dataset_config
from src.gate.diagnose_val_operating_point import load_tau5
from src.gate.select_layer import load_all_gates
from src.models.audio_reasoner_stop import STOP_POLICY
from src.models.base import GenerationResult
from src.models.probing import BOUNDARY_VERSION

ROOT = Path(__file__).resolve().parents[1]


class AuditTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_original_manifest_is_filtered_before_validation(self):
        split = {name: [name] for name in ["gate_train", "calibration", "val", "test"]}
        path = self.root / "split.json"
        path.write_text(json.dumps(split))
        config = replace(get_dataset_config("umwp"), split_path=path)
        pairs, manifest = [], []
        for pair_id in list(split) + ["removed"]:
            pair = dict(pair_id=pair_id, source="synthetic", category=1)
            for condition, suffix in [("answerable", "A"), ("unanswerable", "U")]:
                question = f"Question {pair_id} {suffix}."
                pair[condition] = dict(
                    question=question, answer=[42] if suffix == "A" else []
                )
                manifest.append(
                    dict(
                        pair_id=pair_id,
                        trial_id=pair_id + "_" + suffix,
                        condition=condition,
                        split=pair_id if pair_id in split else "gate_train",
                        source="synthetic",
                        category=1,
                        raw_text=question,
                        tts_text=normalize_tts_text(question),
                        voice="synthetic",
                        audio_path=pair_id + suffix + ".wav",
                    )
                )
            pairs.append(pair)
        with (
            patch.object(audit, "get_dataset_config", return_value=config),
            patch.object(audit, "read_jsonl", side_effect=[pairs, manifest]),
            patch.object(
                audit.sf,
                "info",
                return_value=SimpleNamespace(samplerate=24000, channels=1, frames=10),
            ),
        ):
            expected, report = audit.expected_inputs("umwp")
        self.assertEqual(report["pairs"], 4)
        self.assertEqual(report["original_pairs"], 5)
        self.assertEqual(report["trials"], 8)
        self.assertEqual(report["expected_capture_trials"], 7)
        self.assertEqual(report["expected_split_trials"]["calibration"], 1)
        self.assertFalse(any(r["pair_id"] == "removed" for r in expected.values()))

    def test_default_audit_does_not_require_worldsense(self):
        config = replace(
            get_dataset_config("umwp"),
            trajectory_root=self.root / "absent",
            gate_root=self.root / "absent",
            intervention_root=self.root / "absent",
            judge_root=self.root / "absent",
        )
        cwd = Path.cwd()
        try:
            os.chdir(self.root)
            with (
                patch.object(audit, "expected_inputs", return_value=({}, {})) as inputs,
                patch.object(audit, "get_dataset_config", return_value=config),
            ):
                audit.run(self.root / "report.json", False)
            inputs.assert_called_once_with("umwp")
            self.assertEqual(
                set(json.loads((self.root / "report.json").read_text())["inputs"]),
                {"umwp"},
            )
        finally:
            os.chdir(cwd)

    def make_empty(self):
        metadata = self.root / "af_next/gate_train/metadata.jsonl"
        state = metadata.parent / "states/synthetic_A.pt"
        generation = GenerationResult(
            "af_next",
            [3, 99],
            "FINAL: 42",
            2,
            "eos",
            {},
            {},
            num_layers=28,
            hidden_size=3,
            boundary_version=BOUNDARY_VERSION,
        )
        trial = Trajectory(
            "synthetic_A",
            "synthetic",
            "answerable",
            "gate_train",
            "synthetic",
            1,
            "synthetic",
            "unused.wav",
            "Question.",
            "Question.",
            TASK_PROMPT,
            (42,),
            generation,
        )
        save_trajectory(trial, state, metadata)
        return metadata, state, json.loads(metadata.read_text())

    def test_states_audit_accepts_dimensioned_zero_checkpoint_file(self):
        metadata, state, row = self.make_empty()
        before = (metadata.read_bytes(), state.read_bytes())
        result = audit.audit_capture(
            metadata, "umwp", {row["trial_id"]: row}, TASK_PROMPT, True
        )
        self.assertNotIn("invalid_state", result.get("issues", {}))
        self.assertEqual(result["state_files_checked"], 1)
        self.assertEqual(result["zero_checkpoint_state_files_checked"], 1)
        self.assertEqual((metadata.read_bytes(), state.read_bytes()), before)

    def test_empty_state_wrong_hidden_dimension_is_rejected(self):
        _, state, row = self.make_empty()
        payload = torch.load(state, weights_only=True)
        payload["all_layers"] = torch.empty(0, 28, 4)
        torch.save(payload, state)
        with self.assertRaises(ValueError):
            audit.validate_audited_state(row)

    def test_empty_state_payload_lists_and_norm_shape_are_verified(self):
        _, state, row = self.make_empty()
        payload = torch.load(state, weights_only=True)
        for change in [
            dict(fixed_checkpoints=[1]),
            dict(sentence_checkpoints=[1]),
            dict(final_norm=torch.empty(1, 3)),
            dict(all_layers=torch.empty(0, 28)),
        ]:
            torch.save(dict(payload, **change), state)
            with self.subTest(change=tuple(change)), self.assertRaises(ValueError):
                audit.validate_audited_state(row)

    def test_empty_state_checkpoint_and_metadata_corruption_is_rejected(self):
        _, _, row = self.make_empty()
        for change in [
            dict(checkpoints=[1]),
            dict(eligible_checkpoint_count=1),
            dict(monitoring_status="missing"),
            dict(boundary_events=[{}]),
            dict(num_layers=27),
        ]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit.validate_audited_state(dict(row, **change))

    def test_snapshot_ignores_only_unfinished_trailing_line(self):
        path = self.root / "metadata.jsonl"
        path.write_bytes(b'{"trial_id":"complete"}\n{"trial_id":')
        before = path.read_bytes()
        rows, incomplete = audit.read_metadata_snapshot(path)
        self.assertTrue(incomplete)
        self.assertEqual(rows, [dict(trial_id="complete")])
        self.assertEqual(path.read_bytes(), before)
        path.write_bytes(b"{broken}\n")
        with self.assertRaises(json.JSONDecodeError):
            audit.read_metadata_snapshot(path)


class ARArtifactTests(unittest.TestCase):
    def setUp(self):
        use_synthetic_umwp(self)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.config = replace(
            get_dataset_config("umwp"),
            gate_root=root / "gates",
            layer_selection_root=root / "layers",
            calibration_root=root / "calibration",
        )
        self.payload = dict(
            v10_artifact_provenance("umwp"), model_name="audio-reasoner", dataset="umwp"
        )

    def test_provenance_binds_ar_but_does_not_change_capture_default(self):
        self.assertNotIn("generation_stop_policy", v10_artifact_provenance("umwp"))
        self.assertNotIn(
            "generation_stop_policy", v10_artifact_provenance("umwp", "moss")
        )
        self.assertEqual(
            v10_artifact_provenance("umwp", "audio-reasoner")["generation_stop_policy"],
            STOP_POLICY,
        )

    def test_saved_ar_gate_contains_stop_policy_without_fitting(self):
        from src.gate.train_all_layers import save_gate

        path = self.config.gate_root / "synthetic.pt"
        gate = SimpleNamespace(eval=lambda: None, state_dict=lambda: {})
        standardizer = SimpleNamespace(
            count=1, mean=np.zeros(3), M2=np.zeros(3), eps=1e-6
        )
        save_gate(
            path,
            "umwp",
            "audio-reasoner",
            1,
            17,
            3,
            gate,
            standardizer,
            20,
            ["synthetic"],
            True,
        )
        self.assertEqual(
            torch.load(path, weights_only=True)["generation_stop_policy"], STOP_POLICY
        )

    def test_both_gate_loaders_reject_missing_or_stale_ar_policy(self):
        path = self.config.gate_root / "audio-reasoner/all_layers/layer_01/seed_17.pt"
        path.parent.mkdir(parents=True)
        for policy in [None, "old-policy"]:
            payload = dict(self.payload)
            if policy is not None:
                payload["generation_stop_policy"] = policy
            torch.save(payload, path)
            with (
                self.subTest(policy=policy),
                patch("src.gate.calibrate_tau5.MLP64Gate") as gate,
            ):
                with self.assertRaisesRegex(RuntimeError, "stop policy"):
                    load_gate("audio-reasoner", 1, 17, torch.device("cpu"), self.config)
                gate.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, "stop policy"):
                load_all_gates("audio-reasoner", [1], torch.device("cpu"), self.config)

    def test_selected_layer_rejects_stale_policy_and_accepts_current(self):
        path = self.config.layer_selection_root / "audio-reasoner_all_layers.json"
        path.parent.mkdir(parents=True)
        payload = dict(
            self.payload, split="val", test_read=False, selected_layer_number=1
        )
        path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(RuntimeError, "stop policy"):
            load_selected_layer("audio-reasoner", self.config)
        payload["generation_stop_policy"] = STOP_POLICY
        path.write_text(json.dumps(payload))
        self.assertEqual(load_selected_layer("audio-reasoner", self.config)[0], 1)

    def test_calibration_rejects_stale_policy_and_accepts_current(self):
        path = self.config.calibration_root / "audio-reasoner_tau5.json"
        path.parent.mkdir(parents=True)
        payload = dict(
            self.payload,
            selected_layer_number=1,
            threshold_frozen_after_calibration=True,
            test_read=False,
            seed_results={str(s): {"tau5": 0.8} for s in [17, 29, 43]},
        )
        path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(RuntimeError, "stop policy"):
            load_tau5("audio-reasoner", 1, self.config)
        payload["generation_stop_policy"] = STOP_POLICY
        path.write_text(json.dumps(payload))
        self.assertEqual(
            load_tau5("audio-reasoner", 1, self.config)[0], {17: 0.8, 29: 0.8, 43: 0.8}
        )

    def test_artifact_writers_use_model_bound_provenance(self):
        for name in ["train_all_layers", "select_layer", "calibrate_tau5"]:
            tree = ast.parse((ROOT / f"src/gate/{name}.py").read_text())
            calls = [
                n
                for n in ast.walk(tree)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "v10_artifact_provenance"
            ]
            self.assertTrue(calls)
            self.assertTrue(
                all(
                    len(n.args) == 2
                    and isinstance(n.args[1], ast.Name)
                    and n.args[1].id == "model_name"
                    for n in calls
                )
            )

    def test_output_token_metric_is_not_compute_or_latency(self):
        from src.eval.efficiency import GENERATED_OUTPUT_TOKEN_METRIC as metric

        self.assertEqual(metric["unit"], "generated_output_tokens")
        self.assertEqual(
            metric["post_intervention_generated_tokens"],
            "brake_checkpoint + num_continuation_tokens",
        )
        self.assertIn("GPU compute", metric["not_measured"])
        self.assertIn("latency", metric["not_measured"])


if __name__ == "__main__":
    unittest.main()
