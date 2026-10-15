from dataclasses import replace
from src.gate.datasets import get_dataset_config
from src.data.frozen_umwp import v10_artifact_provenance
from frozen_fixture import use_synthetic_umwp
import importlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.archive_audio_reasoner import archive
from scripts.check_audio_reasoner_run import check_intervention
from src.models.base import COMPLETE_TOKEN_CAPTURE


class ARArchiveTests(unittest.TestCase):
    def test_archive_relocates_paths_preserves_other_models_and_originals(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "data/trajectories/audio-reasoner/val"
            old.mkdir(parents=True)
            state = old / "states/t.pt"
            state.parent.mkdir()
            state.write_bytes(b"original tensor bytes")
            metadata = (
                json.dumps(
                    {"trial_id": "t", "state_path": str(state.relative_to(root))}
                )
                + "\n"
            )
            (old / "metadata.jsonl").write_text(metadata)
            judge = root / "data/judge/val_vanilla_judged_v2.jsonl"
            judge.parent.mkdir()
            original_judge = '{"model":"audio-reasoner","trial_id":"t"}\n{"model":"af_next","trial_id":"a"}\n'
            judge.write_text(original_judge)
            table = root / "data/analysis/val/model_summary.csv"
            table.parent.mkdir(parents=True)
            table.write_text("model,accuracy\naudio-reasoner,0.9\nmoss,0.8\n")
            destination = root / "data/archive/ar_test"
            preview = archive(root, destination)
            self.assertEqual(preview["status"], "planned")
            self.assertEqual(judge.read_text(), original_judge)
            self.assertTrue(state.exists())
            self.assertFalse(destination.exists())
            result = archive(root, destination, apply=True)
            self.assertEqual(result["status"], "complete")
            self.assertFalse(old.exists())
            relocated = destination / old.relative_to(root) / "metadata.jsonl"
            saved = json.loads(relocated.read_text())
            self.assertEqual(
                (root / saved["state_path"]).read_bytes(), b"original tensor bytes"
            )
            self.assertEqual(
                relocated.with_name("metadata.jsonl.original").read_text(), metadata
            )
            self.assertEqual(judge.read_text(), '{"model":"af_next","trial_id":"a"}\n')
            self.assertEqual(
                (destination / judge.relative_to(root)).read_text(), original_judge
            )
            self.assertIn("moss,0.8", table.read_text())
            self.assertNotIn("audio-reasoner", table.read_text())
            with self.assertRaises(FileExistsError):
                archive(root, destination, apply=True)


class ARStageVerificationTests(unittest.TestCase):
    def setUp(self):
        use_synthetic_umwp(self)

    def test_intervention_requires_complete_successful_exact_tokens(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = replace(get_dataset_config("umwp"), intervention_root=root)
            planned = {
                17: [
                    {
                        "trial_id": "t",
                        "triggered": True,
                        "brake_checkpoint": 1,
                        "input_fingerprint": "same",
                    }
                ]
            }
            module = importlib.import_module("src.intervention.run_val_intervention")
            output = root / "audio-reasoner/val/seed_17.jsonl"
            output.parent.mkdir(parents=True)
            row = {
                **v10_artifact_provenance("umwp", "audio-reasoner"),
                **planned[17][0],
                "model_name": "audio-reasoner",
                "split": "val",
                "seed": 17,
                "dataset": "umwp",
                "intervention_run": True,
                "intervention_error": None,
                "intervention_result": {
                    "token_capture_method": COMPLETE_TOKEN_CAPTURE,
                    "prefix_token_ids": [10],
                    "continuation_token_ids": [20, 30],
                    "num_continuation_tokens": 2,
                },
            }
            with (
                patch(
                    "scripts.check_audio_reasoner_run.get_dataset_config",
                    return_value=config,
                ),
                patch.object(
                    module,
                    "plan_interventions",
                    return_value=(
                        [{"trial_id": "t", "token_ids": [10, 11]}],
                        None,
                        1,
                        {},
                        planned,
                        (1, 0, 1),
                    ),
                ),
            ):
                output.write_text(json.dumps(row) + "\n")
                check_intervention("umwp", "val")
                row["intervention_error"] = "generation failed"
                output.write_text(json.dumps(row) + "\n")
                with self.assertRaisesRegex(RuntimeError, "intervention failed"):
                    check_intervention("umwp", "val")
                row["intervention_error"] = None
                row["intervention_result"]["prefix_token_ids"] = [999]
                output.write_text(json.dumps(row) + "\n")
                with self.assertRaisesRegex(RuntimeError, "prefix mismatch"):
                    check_intervention("umwp", "val")
                output.write_text("")
                with self.assertRaisesRegex(RuntimeError, "incomplete intervention"):
                    check_intervention("umwp", "val")


if __name__ == "__main__":
    unittest.main()
