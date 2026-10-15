import copy, json, tempfile, unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from scripts.audits.freeze_treecut_dataset import allocate, validate_pair
from scripts.audits.treecut_pairs import build_pair
from src.capture.run_treecut import select_trials, run
from src.capture.v10_prompt import TASK_PROMPT
from src.gate.datasets import (
    get_dataset_config,
    load_split_ids,
    validate_trajectory_rows,
    validate_artifact_dataset,
)
from src.models.base import GenerationResult
from src.models.probing import BOUNDARY_VERSION


class TreeCutPipelineTests(unittest.TestCase):
    def test_pair_counts_and_changed_condition_detection(self):
        for n in [1, 17, 433, 834]:
            self.assertEqual(sum(allocate(n).values()), n)
        p = build_pair(43, 4, 4, True, 2)
        validate_pair(p)
        bad = copy.deepcopy(p)
        bad["answerable"]["problem"] = bad["answerable"]["problem"].replace(
            "dollars", "euros", 1
        )
        with self.assertRaises(AssertionError):
            validate_pair(bad)

    def test_loader_and_calibration_only_A(self):
        rows = [
            dict(pair_id="p", condition=c, split="calibration")
            for c in ["answerable", "unanswerable"]
        ]
        self.assertEqual(
            [r["condition"] for r in select_trials(rows, "calibration")], ["answerable"]
        )
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "split.json"
            p.write_text(
                json.dumps(
                    {
                        "splits": {
                            "gate_train": ["p"],
                            "calibration": ["c"],
                            "val": ["v"],
                            "test": ["t"],
                        }
                    }
                )
            )
            cfg = replace(get_dataset_config("treecut"), split_path=p)
            self.assertEqual(load_split_ids(cfg, "gate_train"), {"p"})
            self.assertEqual(cfg.expected_counts("calibration"), (1, 0, 1))
            p.write_text(json.dumps({"splits": {"gate_train": ["p"], "val": ["p"]}}))
            with self.assertRaises(RuntimeError):
                load_split_ids(cfg, "gate_train")

    def test_stale_AR_and_cross_dataset_rejected(self):
        cfg = get_dataset_config("treecut")
        provenance = {
            "generation_stop_policy": "ar-native-response-end-v1",
            "frozen_split_sha256": "frozen",
            "prompt_version": "current",
        }
        good = dict(dataset="treecut", model_name="audio-reasoner", **provenance)
        with patch(
            "src.data.frozen_treecut.artifact_provenance", return_value=provenance
        ):
            validate_artifact_dataset(good, cfg, Path("synthetic"))
            for key in provenance:
                bad = dict(good)
                bad.pop(key)
                with self.assertRaises(RuntimeError):
                    validate_artifact_dataset(bad, cfg, Path("synthetic"))
            with self.assertRaises(RuntimeError):
                validate_artifact_dataset(
                    dict(good, dataset="umwp"), cfg, Path("synthetic")
                )

    def test_zero_checkpoint_capture_and_resume(self):
        from src.capture.storage import load_trajectory_states

        fingerprint = json.loads(
            Path("docs/provenance/phase2b/af_next_tokenizer.json").read_text()
        )["tokenizer_fingerprint"]

        class Fake:
            hidden_dim = 3

            def load(self):
                pass

            def generate_full(
                self, audio, prompt, budget, checkpoints, capture_sentence_boundaries
            ):
                assert prompt == TASK_PROMPT
                return GenerationResult(
                    "af_next",
                    [1, 2],
                    "FINAL: I don't know.",
                    2,
                    "eos",
                    {},
                    {},
                    num_layers=28,
                    hidden_size=3,
                    boundary_version=BOUNDARY_VERSION,
                    tokenizer_fingerprint=fingerprint,
                    consumed_generated_tokens=2,
                )

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            wav = d / "dummy.wav"
            wav.write_bytes(b"synthetic")
            import hashlib

            rows = [
                dict(
                    dataset="treecut",
                    pair_id="p",
                    trial_id="p_" + s,
                    condition=c,
                    split="gate_train",
                    source="treecut",
                    category=None,
                    voice="af_heart",
                    audio_path=str(wav),
                    audio_sha256=hashlib.sha256(wav.read_bytes()).hexdigest(),
                    raw_text="synthetic",
                    tts_text="synthetic",
                    gold_answer=["42"] if c == "answerable" else [],
                )
                for c, s in [("answerable", "A"), ("unanswerable", "U")]
            ]
            split = d / "split.json"
            split.write_text(json.dumps({"splits": {"gate_train": ["p"]}}))
            cfg = replace(get_dataset_config("treecut"), split_path=split)
            prov = {"boundary_version": BOUNDARY_VERSION}
            args = SimpleNamespace(
                model="af_next",
                split="gate_train",
                allow_test=False,
                limit_pairs=1,
                output=d / "out",
                resume=False,
            )
            with (
                patch("src.capture.run_treecut.load_trials", return_value=rows),
                patch("src.capture.run_treecut.get_dataset_config", return_value=cfg),
                patch("src.capture.run_treecut.artifact_provenance", return_value=prov),
                patch("src.data.frozen_treecut.artifact_provenance", return_value=prov),
                patch(
                    "src.capture.run_treecut.validate_audio_delivery",
                    return_value=SimpleNamespace(duration=1),
                ),
            ):
                run(args, lambda _: Fake())
                path = args.output / "af_next/gate_train/metadata.jsonl"
                with path.open() as stream:
                    saved = list(map(json.loads, stream))
                for row in saved:
                    self.assertEqual(
                        tuple(
                            load_trajectory_states(row["state_path"])[
                                "all_layers"
                            ].shape
                        ),
                        (0, 28, 3),
                    )
                args.resume = True
                run(args, lambda _: self.fail("Resume must not load model"))
                bad = copy.deepcopy(saved)
                bad[0]["pair_id"] = "outside"
                with self.assertRaises(RuntimeError):
                    validate_trajectory_rows(
                        bad, cfg, "gate_train", "af_next", require_complete=False
                    )

    def test_test_guard(self):
        with self.assertRaises(RuntimeError):
            run(SimpleNamespace(split="test", allow_test=False))


if __name__ == "__main__":
    unittest.main()
