from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from scripts.check_audio_reasoner_run import check_capture, check_gates
from scripts.wait_for_gpu import parse_gpu_processes, process_start_ticks, wait_for_gpu
from src.gate.datasets import get_dataset_config
from src.gate.model import MLP64Gate
from src.gate import select_layer


class QueueWaitingTests(unittest.TestCase):
    def test_process_identity_handles_spaces_and_exited_process(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "42/stat"
            path.parent.mkdir()
            fields = ["S"] + ["0"] * 18 + ["12345"]
            path.write_text("42 (worker (with spaces)) " + " ".join(fields))
            self.assertEqual(process_start_ticks(42, root), "12345")
            path.unlink()
            self.assertIsNone(process_start_ticks(42, root))

    def test_waits_for_predecessor_but_not_gpu1_and_handles_pid_reuse(self):
        self.assertEqual(
            parse_gpu_processes("GPU-0, 11\nGPU-1, 22\ninvalid\n", "GPU-0"), ["11"]
        )
        with TemporaryDirectory() as directory:
            status = Path(directory) / "status.txt"
            with (
                patch(
                    "scripts.wait_for_gpu.process_start_ticks",
                    side_effect=["123", "456"],
                ),
                patch("scripts.wait_for_gpu.time.sleep") as sleep,
                patch(
                    "scripts.wait_for_gpu.subprocess.check_output",
                    side_effect=["GPU-0\n", "GPU-1, 22\n"],
                ),
            ):
                wait_for_gpu(0, 42, "123", status, poll_seconds=1)
                sleep.assert_called_once_with(1)
                self.assertEqual(status.read_text(), "ready: GPU 0\n")


class WorldSenseStageTests(unittest.TestCase):
    def config(self, root):
        split = root / "split.json"
        split.write_text(
            json.dumps(
                {
                    "splits": {
                        "gate_train": ["p"],
                        "calibration": [],
                        "val": [],
                        "test": [],
                    }
                }
            )
        )
        return replace(
            get_dataset_config("worldsense"),
            split_path=split,
            trajectory_root=root / "trajectories",
            gate_root=root / "gates",
        )

    def test_model_specific_capture_requires_stream_marker_only_for_ar(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.config(root)
            for model in ("af_next", "moss", "audio-reasoner"):
                output = config.trajectory_root / model / "gate_train"
                (output / "states").mkdir(parents=True)
                rows = []
                for condition, suffix in (("answerable", "A"), ("unanswerable", "U")):
                    state = output / "states" / f"p_{suffix}.pt"
                    torch.save(
                        {
                            "checkpoints": (1,),
                            "sentence_checkpoints": (1,),
                            "fixed_checkpoints": (),
                            "all_layers": torch.ones(1, 1, 3),
                            "final_norm": torch.ones(1, 3),
                        },
                        state,
                    )
                    rows.append(
                        {
                            "trial_id": f"p_{suffix}",
                            "pair_id": "p",
                            "condition": condition,
                            "split": "gate_train",
                            "model_name": model,
                            "state_path": str(state),
                            "source": "worldsense",
                            "category": None,
                            "voice": "af_heart",
                            "audio_path": "audio.wav",
                            "raw_text": "context",
                            "tts_text": "context",
                            "prompt_text": "question",
                            "gold_answer": ["before"] if suffix == "A" else [],
                            "token_ids": [10, 11],
                            "text": "Answer.",
                            "num_generated_tokens": 2,
                            "stop_reason": "eos",
                            "num_layers": 1,
                            "hidden_dim": 3,
                            "checkpoints": [1],
                            "sentence_checkpoints": [1],
                            "fixed_checkpoints": [],
                        }
                    )
                metadata = output / "metadata.jsonl"
                metadata.write_text("".join(json.dumps(row) + "\n" for row in rows))
                with patch(
                    "scripts.check_audio_reasoner_run.get_dataset_config",
                    return_value=config,
                ):
                    if model == "audio-reasoner":
                        with self.assertRaisesRegex(
                            RuntimeError, "legacy token capture"
                        ):
                            check_capture("worldsense", "gate_train", model=model)
                        for row in rows:
                            row["token_capture_method"] = "generated_stream_v1"
                        metadata.write_text(
                            "".join(json.dumps(row) + "\n" for row in rows)
                        )
                    check_capture("worldsense", "gate_train", model=model)

    def test_full_gate_bank_is_required_and_nonfinite_weights_fail(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.config(root)
            output = config.gate_root / "af_next/all_layers"
            layer = output / "layer_01"
            layer.mkdir(parents=True)
            summary = {
                "dataset": "worldsense",
                "model_name": "af_next",
                "training_complete": True,
                "training_pair_ids": ["p"],
                "n_pairs": 1,
                "epochs": 20,
                "n_layers": 1,
                "seeds": [17, 29, 43],
            }
            (output / "train_summary.json").write_text(json.dumps(summary))
            for seed in (17, 29, 43):
                payload = {
                    "dataset": "worldsense",
                    "model_name": "af_next",
                    "training_complete": True,
                    "protocol": "v9_sentence_boundary_cumulative_mean",
                    "hidden_dim": 3,
                    "training_positions": "sentence_boundaries_only",
                    "layer_number_1based": 1,
                    "layer_index_0based": 0,
                    "seed": seed,
                    "gate_state_dict": MLP64Gate(3).state_dict(),
                    "standardizer": {
                        "count": 2,
                        "mean": torch.zeros(3),
                        "M2": torch.ones(3),
                        "eps": 1e-6,
                    },
                }
                torch.save(payload, layer / f"seed_{seed}.pt")
            with (
                patch(
                    "scripts.check_audio_reasoner_run.get_dataset_config",
                    return_value=config,
                ),
                patch.dict(select_layer.MODEL_LAYERS, {"af_next": 1}),
            ):
                check_gates("worldsense", "af_next")
                path = layer / "seed_17.pt"
                payload = torch.load(path, weights_only=False)
                payload["gate_state_dict"]["net.0.weight"][0, 0] = float("nan")
                torch.save(payload, path)
                with self.assertRaisesRegex(RuntimeError, "non-finite weights"):
                    check_gates("worldsense", "af_next")
                summary["training_pair_ids"] = []
                (output / "train_summary.json").write_text(json.dumps(summary))
                with self.assertRaisesRegex(RuntimeError, "full 20-epoch protocol"):
                    check_gates("worldsense", "af_next")


if __name__ == "__main__":
    unittest.main()
