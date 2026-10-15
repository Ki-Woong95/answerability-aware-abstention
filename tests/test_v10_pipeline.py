from frozen_fixture import use_synthetic_umwp

"""CPU-only integration regressions using synthetic tensors and fake backends.

No backbone, optimizer, standardizer fit, calibration fit, judge or test data.
"""

import ast
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

from src.capture import run as capture
from src.capture.storage import save_trajectory, metadata_to_trajectory
from src.capture.trajectory import Trajectory
from src.capture.v10_prompt import TASK_PROMPT
from src.gate.datasets import get_dataset_config, validate_trajectory_rows
from src.gate.select_layer import load_sentence_states
from src.gate.train_all_layers import sentence_only_states
from src.gate.calibrate_tau5 import load_calibration_rows
from src.intervention import run_val_intervention as intervention
from src.models.base import GenerationResult, generation_stop_reason
from src.models.probing import BoundaryConfirmationTracker, BOUNDARY_VERSION

ROOT = Path(__file__).resolve().parents[1]


def selected_nodes(file, names, scope):
    tree = ast.parse((ROOT / file).read_text())
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future] + nodes, type_ignores=[])
            ),
            file,
            "exec",
        ),
        scope,
    )
    return scope


class FakeDecoder:
    fingerprint = "synthetic-byte-decoder"

    def __init__(self, pieces):
        self.pieces = pieces

    def __call__(self, token):
        return self.pieces[int(token) - 1]


class FakeModule:
    def __init__(self):
        self.pre, self.post = [], []

    def register_forward_pre_hook(self, fn):
        self.pre.append(fn)
        return SimpleNamespace(remove=lambda: self.pre.remove(fn))

    def register_forward_hook(self, fn):
        self.post.append(fn)
        return SimpleNamespace(remove=lambda: self.post.remove(fn))

    def call(self, inputs, output):
        for fn in self.pre:
            fn(self, (inputs,))
        for fn in self.post:
            fn(self, (inputs,), output)


class HookAlignmentTests(unittest.TestCase):
    def check_class(self, model):
        scope = selected_nodes(
            f"src/models/{model}.py",
            {"tensor_from_output", "AllLayerSelectiveCapture"},
            dict(torch=torch, BoundaryConfirmationTracker=BoundaryConfirmationTracker),
        )
        embed, layers, norm = FakeModule(), [FakeModule(), FakeModule()], FakeModule()
        pieces = ["Done", ".", " Next", "\nFINAL: 42", "<eos>"]
        capture_hook = scope["AllLayerSelectiveCapture"](
            embed, layers, norm, 2, (), FakeDecoder(pieces), True
        )
        with capture_hook:
            embed.call(torch.tensor([[8, 9]]), None)  # prefill is not generated output
            for k in range(1, 5):  # EOS ID 5 is emitted but never fed back
                embed.call(torch.tensor([[k]]), None)
                for index, layer in enumerate(layers):
                    layer.call(
                        None, torch.full((1, 1, 2), 10 * k + index, dtype=torch.float32)
                    )
                norm.call(None, torch.full((1, 1, 2), 100 + k, dtype=torch.float32))
        self.assertEqual(capture_hook.sentence_positions, [3])
        self.assertEqual(
            capture_hook.all_layers[3].tolist(), [[30.0, 30.0], [31.0, 31.0]]
        )
        self.assertEqual(capture_hook.final_norm_states[3].tolist(), [103.0, 103.0])
        self.assertEqual(capture_hook.decode_token_ids, [1, 2, 3, 4])
        self.assertEqual(capture_hook.boundary_tracker.stop_event["token_index"], 4)
        self.assertEqual(embed.pre, [])
        self.assertEqual(norm.post, [])

    def test_ar_consumed_token_state_alignment(self):
        self.check_class("audio_reasoner")

    def test_moss_consumed_token_state_alignment(self):
        self.check_class("moss")

    def test_af_consumed_token_state_alignment(self):
        tree = ast.parse((ROOT / "src/models/af_next.py").read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "AFNextModel"
        )
        node = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "_generate_full"
        )
        scope = dict(
            torch=torch,
            GenerationResult=GenerationResult,
            BoundaryConfirmationTracker=BoundaryConfirmationTracker,
            BOUNDARY_VERSION=BOUNDARY_VERSION,
            generation_stop_reason=generation_stop_reason,
            TokenTextDecoder=lambda tok: FakeDecoder(tok),
        )
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                "fake_af_method",
                "exec",
            ),
            scope,
        )
        embed, layers, norm = FakeModule(), [FakeModule(), FakeModule()], FakeModule()

        def generate(**kwargs):
            embed.call(torch.tensor([[8, 9]]), None)
            for k in range(1, 5):
                embed.call(torch.tensor([[k]]), None)
                for index, layer in enumerate(layers):
                    layer.call(None, torch.full((1, 1, 2), 10 * k + index))
                norm.call(None, torch.full((1, 1, 2), 100 + k))
            return torch.tensor([[8, 9, 1, 2, 3, 4, 5]])

        fake = SimpleNamespace(
            _prepare_batch=lambda **kwargs: {"input_ids": torch.tensor([[8, 9]])},
            _tokenizer=["Done", ".", " Next", "\nFINAL: 42", "<eos>"],
            model_name="af_next",
            hidden_dim=2,
            _decoder_layers=layers,
            _final_norm=norm,
            _embed_tokens=embed,
            model=SimpleNamespace(
                generate=generate, generation_config=SimpleNamespace(eos_token_id=5)
            ),
            processor=SimpleNamespace(
                batch_decode=lambda *args, **kwargs: ["Done. Next\nFINAL: 42"]
            ),
        )
        result = scope["_generate_full"](fake, "unused", TASK_PROMPT, 5, (), True)
        self.assertEqual(result.token_ids, [1, 2, 3, 4, 5])
        self.assertEqual(result.sentence_checkpoints, (3,))
        self.assertEqual(result.all_layers[3].tolist(), [[30.0, 30.0], [31.0, 31.0]])
        self.assertEqual(result.final_norm[3].tolist(), [103.0, 103.0])
        self.assertEqual(embed.pre, [])


class StorageAndSplitTests(unittest.TestCase):
    def setUp(self):
        use_synthetic_umwp(self)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.split = self.root / "split.json"
        self.split.write_text(
            json.dumps(
                dict(
                    gate_train=["train"],
                    calibration=["cal"],
                    val=["val"],
                    test=["synthetic_test"],
                )
            )
        )
        self.config = replace(
            get_dataset_config("umwp"),
            split_path=self.split,
            trajectory_root=self.root / "trajectories",
        )

    def row(self, pair, condition, split):
        trial = f"{pair}_{condition[0]}"
        state = self.root / f"{trial}.pt"
        result = GenerationResult(
            "af_next",
            [11, 99],
            "FINAL: I don't know.",
            2,
            "eos",
            {},
            {},
            num_layers=2,
            hidden_size=3,
            boundary_version=BOUNDARY_VERSION,
        )
        trajectory = Trajectory(
            trial,
            pair,
            condition,
            split,
            "synthetic",
            1,
            "fake",
            "unused.wav",
            "Question",
            "Question",
            TASK_PROMPT,
            (42,) if condition == "answerable" else (),
            result,
        )
        path = self.root / f"{trial}.jsonl"
        save_trajectory(trajectory, state, path)
        return json.loads(path.read_text())

    def test_dimensioned_zero_state_roundtrip_and_both_loaders(self):
        row = self.row("val", "answerable", "val")
        restored = metadata_to_trajectory(row)
        self.assertEqual(restored.generation.token_ids, [11, 99])
        self.assertEqual(restored.generation.text, row["text"])
        self.assertEqual(restored.generation.num_layers, 2)
        for loader in (load_sentence_states, sentence_only_states):
            self.assertEqual(
                tuple(loader(row, self.root / "metadata.jsonl", 2).shape), (0, 2, 3)
            )
        self.assertFalse(row["triggered"])
        self.assertIsNone(row["brake_checkpoint"])

    def test_missing_zero_state_file_is_failure_not_zero_coverage(self):
        row = self.row("val", "answerable", "val")
        Path(row["state_path"]).unlink()
        with self.assertRaises(FileNotFoundError):
            load_sentence_states(row, self.root / "meta", 2)

    def test_capture_filters_removed_pair_ids(self):
        mapping = capture.load_split_map(self.split)
        manifest = [
            dict(pair_id=p, condition=c)
            for p in ("train", "removed", "val")
            for c in ("answerable", "unanswerable")
        ]
        selected = capture.select_rows(manifest, mapping, "gate_train", "all", None)
        self.assertEqual(len(selected), 2)
        self.assertEqual({r["pair_id"] for r in selected}, {"train"})

    def test_downstream_membership_all_split_roles(self):
        for split, pair in [
            ("gate_train", "train"),
            ("calibration", "cal"),
            ("val", "val"),
            ("test", "synthetic_test"),
        ]:
            conditions = (
                ["answerable"]
                if split == "calibration"
                else ["answerable", "unanswerable"]
            )
            rows = [
                dict(
                    trial_id=f"{pair}_{c}",
                    pair_id=pair,
                    split=split,
                    condition=c,
                    model_name="af_next",
                )
                for c in conditions
            ]
            validate_trajectory_rows(rows, self.config, split, "af_next")
            rows[0]["pair_id"] = "removed"
            with self.assertRaisesRegex(RuntimeError, "outside frozen"):
                validate_trajectory_rows(rows, self.config, split, "af_next")

    def test_calibration_loader_dynamic_count_preserves_zero_A(self):
        row = self.row("cal", "answerable", "calibration")
        path = self.config.trajectory_root / "af_next/calibration/metadata.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(row) + "\n")
        rows, _ = load_calibration_rows("af_next", self.config)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["eligible_checkpoint_count"], 0)
        row["pair_id"] = "removed"
        path.write_text(json.dumps(row) + "\n")
        with self.assertRaisesRegex(RuntimeError, "not in final calibration"):
            load_calibration_rows("af_next", self.config)

    def test_zero_validation_plan_preserves_A_U_and_all_seeds_without_gate(self):
        rows = [self.row("val", c, "val") for c in ("answerable", "unanswerable")]
        path = self.config.trajectory_root / "af_next/val/metadata.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

        def forbidden_gate(*args):
            raise AssertionError("Zero states must not call a gate")

        with (
            patch.object(intervention, "load_selected_layer", return_value=(1, {})),
            patch.object(
                intervention,
                "load_tau5",
                return_value=(
                    {s: np.nextafter(0.0, np.inf) for s in intervention.SEEDS},
                    {},
                ),
            ),
            patch.object(
                intervention, "load_gate", return_value=(forbidden_gate, None, None)
            ),
            patch.object(intervention, "verify_against_diagnostic"),
            patch.object(intervention, "MODEL_LAYERS", {"af_next": 2}),
        ):
            _, _, _, _, planned, counts = intervention.plan_interventions(
                "af_next", self.config
            )
        self.assertEqual(counts, (1, 1, 1))
        for records in planned.values():
            self.assertEqual(len(records), 2)
            self.assertTrue(
                all(
                    not r["triggered"]
                    and r["trajectory_score"] == 0.0
                    and r["brake_checkpoint"] is None
                    for r in records
                )
            )

    def test_frozen_manifest_hash_guard_is_read_only(self):
        import hashlib
        from src.data import frozen_umwp

        original = self.split.read_bytes()
        expected = hashlib.sha256(original).hexdigest()
        with (
            patch.object(frozen_umwp, "CANONICAL_SPLIT", self.split),
            patch.object(
                frozen_umwp, "frozen_split_record", return_value={"sha256": expected}
            ),
        ):
            frozen_umwp.verify_frozen_split(self.split)
            self.split.write_text(self.split.read_text() + "\n")
            changed = self.split.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "Frozen UMWP manifest changed"):
                capture.load_split_map(self.split)
            self.assertEqual(self.split.read_bytes(), changed)

    def test_v10_replay_uses_exact_stored_initial_prompt(self):
        for module_name in ("run_val_intervention", "run_test_intervention"):
            from importlib import import_module

            module = import_module("src.intervention." + module_name)
            for condition in ("answerable", "unanswerable"):
                self.assertEqual(
                    module.resolve_prompt_text(
                        {"prompt_text": TASK_PROMPT, "condition": condition}
                    ),
                    capture.TASK_PROMPT,
                )

    def test_missing_retained_manifest_member_fails_before_capture(self):
        from src.capture.umwp_validation import validate_manifest

        with self.assertRaisesRegex(RuntimeError, "required recordings missing"):
            validate_manifest([], [], {"pair": "gate_train"})

    def test_AR_rejects_over_limit_wave_before_processing(self):
        from src.models.audio_delivery import validate_audio_delivery

        path = self.root / "oversized.wav"
        sf.write(path, np.zeros(480001, dtype=np.float32), 16000)
        with self.assertRaisesRegex(ValueError, "silent"):
            validate_audio_delivery(path, "audio-reasoner")
        sf.write(path, np.zeros(480000, dtype=np.float32), 16000)
        self.assertEqual(validate_audio_delivery(path, "audio-reasoner").frames, 480000)


if __name__ == "__main__":
    unittest.main()
