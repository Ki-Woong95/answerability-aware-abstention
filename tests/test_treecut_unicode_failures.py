import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.models.token_text import IncrementalUTF8TextDecoder, TokenTextDecoder
from src.models.probing import BoundaryConfirmationTracker
from src.capture.treecut_failures import (
    capture_population,
    load_failures,
    recoverable_generation_failure,
    write_failure,
)


class ByteTokenizer:
    all_special_ids = [256]
    byte_decoder = True

    def __init__(self):
        values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        chars = values[:]
        n = 0
        for value in range(256):
            if value not in values:
                values.append(value)
                chars.append(256 + n)
                n += 1
        self.tokens = dict(zip(values, map(chr, chars)))
        self.tokens[256] = "<end>"

    def get_vocab(self):
        return {v: k for k, v in self.tokens.items()}

    def convert_ids_to_tokens(self, i):
        return self.tokens[i]


class UnicodeFailureTests(unittest.TestCase):
    def test_native_replacement_and_delayed_valid_utf8(self):
        rng = random.Random(17)
        for raw in [
            b"hello\x91 world. ",
            "한글 é ‘text’. ".encode(),
            b"\xe2\x80",
            b"\xe2x. ",
        ] + [rng.randbytes(40) for _ in range(40)]:
            d = IncrementalUTF8TextDecoder("replace")
            output = "".join(d.feed_bytes(bytes([b])) for b in raw) + d.feed_bytes(
                b"", final=True
            )
            self.assertEqual(output, raw.decode("utf-8", errors="replace"))
        d = IncrementalUTF8TextDecoder("replace")
        self.assertEqual(d.feed_bytes(b"\xe2"), "")
        self.assertIsNone(d.last_error)
        self.assertEqual(d.feed_bytes(b"\x80"), "")
        self.assertEqual(d.feed_bytes(b"\x91"), "‑")
        self.assertIsNone(d.last_error)
        with self.assertRaises(UnicodeDecodeError):
            IncrementalUTF8TextDecoder().feed_bytes(b"\x91")

    def test_malformed_bytes_keep_checkpoint_token_and_replay_prefix_alignment(self):
        decoder = TokenTextDecoder(ByteTokenizer(), errors="replace")
        tracker = BoundaryConfirmationTracker(decoder, "af_next")
        raw = b"Bad\x91 text. Next step"
        for token_index, token_id in enumerate(raw, 1):
            event = tracker.observe(token_id)
            if event:
                checkpoint = tracker.events[-1]
                self.assertEqual(checkpoint["confirmation_token_index"], token_index)
                self.assertEqual(
                    tracker.text, raw[:token_index].decode("utf-8", errors="replace")
                )
        self.assertEqual(tracker.position, len(raw))
        self.assertEqual(decoder.observed_token_ids, list(raw))
        self.assertEqual(decoder.replacement_events[0]["token_index"], 4)
        self.assertEqual(decoder.replacement_events[0]["raw_bytes_hex"], "91")
        self.assertTrue(tracker.events)

    def test_special_marker_commits_invalid_pending_bytes_at_observed_token(self):
        decoder = TokenTextDecoder(ByteTokenizer(), errors="replace")
        self.assertEqual(decoder(0xE2), "")
        self.assertEqual(decoder(256), "�<end>")
        self.assertEqual(decoder.replacement_events[0]["token_index"], 2)
        self.assertEqual(decoder(65), "A")

    def test_pair_failure_resume_coverage_and_stale_records(self):
        rows = [
            dict(pair_id=p, trial_id=p + s, condition=c, split="gate_train")
            for p in ["p", "q"]
            for c, s in [("answerable", "A"), ("unanswerable", "U")]
        ]
        with tempfile.TemporaryDirectory() as d:
            prov = {"frozen_split_sha256": "frozen"}
            write_failure(
                d,
                rows[1],
                "af_next",
                prov,
                ValueError("invalid output"),
                SimpleNamespace(),
            )
            active, report = capture_population(rows, d, "af_next", prov)
            self.assertEqual({r["pair_id"] for r in active}, {"q"})
            self.assertEqual(report["conditions"]["answerable"]["excluded"], 1)
            self.assertEqual(report["conditions"]["unanswerable"]["excluded"], 1)
            self.assertEqual(report["conditions"]["answerable"]["exclusion_rate"], 0.5)
            self.assertEqual(set(load_failures(d, rows, "af_next", prov)), {"p"})
            with self.assertRaises(RuntimeError):
                load_failures(d, rows, "af_next", {"frozen_split_sha256": "changed"})
            with self.assertRaises(RuntimeError):
                load_failures(d, rows, "moss", prov)
            # Files for a successfully captured mate are not deleted by population filtering.
            saved = Path(d) / "mate.pt"
            saved.write_bytes(b"preserved")
            capture_population(rows, d, "af_next", prov)
            self.assertEqual(saved.read_bytes(), b"preserved")

    def test_resource_errors_do_not_get_skipped(self):
        self.assertTrue(
            recoverable_generation_failure(
                UnicodeDecodeError("utf-8", b"\x91", 0, 1, "bad")
            )
        )
        self.assertTrue(
            recoverable_generation_failure(ValueError("special interrupts UTF-8"))
        )
        for exc in [
            MemoryError(),
            OSError("disk full"),
            RuntimeError("CUDA error: illegal memory access"),
            ValueError("checkpoint alignment"),
        ]:
            self.assertFalse(recoverable_generation_failure(exc))

    def test_capture_continues_after_failed_U_and_resume_preserves_A(self):
        from dataclasses import replace
        import hashlib
        from src.capture.run_treecut import run
        from src.models.base import GenerationResult
        from src.models.probing import BOUNDARY_VERSION
        from src.gate.datasets import get_dataset_config

        fingerprint = json.loads(
            Path("docs/provenance/phase2b/af_next_tokenizer.json").read_text()
        )["tokenizer_fingerprint"]
        calls = []

        class Fake:
            hidden_dim = 3

            def load(self):
                pass

            def generate_full(self, *args, **kwargs):
                calls.append(len(calls) + 1)
                if len(calls) == 2:
                    raise UnicodeDecodeError("utf-8", b"\x91", 0, 1, "bad byte")
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
            rows = [
                dict(
                    dataset="treecut",
                    pair_id=p,
                    trial_id=p + s,
                    condition=c,
                    split="gate_train",
                    source="treecut",
                    voice="af_heart",
                    audio_path=str(wav),
                    audio_sha256=hashlib.sha256(wav.read_bytes()).hexdigest(),
                    raw_text="synthetic",
                    tts_text="synthetic",
                    gold_answer=[],
                )
                for p in ["p", "q"]
                for c, s in [("answerable", "A"), ("unanswerable", "U")]
            ]
            split = d / "split.json"
            split.write_text(json.dumps({"splits": {"gate_train": ["p", "q"]}}))
            cfg = replace(get_dataset_config("treecut"), split_path=split)
            prov = {"boundary_version": BOUNDARY_VERSION}
            args = SimpleNamespace(
                model="af_next",
                split="gate_train",
                allow_test=False,
                limit_pairs=2,
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
                dest = args.output / "af_next/gate_train"
                saved = [
                    json.loads(l)
                    for l in (dest / "metadata.jsonl").read_text().splitlines()
                ]
                self.assertEqual({r["trial_id"] for r in saved}, {"pA", "qA", "qU"})
                before = (dest / "states/pA.pt").read_bytes()
                args.resume = True
                run(
                    args,
                    lambda _: self.fail("Accounted population must not reload model"),
                )
                self.assertEqual(before, (dest / "states/pA.pt").read_bytes())
                self.assertEqual(
                    len((dest / "generation_failures.jsonl").read_text().splitlines()),
                    1,
                )
                self.assertEqual(
                    json.loads((dest / "capture_coverage.json").read_text())[
                        "excluded_pairs"
                    ],
                    1,
                )


if __name__ == "__main__":
    unittest.main()
