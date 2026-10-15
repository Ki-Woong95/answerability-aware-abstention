import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.models import local_snapshot


class LocalSnapshotTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.snapshot = self.root / "snapshot"
        self.snapshot.mkdir()
        for name in ["config.json", "tokenizer.json", "model-1.safetensors"]:
            (self.snapshot / name).write_text("{}")
        (self.snapshot / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"layer.weight": "model-1.safetensors"}})
        )
        record = self.root / "docs/provenance/phase2b/audio-reasoner_tokenizer.json"
        record.parent.mkdir(parents=True)
        record.write_text(json.dumps({"tokenizer_path": str(self.snapshot)}))

    def test_default_id_resolves_to_audited_local_snapshot(self):
        with patch.object(local_snapshot, "ROOT", self.root):
            self.assertEqual(
                local_snapshot.resolve_audio_reasoner_snapshot(),
                str(self.snapshot.resolve()),
            )

    def test_explicit_local_snapshot_supported(self):
        self.assertEqual(
            local_snapshot.resolve_audio_reasoner_snapshot(self.snapshot),
            str(self.snapshot.resolve()),
        )

    def test_missing_shard_fails_before_hub_lookup(self):
        (self.snapshot / "model-1.safetensors").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "refusing remote download"):
            local_snapshot.resolve_audio_reasoner_snapshot(self.snapshot)

    def test_missing_local_snapshot_does_not_fall_back_to_remote(self):
        with self.assertRaisesRegex(FileNotFoundError, "refusing remote download"):
            local_snapshot.resolve_audio_reasoner_snapshot(self.root / "absent")


if __name__ == "__main__":
    unittest.main()
