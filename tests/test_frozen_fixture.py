"""Check that source-only fixtures retain the production integrity checks."""

import unittest
from pathlib import Path

from frozen_fixture import use_synthetic_umwp
from src.data import frozen_umwp
from src.gate.datasets import get_dataset_config, load_split_ids


class FrozenFixtureTests(unittest.TestCase):
    def setUp(self):
        use_synthetic_umwp(self)

    def test_fixture_keeps_frozen_hash_verification_enabled(self):
        frozen_umwp.verify_frozen_split(frozen_umwp.CANONICAL_SPLIT)
        frozen_umwp.CANONICAL_SPLIT.write_text('{"splits": {}}')
        with self.assertRaisesRegex(RuntimeError, "Frozen UMWP manifest changed"):
            frozen_umwp.verify_frozen_split(frozen_umwp.CANONICAL_SPLIT)

    def test_fixture_uses_temporary_split_and_real_membership_loader(self):
        root = Path(__file__).resolve().parents[1]
        self.assertNotEqual(
            frozen_umwp.CANONICAL_SPLIT,
            root / "data/splits/umwp_final_split.json",
        )
        self.assertEqual(
            load_split_ids(get_dataset_config("umwp"), "gate_train"),
            {"synthetic_gate_train"},
        )


if __name__ == "__main__":
    unittest.main()
