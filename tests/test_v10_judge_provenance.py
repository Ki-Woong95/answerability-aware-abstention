from frozen_fixture import use_synthetic_umwp
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from src.data.frozen_umwp import v10_artifact_provenance
from src.eval.build_judge_inputs import common_fields
from src.eval import run_judge
from src.gate.datasets import get_dataset_config, validate_artifact_dataset


class JudgeProvenanceTests(unittest.TestCase):
    def setUp(self):
        use_synthetic_umwp(self)

    def test_inputs_and_saved_judgments_preserve_model_bound_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = replace(get_dataset_config("umwp"), judge_root=Path(directory))
            rows = []
            for model in ("af_next", "moss", "audio-reasoner"):
                row = common_fields(
                    model,
                    dict(trial_id="synthetic_A", condition="answerable"),
                    dict(gold_answer=[42]),
                    cfg,
                )
                row.update(response_text="FINAL: 42", hit_token_limit=False)
                validate_artifact_dataset(row, cfg, Path(directory), model_name=model)
                rows.append(row)
            path = Path(directory) / "val_vanilla.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with (
                patch.object(run_judge, "get_dataset_config", return_value=cfg),
                patch.object(run_judge, "OpenAI"),
                patch.object(
                    run_judge,
                    "judge_one",
                    return_value=(
                        dict(label="correct", extracted_answer="42", note="Synthetic."),
                        "synthetic",
                    ),
                ),
                patch(
                    "sys.argv",
                    [
                        "judge",
                        "--dataset",
                        "umwp",
                        "--split",
                        "val",
                        "--kind",
                        "vanilla",
                    ],
                ),
            ):
                run_judge.main()
            output = Path(directory) / "val_vanilla_judged_v2.jsonl"
            judgments = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(judgments), 3)
            for row in judgments:
                validate_artifact_dataset(row, cfg, output, model_name=row["model"])
                for key, value in v10_artifact_provenance("umwp", row["model"]).items():
                    self.assertEqual(row[key], value)
            stale = dict(judgments[-1])
            stale.pop("generation_stop_policy")
            with self.assertRaises(RuntimeError):
                validate_artifact_dataset(
                    stale, cfg, output, model_name="audio-reasoner"
                )
