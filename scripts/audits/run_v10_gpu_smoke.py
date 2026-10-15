"""Bounded gate_train A/U capture with independent GPU hook alignment checks.

No gate fitting, threshold calculation, intervention, judging or test evaluation.
Uses the audited local model snapshots and an isolated output root.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch

from src.capture import run as capture
from src.capture.storage import load_trajectory_states
from src.capture.v10_prompt import TASK_PROMPT
from src.data.frozen_umwp import verify_frozen_split
from src.models.probing import BoundaryConfirmationTracker
from src.models.token_text import TokenTextDecoder


class AlignmentAudit:
    """Observe consumed IDs and independently copy states at confirmed positions."""

    def __init__(self, model):
        self.model = model
        self.embed = getattr(model, "_embed_tokens", None) or model.embed_tokens
        self.layers = getattr(model, "_decoder_layers", None) or model.decoder_layers
        self.norm = getattr(model, "_final_norm", None) or model.final_norm
        tokenizer = getattr(model, "_tokenizer", None) or model.tokenizer
        decoder = TokenTextDecoder(
            tokenizer, errors=getattr(model, "token_text_errors", "strict")
        )
        self.tracker = BoundaryConfirmationTracker(
            decoder, model.model_name, decoder.native_stop_token_ids
        )
        self.ids, self.events, self.states, self.norm_states = [], [], {}, {}
        self.prefill = False
        self.active = None
        self.handles = []

    def embed_hook(self, module, args):
        self.active = None
        ids = args[0]
        if not self.prefill:
            self.prefill = True
            return
        if ids.ndim != 2 or tuple(ids.shape) != (1, 1):
            raise RuntimeError(
                f"Unexpected smoke decode embedding shape: {tuple(ids.shape)}"
            )
        self.ids.append(int(ids[0, 0]))
        events = self.tracker.observe(self.ids[-1])
        if events:
            self.active = len(self.ids)
            self.events.extend(events)
            self.states[self.active] = {}

    def state_hook(self, index):
        def hook(module, inputs, output):
            if self.active is None:
                return
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            state = tensor[0, -1].detach().float().cpu().clone()
            if index is None:
                self.norm_states[self.active] = state
            else:
                self.states[self.active][index] = state

        return hook

    def __enter__(self):
        self.handles.append(self.embed.register_forward_pre_hook(self.embed_hook))
        self.handles.extend(
            layer.register_forward_hook(self.state_hook(i))
            for i, layer in enumerate(self.layers)
        )
        self.handles.append(self.norm.register_forward_hook(self.state_hook(None)))
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()

    def verify(self, result):
        assert (
            self.ids == result.token_ids[: len(self.ids)]
        ), "Consumed/emitted ID mismatch"
        assert (
            len(self.ids) == result.consumed_generated_tokens
        ), "Consumed count mismatch"
        assert self.events == list(
            result.boundary_events
        ), "Causal boundary-event mismatch"
        assert (
            self.tracker.stop_event == result.monitoring_stop
        ), "Monitoring stop mismatch"
        assert sorted(self.states) == list(
            result.sentence_checkpoints
        ), "Checkpoint mismatch"
        for k in result.sentence_checkpoints:
            assert len(self.states[k]) == len(
                self.layers
            ), "Missing independent layer readout"
            state = torch.stack([self.states[k][i] for i in range(len(self.layers))])
            assert torch.equal(
                state, result.all_layers[k]
            ), f"Layer-state mismatch at {k}"
            assert torch.equal(
                self.norm_states[k], result.final_norm[k]
            ), f"Norm-state mismatch at {k}"
            assert (
                torch.isfinite(state).all()
                and torch.isfinite(self.norm_states[k]).all()
            )
        return dict(
            independent_hook_alignment_passed=True,
            consumed_tokens=len(self.ids),
            emitted_tokens=len(result.token_ids),
            eligible_checkpoints=len(self.states),
            monitoring_stop=result.monitoring_stop,
            stop_reason=result.stop_reason,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, choices=["af_next", "audio-reasoner", "moss"]
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / f"{args.model}_smoke.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    readiness = json.loads(
        Path("docs/provenance/phase2b/implementation_readiness.json").read_text()
    )
    assert readiness["cpu_regressions_passed"] and readiness["required_skips"] == 0
    assert (
        readiness["audio_delivery_passed"]
        and readiness["native_processor_prompt_equality_passed_for_all_models"]
    )
    verify_frozen_split(capture.DEFAULT_SPLITS)
    assert (
        torch.cuda.is_available() and torch.cuda.device_count() == 1
    ), "Expose exactly the assigned GPU"
    snapshot_record = json.loads(
        Path(f"docs/provenance/phase2b/{args.model}_tokenizer.json").read_text()
    )
    snapshot = Path(snapshot_record["tokenizer_path"])
    assert snapshot.is_dir()
    split_map = capture.load_split_map(capture.DEFAULT_SPLITS)
    manifest = capture.read_jsonl(capture.DEFAULT_MANIFEST)
    selected = capture.select_rows(manifest, split_map, "gate_train", "all", 1)
    assert len(selected) == 2 and {r["condition"] for r in selected} == {
        "answerable",
        "unanswerable",
    }
    report = dict(
        model=args.model,
        status="running",
        physical_gpu=os.environ["CUDA_VISIBLE_DEVICES"],
        gpu_name=torch.cuda.get_device_name(0),
        model_snapshot=str(snapshot),
        weight_revision=snapshot.name,
        tokenizer_audit=snapshot_record,
        task_prompt_sha256=hashlib.sha256(TASK_PROMPT.encode()).hexdigest(),
        selected_pair_id=selected[0]["pair_id"],
        selected_trial_ids=[r["trial_id"] for r in selected],
        scope="one gate_train A/U pair; no training, calibration, intervention or evaluation",
        checks=[],
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    def factory(name):
        model = capture.build_model(name)
        model.model_id = str(snapshot)
        original_generate = model.generate_full

        def audited_generate(**kwargs):
            with AlignmentAudit(model) as audit:
                result = original_generate(**kwargs)
            assert (
                result.tokenizer_fingerprint == snapshot_record["tokenizer_fingerprint"]
            )
            check = audit.verify(result)
            check["audio_path"] = kwargs["audio_path"]
            report["checks"].append(check)
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            return result

        model.generate_full = audited_generate
        return model

    try:
        capture.run(
            SimpleNamespace(
                model=args.model,
                split="gate_train",
                condition="all",
                limit_pairs=1,
                resume=False,
                allow_test=False,
                manifest=capture.DEFAULT_MANIFEST,
                pairs=capture.DEFAULT_PAIRS,
                splits=capture.DEFAULT_SPLITS,
                output=args.output,
            ),
            model_factory=factory,
        )
        rows = capture.read_jsonl(
            args.output / args.model / "gate_train" / "metadata.jsonl"
        )
        assert len(rows) == 2
        report["response_formats"] = []
        for row in rows:
            assert row["prompt_text"] == TASK_PROMPT
            payload = load_trajectory_states(row["state_path"])
            assert tuple(payload["all_layers"].shape) == (
                len(row["checkpoints"]),
                row["num_layers"],
                row["hidden_dim"],
            )
            assert list(payload["checkpoints"]) == row["checkpoints"]
            report["response_formats"].append(
                dict(trial_id=row["trial_id"], **row["response_format"])
            )
        verify_frozen_split(capture.DEFAULT_SPLITS)
        report["status"] = "capture_and_alignment_passed"
        report["format_review_required"] = any(
            not r["format_ok"] for r in report["response_formats"]
        )
        report["zero_checkpoint_trials"] = sum(
            c["eligible_checkpoints"] == 0 for c in report["checks"]
        )
        report["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated()
        print(json.dumps(report, indent=2), flush=True)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
