"""Offline CPU native processor/template pilots on gate_train audio only.

Instantiates processors/tokenizers/configs, never a neural backbone or PtEngine.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from transformers import AutoProcessor, AutoConfig
from src.capture.v10_prompt import TASK_PROMPT
from src.models.audio_delivery import validate_audio_delivery

torch.set_num_threads(1)
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
args = parser.parse_args()
root = Path.cwd()
audit = json.loads(Path("docs/provenance/phase2b/audio_delivery.json").read_text())
manifest = {
    r["trial_id"]: r
    for r in map(
        json.loads, Path("data/tts/umwp_manifest.jsonl").read_text().splitlines()
    )
}
ledger = Path(audit["inventory_path"])
assert hashlib.sha256(ledger.read_bytes()).hexdigest() == audit["inventory_sha256"]
inventory = [json.loads(line) for line in ledger.read_text().splitlines()]
pilots = []
for condition in ("answerable", "unanswerable"):
    choices = [
        x
        for x in inventory
        if x["split"] == "gate_train" and x["condition"] == condition
    ]
    pilots.append(max(choices, key=lambda x: x["resampled_samples"]))
record = json.loads(
    Path(f"docs/provenance/phase2b/{args.model}_tokenizer.json").read_text()
)
path = record["tokenizer_path"]
if args.model == "moss":
    sys.path.insert(0, str(root / "MOSS-Audio/src"))
    from processing_moss_audio import MossAudioProcessor

    processor = MossAudioProcessor.from_pretrained(
        path, local_files_only=True, enable_time_marker=True
    )
else:
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
reports = []
for pilot in pilots:
    row = manifest[pilot["trial_id"]]
    validate_audio_delivery(row["audio_path"], args.model)
    if args.model == "af_next":
        from src.models.af_next import AFNextModel

        model = AFNextModel("unused", "{question}")
        model.processor = processor
        model.model = SimpleNamespace(device=torch.device("cpu"), dtype=torch.float32)
        vanilla = model._prepare_batch(row["audio_path"], TASK_PROMPT)
        from src.intervention.af_next import AFNextInterventionRunner

        replay = AFNextInterventionRunner.__new__(AFNextInterventionRunner)
        replay.processor = processor
        replay.model = model.model
        gate = replay.prepare_base_batch(row["audio_path"], TASK_PROMPT)
    elif args.model == "moss":
        from src.models.moss import MOSSModel
        from src.intervention.moss import MOSSInterventionRunner

        model = MOSSModel()
        model.processor = processor
        dummy = SimpleNamespace(parameters=lambda: iter([torch.empty(0)]))
        model.model = SimpleNamespace(language_model=dummy, dtype=torch.float32)
        vanilla = model._prepare_inputs(row["audio_path"], TASK_PROMPT)
        replay = MOSSInterventionRunner.__new__(MOSSInterventionRunner)
        replay.processor = processor
        replay.model = model.model
        gate = replay.prepare_base_batch(row["audio_path"], TASK_PROMPT)
    else:
        from swift.llm import get_template, InferRequest
        from src.models.audio_reasoner import AudioReasonerModel, AR_SYSTEM
        from src.intervention.audio_reasoner import AudioReasonerInterventionRunner

        processor.model_info = SimpleNamespace(
            config=AutoConfig.from_pretrained(path, local_files_only=True),
            is_multimodal=True,
        )
        processor.model_meta = SimpleNamespace(is_multimodal=True)
        template = get_template("qwen2_audio", processor, default_system=AR_SYSTEM)
        model = AudioReasonerModel()
        replay = AudioReasonerInterventionRunner.__new__(
            AudioReasonerInterventionRunner
        )
        a = model._messages(row["audio_path"], TASK_PROMPT)
        b = replay._messages(row["audio_path"], TASK_PROMPT)
        assert a == b
        vanilla = template.encode(InferRequest(messages=a))
        gate = template.encode(InferRequest(messages=b))
    a = vanilla["input_ids"]
    b = gate["input_ids"]
    if torch.is_tensor(a):
        a = a.tolist()
        b = b.tolist()
    assert a == b, "Vanilla/Gate initial native tokens differ"
    flat = a[0] if a and isinstance(a[0], list) else a
    assert TASK_PROMPT in processor.tokenizer.decode(
        flat, skip_special_tokens=False, clean_up_tokenization_spaces=False
    ), "Native template altered or omitted initial task prompt"
    shapes = {k: list(v.shape) for k, v in vanilla.items() if torch.is_tensor(v)}
    token_count = len(a[0]) if a and isinstance(a[0], list) else len(a)
    if args.model == "audio-reasoner":
        # Qwen2-Audio conv stride 2 then pool stride 2 expands the placeholder.
        frames = int(vanilla["feature_attention_mask"].sum())
        conv = (frames - 1) // 2 + 1
        audio_tokens = (conv - 2) // 2 + 1
        effective_tokens = token_count - 1 + audio_tokens
        assert effective_tokens + 2048 <= 8192
        feature = vanilla["input_features"]
        mask = vanilla["feature_attention_mask"]
        assert int(mask.sum()) == int(np.ceil(pilot["resampled_samples"] / 160))
        import librosa

        audio, _ = librosa.load(row["audio_path"], sr=16000)
        untruncated = processor.feature_extractor(
            audio,
            sampling_rate=16000,
            truncation=False,
            return_attention_mask=True,
            return_tensors="pt",
        )
        assert torch.equal(feature, untruncated["input_features"])
    for k, v in vanilla.items():
        if torch.is_tensor(v) and k in gate:
            assert torch.equal(v, gate[k]), f"Native feature mismatch: {k}"
    reports.append(
        dict(
            trial_id=row["trial_id"],
            condition=row["condition"],
            resampled_samples=pilot["resampled_samples"],
            native_input_token_count=token_count,
            shapes=shapes,
            vanilla_gate_inputs_identical=True,
            exact_initial_prompt_present=True,
            effective_input_tokens_including_audio=(
                effective_tokens if args.model == "audio-reasoner" else token_count
            ),
        )
    )
result = dict(
    model=args.model,
    cpu_only=True,
    neural_model_initialized=False,
    gate_train_only=True,
    passed=True,
    pilots=reports,
)
Path(f"docs/provenance/phase2b/{args.model}_processor.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(json.dumps(result, indent=2))
