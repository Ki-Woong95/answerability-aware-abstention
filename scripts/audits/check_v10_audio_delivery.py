"""CPU-only full-wave decoding/resampling audit; no models, judges or scoring.

Visits every retained recording (test audio QC only, never test performance).
Checks exact Swift librosa input length against Whisper's 480000-sample limit.
"""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import librosa
import numpy as np
import soundfile as sf
from src.data.frozen_umwp import verify_frozen_split, CANONICAL_SPLIT
from src.models.audio_delivery import validate_audio_delivery


def main():
    verify_frozen_split(CANONICAL_SPLIT)
    frozen_before = hashlib.sha256(CANONICAL_SPLIT.read_bytes()).hexdigest()
    split = json.loads(CANONICAL_SPLIT.read_text())
    split = split.get("splits", split)
    ids = {
        str(i): name
        for name in ("gate_train", "calibration", "val", "test")
        for i in split[name]
    }
    source = Path("data/tts/umwp_manifest.jsonl")
    rows = [
        json.loads(line) for line in source.read_text().splitlines() if line.strip()
    ]
    retained = [row for row in rows if row["pair_id"] in ids]
    inventory, failures, seen = [], [], set()
    for index, row in enumerate(retained, 1):
        try:
            key = (row["pair_id"], row["condition"])
            if key in seen:
                raise ValueError("Duplicate retained audio entry")
            seen.add(key)
            if row["split"] != ids[row["pair_id"]]:
                raise ValueError("Manifest split mismatch")
            path = Path(row["audio_path"])
            for model in ("af_next", "audio-reasoner", "moss"):
                info = validate_audio_delivery(path, model)
            audio, rate = librosa.load(str(path), sr=16000)
            expected = math.ceil(info.frames * 16000 / info.samplerate)
            if (
                len(audio) != expected
                or len(audio) > 480000
                or not np.isfinite(audio).all()
            ):
                raise ValueError("Full resampled audio length/nonfinite mismatch")
            inventory.append(
                dict(
                    trial_id=row["trial_id"],
                    split=row["split"],
                    condition=row["condition"],
                    source_samples=info.frames,
                    source_rate=info.samplerate,
                    resampled_samples=len(audio),
                    audio_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            )
        except Exception as exc:
            failures.append(dict(trial_id=row["trial_id"], error=str(exc)))
        if index % 500 == 0:
            print(f"CPU full-wave audit {index}/{len(retained)}", flush=True)
    expected_keys = {
        (pair, condition)
        for pair in ids
        for condition in ("answerable", "unanswerable")
    }
    missing = sorted(expected_keys - seen)
    frozen_after = hashlib.sha256(CANONICAL_SPLIT.read_bytes()).hexdigest()
    assert frozen_before == frozen_after
    report = dict(
        frozen_split_sha256=frozen_before,
        split_unchanged=True,
        manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        retained_pairs=len(ids),
        retained_recordings=len(retained),
        checked_recordings=len(inventory),
        recordings_by_split=dict(Counter(row["split"] for row in retained)),
        max_resampled_samples=max(
            (x["resampled_samples"] for x in inventory), default=0
        ),
        ar_sample_limit=480000,
        removed_manifest_rows=len(rows) - len(retained),
        full_waveform_read=True,
        swift_librosa_resampling_exercised=True,
        missing_recordings=missing,
        failures=failures,
        passed=not failures and not missing,
        gpu_inference=False,
        test_performance_accessed=False,
        evidence_scope="Every input waveform audited; feature routing/template integration remains a GPU smoke-test check.",
        inventory=inventory,
    )
    ledger = Path("data/audits/phase2b/retained_audio_inventory.jsonl")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n" for row in report.pop("inventory")
        )
    )
    report["inventory_path"] = str(ledger)
    report["inventory_sha256"] = hashlib.sha256(ledger.read_bytes()).hexdigest()
    output = Path("docs/provenance/phase2b/audio_delivery.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                **{
                    k: v
                    for k, v in report.items()
                    if k not in ("inventory", "failures")
                },
                "failure_count": len(failures),
                "failure_examples": failures[:5],
            },
            indent=2,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
