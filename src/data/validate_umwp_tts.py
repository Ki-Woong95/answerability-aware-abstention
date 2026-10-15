import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 24000
pairs = [json.loads(x) for x in Path("data/processed/umwp_pairs.jsonl").open()]
manifest = [json.loads(x) for x in Path("data/tts/umwp_manifest.jsonl").open()]
split_data = json.load(open("data/splits/umwp_final_split.json"))

pair_map = {x["pair_id"]: x for x in pairs}
pair_to_split = {
    pid: split
    for split in ["gate_train", "calibration", "val", "test"]
    for pid in split_data[split]
}

errors = []
seen_trials = set()
seen_paths = set()
by_pair = defaultdict(set)
durations = defaultdict(list)

for row in manifest:
    tid, pid = row["trial_id"], row["pair_id"]

    if tid in seen_trials:
        errors.append(f"duplicate trial_id: {tid}")
    seen_trials.add(tid)

    if row["audio_path"] in seen_paths:
        errors.append(f"duplicate audio_path: {row['audio_path']}")
    seen_paths.add(row["audio_path"])

    if pid not in pair_map:
        errors.append(f"unknown pair_id: {pid}")
        continue

    if row["split"] != pair_to_split[pid]:
        errors.append(f"split mismatch: {tid}")

    condition = row["condition"]
    if condition not in {"answerable", "unanswerable"}:
        errors.append(f"bad condition: {tid} {condition}")
        continue

    by_pair[pid].add(condition)
    path = Path(row["audio_path"])

    if not path.exists():
        errors.append(f"missing audio: {tid}")
        continue

    try:
        info = sf.info(path)
    except Exception as e:
        errors.append(f"unreadable audio: {tid} {e}")
        continue

    if info.samplerate != SR:
        errors.append(f"bad sample rate: {tid} {info.samplerate}")
    if info.channels != 1:
        errors.append(f"not mono: {tid} channels={info.channels}")
    if info.frames <= 0 or info.duration <= 0:
        errors.append(f"empty audio: {tid}")
    if not np.isfinite(info.duration):
        errors.append(f"bad duration: {tid}")

    durations[condition].append(info.duration)

for pid in pair_map:
    if by_pair[pid] != {"answerable", "unanswerable"}:
        errors.append(f"incomplete pair: {pid} {sorted(by_pair[pid])}")

print("Manifest rows:", len(manifest))
print("Expected rows:", len(pairs) * 2)
print("Unique trials:", len(seen_trials))
print(
    "Complete pairs:",
    sum(by_pair[x] == {"answerable", "unanswerable"} for x in pair_map),
)
print("Errors:", len(errors))

print("\nSplit trials:")
print(Counter(x["split"] for x in manifest))

print("\nSources:")
print(Counter(x["source"] for x in manifest))

print("\nCategories:")
print(Counter(x["category"] for x in manifest if x["condition"] == "unanswerable"))

for condition in ["answerable", "unanswerable"]:
    x = np.array(durations[condition])
    print(
        f"\n{condition}: N={len(x)} mean={x.mean():.2f}s median={np.median(x):.2f}s min={x.min():.2f}s max={x.max():.2f}s"
    )

ratios = []
manifest_map = {(x["pair_id"], x["condition"]): x for x in manifest}

for pid in pair_map:
    a = manifest_map.get((pid, "answerable"))
    u = manifest_map.get((pid, "unanswerable"))
    if a and u and a["duration"] > 0:
        ratios.append(u["duration"] / a["duration"])

if ratios:
    ratios = np.array(ratios)
    print(
        f"\nU/A duration ratio: mean={ratios.mean():.3f} median={np.median(ratios):.3f}"
    )

if errors:
    print("\nFirst errors:")
    for e in errors[:20]:
        print("-", e)
    raise SystemExit(1)

if len(manifest) != 5200:
    raise SystemExit(f"Expected 5200 manifest rows, got {len(manifest)}")

print("\nALL TTS CHECKS PASSED")
