import json, random
from collections import defaultdict, Counter
from pathlib import Path

SEED = 20261002

pairs = [json.loads(x) for x in Path("data/processed/umwp_pairs.jsonl").open()]
pair_map = {x["pair_id"]: x for x in pairs}
base = json.load(Path("data/splits/umwp_split.json").open())

train_ids = base["splits"]["train"]
groups = defaultdict(list)

for pid in train_ids:
    x = pair_map[pid]
    groups[(x["source"], x["category"])].append(pid)

rng = random.Random(SEED)
gate_train, calibration = [], []

for key, ids in sorted(groups.items()):
    rng.shuffle(ids)
    n_cal = round(len(ids) * 0.20)
    calibration += ids[:n_cal]
    gate_train += ids[n_cal:]

rng.shuffle(gate_train)
rng.shuffle(calibration)

final = {
    "outer_split_seed": base["seed"],
    "train_calibration_seed": SEED,
    "stratify": ["source", "category"],
    "gate_train": gate_train,
    "calibration": calibration,
    "val": base["splits"]["val"],
    "test": base["splits"]["test"],
}

out = Path("data/splits/umwp_final_split.json")
json.dump(final, out.open("w"), indent=2)

for name in ["gate_train", "calibration", "val", "test"]:
    ids = final[name]
    print(f"\n{name}: {len(ids)} pairs / {len(ids)*2} trials")
    print("sources:", Counter(pair_map[i]["source"] for i in ids))
    print("categories:", Counter(pair_map[i]["category"] for i in ids))

sets = {k: set(final[k]) for k in ["gate_train", "calibration", "val", "test"]}
names = list(sets)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        print(f"{names[i]} ∩ {names[j]}:", len(sets[names[i]] & sets[names[j]]))

print("\nTotal unique pairs:", len(set().union(*sets.values())))
