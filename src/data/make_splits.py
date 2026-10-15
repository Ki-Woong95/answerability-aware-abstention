import json, random
from collections import defaultdict, Counter
from pathlib import Path

SEED = 20261001
src = Path("data/processed/umwp_pairs.jsonl")
out = Path("data/splits/umwp_split.json")

pairs = [json.loads(x) for x in src.open()]
groups = defaultdict(list)

for x in pairs:
    groups[(x["source"], x["category"])].append(x["pair_id"])

rng = random.Random(SEED)
splits = {"train": [], "val": [], "test": []}

for key, ids in sorted(groups.items()):
    rng.shuffle(ids)
    n = len(ids)
    n_val = round(n * 0.15)
    n_test = round(n * 0.15)
    n_train = n - n_val - n_test
    splits["train"] += ids[:n_train]
    splits["val"] += ids[n_train : n_train + n_val]
    splits["test"] += ids[n_train + n_val :]

for split in splits:
    rng.shuffle(splits[split])

out.parent.mkdir(parents=True, exist_ok=True)
json.dump(
    {"seed": SEED, "stratify": ["source", "category"], "splits": splits},
    out.open("w"),
    indent=2,
)

lookup = {x["pair_id"]: x for x in pairs}
for split, ids in splits.items():
    print(f"\n{split}: {len(ids)} pairs / {len(ids)*2} trials")
    print("sources:", Counter(lookup[i]["source"] for i in ids))
    print("categories:", Counter(lookup[i]["category"] for i in ids))
