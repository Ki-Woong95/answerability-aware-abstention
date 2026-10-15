import json
from collections import Counter
from pathlib import Path

path = Path("data/raw/umwp/StandardDataset.jsonl")
rows = [json.loads(line) for line in path.open()]
A = [x for x in rows if x["answerable"]]
U = [x for x in rows if not x["answerable"]]

print("Total:", len(rows))
print("Answerable:", len(A))
print("Unanswerable:", len(U))
print("\nSources:", Counter(x["source"] for x in rows))
print("\nU categories:", Counter(x["category"] for x in U))

print("\nIDs contiguous:", [x["id"] for x in rows] == list(range(1, len(rows) + 1)))

n = len(rows) // 2
first, second = rows[:n], rows[n:]
print("First half all A:", all(x["answerable"] for x in first))
print("Second half all U:", all(not x["answerable"] for x in second))
print(
    "Same source by half-pair:",
    sum(a["source"] == u["source"] for a, u in zip(first, second)),
    "/",
    n,
)

print("\nRelevant_ids types:", Counter(type(x["relevant_ids"]).__name__ for x in U))

for category in sorted(set(x["category"] for x in U)):
    examples = [x for x in U if x["category"] == category][:2]
    print(f"\n{'='*80}\nCATEGORY: {category}")
    for x in examples:
        print(
            f"\nID: {x['id']}  source={x['source']}  relevant_ids={x['relevant_ids']}"
        )
        print("Q:", x["question"])
        print("A:", x["answer"])

a_by_id = {x["id"]: x for x in A}

lengths = Counter(len(x["relevant_ids"]) for x in U)
print("\nRelevant_ids lengths:", lengths)

missing = []
source_mismatch = []
zip_mismatch = []

for i, u in enumerate(U):
    rid = u["relevant_ids"][0] if len(u["relevant_ids"]) == 1 else None
    if rid not in a_by_id:
        missing.append((u["id"], rid))
        continue
    a = a_by_id[rid]
    if a["source"] != u["source"]:
        source_mismatch.append((a["id"], u["id"], a["source"], u["source"]))
    if A[i]["id"] != rid:
        zip_mismatch.append((i, A[i]["id"], u["id"], rid))

print("Missing referenced A:", len(missing))
print("Source mismatch via relevant_ids:", len(source_mismatch))
print("Zip pair != relevant_ids pair:", len(zip_mismatch))
print(
    "Unique referenced A ids:",
    len(set(x["relevant_ids"][0] for x in U if len(x["relevant_ids"]) == 1)),
)

import numpy as np

for category in sorted(set(x["category"] for x in U)):
    us = [x for x in U if x["category"] == category]
    diffs, ratios = [], []
    for u in us:
        a = a_by_id[u["relevant_ids"][0]]
        la, lu = len(a["question"].split()), len(u["question"].split())
        diffs.append(lu - la)
        ratios.append(lu / la)
    print(f"\nCategory {category}: N={len(us)}")
    print(
        f"Mean A words: {np.mean([len(a_by_id[u['relevant_ids'][0]]['question'].split()) for u in us]):.1f}"
    )
    print(f"Mean U words: {np.mean([len(u['question'].split()) for u in us]):.1f}")
    print(f"Mean U-A difference: {np.mean(diffs):.1f}")
    print(f"Median U/A ratio: {np.median(ratios):.3f}")
