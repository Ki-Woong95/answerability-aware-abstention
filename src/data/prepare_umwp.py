import json
from pathlib import Path

src = Path("data/raw/umwp/StandardDataset.jsonl")
out = Path("data/processed/umwp_pairs.jsonl")
rows = [json.loads(x) for x in src.open()]
A = {x["id"]: x for x in rows if x["answerable"]}
U = [x for x in rows if not x["answerable"]]

out.parent.mkdir(parents=True, exist_ok=True)

with out.open("w") as f:
    for u in U:
        aid = u["relevant_ids"][0]
        a = A[aid]
        pair = {
            "pair_id": f"umwp_{aid}",
            "source": u["source"],
            "category": u["category"],
            "answerable": {
                "id": a["id"],
                "question": a["question"],
                "answer": a["answer"],
            },
            "unanswerable": {"id": u["id"], "question": u["question"]},
        }
        f.write(json.dumps(pair) + "\n")

print("Pairs:", len(U))
print("Saved:", out)
