import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 17

TRAIN_QC = Path("external/worldsense/qc/train_completeness_qc_v2.csv")
TEST_QC = Path("external/worldsense/qc/test_completeness_qc_v2.csv")
SPLIT_PATH = Path("data/splits/worldsense_openended_split.json")

PAIR_OUT = Path("data/processed/worldsense_pairs_openended.jsonl")
TRIAL_OUT = Path("data/processed/worldsense_trials_openended.jsonl")

PAIRWISE = {
    "before_after": ("before", "after"),
    "taller_shorter": ("taller", "shorter"),
    "older_younger": ("older", "younger"),
    "faster_slower": ("faster", "slower"),
    "happier_sadder": ("happier", "sadder"),
}

SHARED_FAMILIES = set(PAIRWISE) | {"between_outside"}


def norm(x):
    return " ".join(str(x).split())


def parse_options(q):
    q = norm(q)

    q = re.sub(
        r"^Choose one of the following alternatives:\s*",
        "",
        q,
        flags=re.I,
    )

    q = re.sub(
        r"\s*,?\s*or\s*\(3\)\s*(?:it\s+is|is\s+it)"
        r"\s+impossible\s+to\s+decide\s*[?.]?\s*$",
        "",
        q,
        flags=re.I,
    )

    m = re.match(
        r"^\(1\)\s*(.*?)\s*,?\s*\(2\)\s*(.*?)\s*$",
        q,
        flags=re.I,
    )

    if not m:
        raise RuntimeError(f"Could not parse:\n{q}")

    return (
        m.group(1).strip(" ,.?"),
        m.group(2).strip(" ,.?"),
    )


def detect_family(o1, o2):
    if re.search(r"\bbetween\b", o1, flags=re.I) and re.search(
        r"\boutside\b", o2, flags=re.I
    ):
        return "between_outside"

    s = f"{o1} {o2}".lower()
    hits = []

    for fam, (a, b) in PAIRWISE.items():
        if re.search(rf"\b{re.escape(a)}\b", s) and re.search(
            rf"\b{re.escape(b)}\b", s
        ):
            hits.append(fam)

    if len(hits) != 1:
        raise RuntimeError(
            f"Could not classify:\nOPTION1: {o1}\nOPTION2: {o2}\nHITS: {hits}"
        )

    return hits[0]


def relation_in_option(option, words):
    hits = [
        word
        for word in words
        if re.search(rf"\b{re.escape(word)}\b", option, flags=re.I)
    ]

    if len(hits) != 1:
        raise RuntimeError(
            f"Expected one relation from {words}:\n{option}\nHITS={hits}"
        )

    return hits[0]


def rewrite(q):
    o1, o2 = parse_options(q)
    family = detect_family(o1, o2)

    if family == "between_outside":
        assert re.search(r"\bbetween\b", o1, flags=re.I)
        assert re.search(r"\boutside\b", o2, flags=re.I)

        template, n = re.subn(
            r"\bin\s+between\b|\bbetween\b",
            "[RELATION]",
            o1,
            count=1,
            flags=re.I,
        )

        if n != 1:
            raise RuntimeError(f"Could not rewrite:\n{o1}")

        relation1 = "between"
        relation2 = "outside the interval between"

        if re.search(r"\bbetween\b|\boutside\b", template, flags=re.I):
            raise RuntimeError(f"Relation leakage:\n{template}")

    else:
        words = PAIRWISE[family]
        relation1 = relation_in_option(o1, words)
        relation2 = relation_in_option(o2, words)

        template, n = re.subn(
            rf"\b{re.escape(relation1)}\b",
            "[RELATION]",
            o1,
            count=1,
            flags=re.I,
        )

        if n != 1:
            raise RuntimeError(f"Could not rewrite:\n{o1}")

        for word in words:
            if re.search(rf"\b{re.escape(word)}\b", template, flags=re.I):
                raise RuntimeError(f"Relation leakage:\n{template}")

    template = norm(template)

    question = (
        "What relation should replace [RELATION] in the expression below?\n"
        f"{template}"
    )

    assert template.count("[RELATION]") == 1
    assert question.count("[RELATION]") == 2

    return {
        "family": family,
        "option1": o1,
        "option2": o2,
        "relation1": relation1,
        "relation2": relation2,
        "template": template,
        "question": question,
    }


def load_qc(path):
    df = pd.read_csv(path)
    df = df[df["qc_v2"] == "keep"].copy()

    if not df["group_id"].is_unique:
        raise RuntimeError(f"group_id is not unique in {path}")

    return df.set_index("group_id", drop=False)


def balanced_choices(n, rng):
    choices = np.array(["1"] * (n // 2) + ["2"] * (n // 2))

    if n % 2:
        choices = np.append(choices, "1" if rng.integers(0, 2) == 0 else "2")

    rng.shuffle(choices)
    return choices


def main():
    train = load_qc(TRAIN_QC)
    test = load_qc(TEST_QC)

    with open(SPLIT_PATH) as f:
        split_spec = json.load(f)

    group_ids = split_spec["group_ids"]

    split_sources = {
        "gate_train": train,
        "calibration": train,
        "val": train,
        "test": test,
    }

    expected = {
        "gate_train": 1500,
        "calibration": 400,
        "val": 400,
        "test": 1000,
    }

    pairs = []

    for j, split in enumerate(["gate_train", "calibration", "val", "test"]):
        ids = group_ids[split]
        source_df = split_sources[split]

        assert len(ids) == expected[split]
        assert len(ids) == len(set(ids))

        rng = np.random.default_rng(SEED + 100 + j)
        choices = balanced_choices(len(ids), rng)

        for i, (group_id, choice) in enumerate(zip(ids, choices)):
            if group_id not in source_df.index:
                raise RuntimeError(f"{split}: missing group_id {group_id}")

            r = source_df.loc[group_id]

            a_raw = r["A1_query"] if choice == "1" else r["A2_query"]

            a = rewrite(a_raw)
            u = rewrite(r["U_query"])

            if a["family"] != u["family"]:
                raise RuntimeError(
                    f"A/U family mismatch: {group_id}: "
                    f"{a['family']} vs {u['family']}"
                )

            if a["family"] not in SHARED_FAMILIES:
                raise RuntimeError(f"Unexpected family: {a['family']}")

            a_gold = a["relation1"] if choice == "1" else a["relation2"]

            source_id = f"worldsense_{split}_{i:04d}"
            audio_path = str(Path("data/tts/worldsense") / split / f"{source_id}.wav")

            pairs.append(
                {
                    "source_id": source_id,
                    "audio_path": audio_path,
                    "dataset": "worldsense",
                    "variant": "openended_relation_completion",
                    "split": split,
                    "original_group_id": group_id,
                    "qc_v2": r["qc_v2"],
                    "audio_text": norm(r["context"]),
                    "selected_a_label": choice,
                    "a_query_raw": norm(a_raw),
                    "u_query_raw": norm(r["U_query"]),
                    "a_query": a["question"],
                    "u_query": u["question"],
                    "a_family": a["family"],
                    "u_family": u["family"],
                    "a_gold": a_gold,
                    "u_gold": None,
                    "a_option1": a["option1"],
                    "a_option2": a["option2"],
                    "u_option1": u["option1"],
                    "u_option2": u["option2"],
                }
            )

    assert len(pairs) == 3300

    trials = []

    for p in pairs:
        trials.append(
            {
                "trial_id": f"{p['source_id']}_A",
                "source_id": p["source_id"],
                "audio_path": p["audio_path"],
                "dataset": "worldsense",
                "variant": p["variant"],
                "split": p["split"],
                "label": "A",
                "condition": "answerable",
                "audio_text": p["audio_text"],
                "question": p["a_query"],
                "raw_question": p["a_query_raw"],
                "gold_answer": p["a_gold"],
                "original_gold": p["selected_a_label"],
                "relation_family": p["a_family"],
                "target_behavior": "answer",
            }
        )

        trials.append(
            {
                "trial_id": f"{p['source_id']}_U",
                "source_id": p["source_id"],
                "audio_path": p["audio_path"],
                "dataset": "worldsense",
                "variant": p["variant"],
                "split": p["split"],
                "label": "U",
                "condition": "unanswerable",
                "audio_text": p["audio_text"],
                "question": p["u_query"],
                "raw_question": p["u_query_raw"],
                "gold_answer": None,
                "original_gold": "3",
                "relation_family": p["u_family"],
                "target_behavior": "abstain",
            }
        )

    assert len(trials) == 6600
    assert len({x["trial_id"] for x in trials}) == 6600

    # Pair-level A/U checks.
    for p in pairs:
        assert p["a_family"] == p["u_family"]
        assert p["a_gold"] is not None
        assert p["u_gold"] is None

    # Question-format checks over ALL splits, including test.
    # Do not print test question contents.
    for t in trials:
        q = t["question"]

        assert "(1)" not in q
        assert "(2)" not in q
        assert "(3)" not in q
        assert "impossible to decide" not in q.lower()
        assert "choose one of" not in q.lower()
        assert q.count("[RELATION]") == 2

    PAIR_OUT.parent.mkdir(parents=True, exist_ok=True)

    with open(PAIR_OUT, "w") as f:
        for x in pairs:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    with open(TRIAL_OUT, "w") as f:
        for x in trials:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    print("pairs:", len(pairs))
    print("trials:", len(trials))

    for split in ["gate_train", "calibration", "val", "test"]:
        ps = [x for x in pairs if x["split"] == split]
        ts = [x for x in trials if x["split"] == split]

        print(
            f"{split:12s}",
            f"pairs={len(ps):4d}",
            f"trials={len(ts):4d}",
            f"A={sum(x['label']=='A' for x in ts):4d}",
            f"U={sum(x['label']=='U' for x in ts):4d}",
        )

    print("\nA1/A2")
    print("A1:", sum(p["selected_a_label"] == "1" for p in pairs))
    print("A2:", sum(p["selected_a_label"] == "2" for p in pairs))

    print("\nPASS")
    print("A/U family match: 100%")
    print("MC markers: 0")
    print("unknown families: 0")
    print("test structural QC: PASS (content not printed)")
    print("wrote:", PAIR_OUT)
    print("wrote:", TRIAL_OUT)


if __name__ == "__main__":
    main()
