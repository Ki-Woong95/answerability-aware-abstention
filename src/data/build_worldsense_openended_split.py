import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 17

TRAIN_QC = Path("external/worldsense/qc/train_completeness_qc_v2.csv")
TEST_QC = Path("external/worldsense/qc/test_completeness_qc_v2.csv")

OUT = Path("data/splits/worldsense_openended_split.json")

SHARED_FAMILIES = [
    "between_outside",
    "before_after",
    "taller_shorter",
    "older_younger",
    "faster_slower",
    "happier_sadder",
]

PAIRWISE = {
    "before_after": ("before", "after"),
    "taller_shorter": ("taller", "shorter"),
    "older_younger": ("older", "younger"),
    "faster_slower": ("faster", "slower"),
    "happier_sadder": ("happier", "sadder"),
}

# Fixed from TRAIN matched-pool proportions.
QUOTAS = {
    "gate_train": {
        "between_outside": 974,
        "before_after": 366,
        "taller_shorter": 47,
        "older_younger": 42,
        "faster_slower": 39,
        "happier_sadder": 32,
    },
    "calibration": {
        "between_outside": 260,
        "before_after": 98,
        "taller_shorter": 12,
        "older_younger": 11,
        "faster_slower": 10,
        "happier_sadder": 9,
    },
    "val": {
        "between_outside": 260,
        "before_after": 98,
        "taller_shorter": 12,
        "older_younger": 11,
        "faster_slower": 10,
        "happier_sadder": 9,
    },
    "test": {
        "between_outside": 649,
        "before_after": 244,
        "taller_shorter": 31,
        "older_younger": 28,
        "faster_slower": 26,
        "happier_sadder": 22,
    },
}


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

    return m.group(1), m.group(2)


def family(q):
    o1, o2 = parse_options(q)

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
        return "OTHER"

    return hits[0]


def prepare(path):
    df = pd.read_csv(path)
    df = df[df["qc_v2"] == "keep"].reset_index(drop=True)

    df["A1_family"] = df["A1_query"].map(family)
    df["A2_family"] = df["A2_query"].map(family)
    df["U_family"] = df["U_query"].map(family)

    df["family_match"] = (df["A1_family"] == df["A2_family"]) & (
        df["A1_family"] == df["U_family"]
    )

    df = df[df["family_match"] & df["A1_family"].isin(SHARED_FAMILIES)].copy()

    return df.reset_index(drop=True)


def sample_split(pool, quotas, rng):
    pieces = []

    for fam in SHARED_FAMILIES:
        d = pool[pool["A1_family"] == fam]

        n = quotas[fam]

        if len(d) < n:
            raise RuntimeError(f"Not enough {fam}: need {n}, have {len(d)}")

        idx = rng.choice(d.index.to_numpy(), size=n, replace=False)
        picked = d.loc[idx].copy()
        pieces.append(picked)

        pool = pool.drop(index=idx)

    selected = pd.concat(pieces, ignore_index=True)

    # Shuffle within split after stratified selection.
    selected = selected.iloc[rng.permutation(len(selected))].reset_index(drop=True)

    return selected, pool


def ids_for(df, prefix):
    return [f"worldsense_{prefix}_{i:04d}" for i in range(len(df))]


def print_counts(name, df):
    print(f"\n{name}: {len(df)}")
    print(
        df["A1_family"]
        .value_counts()
        .reindex(SHARED_FAMILIES, fill_value=0)
        .to_string()
    )


def main():
    train = prepare(TRAIN_QC)
    test = prepare(TEST_QC)

    print("Eligible family-matched shared-family pools")
    print_counts("TRAIN", train)
    print_counts("TEST", test)

    assert len(train) >= 2300
    assert len(test) >= 1000

    rng = np.random.default_rng(SEED)

    gate_train, train = sample_split(train, QUOTAS["gate_train"], rng)
    calibration, train = sample_split(train, QUOTAS["calibration"], rng)
    val, train = sample_split(train, QUOTAS["val"], rng)

    # Independent deterministic test selection.
    rng_test = np.random.default_rng(SEED)
    test_sel, _ = sample_split(test, QUOTAS["test"], rng_test)

    splits = {
        "gate_train": ids_for(gate_train, "gate_train"),
        "calibration": ids_for(calibration, "calibration"),
        "val": ids_for(val, "val"),
        "test": ids_for(test_sel, "test"),
    }

    # Save original group IDs as provenance as well.
    provenance = {
        "gate_train": gate_train["group_id"].tolist(),
        "calibration": calibration["group_id"].tolist(),
        "val": val["group_id"].tolist(),
        "test": test_sel["group_id"].tolist(),
    }

    payload = {
        "seed": SEED,
        "selection": {
            "qc_v2": "keep",
            "require_a_u_family_match": True,
            "shared_families": SHARED_FAMILIES,
            "sampling": "stratified_fixed_train-derived-proportions",
        },
        "quotas": QUOTAS,
        "splits": splits,
        "group_ids": provenance,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2)

    print("\nSELECTED SPLITS")
    print_counts("gate_train", gate_train)
    print_counts("calibration", calibration)
    print_counts("val", val)
    print_counts("test", test_sel)

    assert len(gate_train) == 1500
    assert len(calibration) == 400
    assert len(val) == 400
    assert len(test_sel) == 1000

    # No original group may overlap between train-side splits.
    gt = set(gate_train["group_id"])
    cal = set(calibration["group_id"])
    va = set(val["group_id"])

    assert not (gt & cal)
    assert not (gt & va)
    assert not (cal & va)

    print("\nPASS")
    print("train-side selected:", 2300)
    print("test selected:", 1000)
    print("A/U family-matched: 100%")
    print("shared family inventory:", len(SHARED_FAMILIES))
    print("train split overlaps: 0")
    print("saved:", OUT)


if __name__ == "__main__":
    main()
