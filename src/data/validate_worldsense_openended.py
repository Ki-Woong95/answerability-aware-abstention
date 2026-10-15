import re
import pandas as pd
from pathlib import Path
from collections import Counter

PATH = Path("data/processed/worldsense_openended_train_draft.csv")

PAIRWISE = {
    "before_after": ("before", "after"),
    "bigger_smaller": ("bigger", "smaller"),
    "taller_shorter": ("taller", "shorter"),
    "heavier_lighter": ("heavier", "lighter"),
    "warmer_cooler": ("warmer", "cooler"),
    "older_younger": ("older", "younger"),
    "faster_slower": ("faster", "slower"),
    "happier_sadder": ("happier", "sadder"),
    "higher_lower": ("higher", "lower"),
    "louder_quieter": ("louder", "quieter"),
}

KNOWN = set(PAIRWISE) | {"between_outside"}


def norm(x):
    return " ".join(str(x).split()).strip(" ,.?").lower()


def fail(msg):
    raise AssertionError(msg)


df = pd.read_csv(PATH)

print("=" * 80)
print("1. BASIC COUNTS")
print("=" * 80)

print("pairs:", len(df))
print(df["split"].value_counts().to_string())

assert len(df) == 2300
assert (df["split"] == "gate_train").sum() == 1500
assert (df["split"] == "calibration").sum() == 400
assert (df["split"] == "val").sum() == 400

print("PASS basic counts")


print("\n" + "=" * 80)
print("2. A/U + GOLD STRUCTURE")
print("=" * 80)

assert df["A_question_open"].notna().all()
assert df["U_question_open"].notna().all()
assert df["A_gold_semantic"].notna().all()
assert df["U_gold_semantic"].isna().all()

assert (df["selected_a"].astype(int) == 1).sum() == 1150
assert (df["selected_a"].astype(int) == 2).sum() == 1150

print("A questions:", df["A_question_open"].notna().sum())
print("U questions:", df["U_question_open"].notna().sum())
print("A1 selected:", (df["selected_a"].astype(int) == 1).sum())
print("A2 selected:", (df["selected_a"].astype(int) == 2).sum())
print("PASS A/U structure")


print("\n" + "=" * 80)
print("3. FAMILY COVERAGE")
print("=" * 80)

families = set(df["A_family"]) | set(df["U_family"])
unknown = families - KNOWN

print("families:", sorted(families))
print("unknown:", sorted(unknown))

assert not unknown

counts = Counter(list(df["A_family"]) + list(df["U_family"]))

for fam, n in counts.most_common():
    print(f"{fam:20s} {n:4d}")

print("PASS family coverage")


print("\n" + "=" * 80)
print("4. MULTIPLE-CHOICE / ANSWER LEAKAGE")
print("=" * 80)

bad = []

for i, r in df.iterrows():
    for side in ["A", "U"]:
        q = str(r[f"{side}_question_open"])
        template = str(r[f"{side}_template"])

        checks = {
            "(1)": "(1)" in q,
            "(2)": "(2)" in q,
            "(3)": "(3)" in q,
            "impossible": "impossible to decide" in q.lower(),
            "choose": "choose one of" in q.lower(),
        }

        hits = [k for k, v in checks.items() if v]

        if hits:
            bad.append((i, side, hits, q))

        if template.count("[RELATION]") != 1:
            bad.append(
                (
                    i,
                    side,
                    [f"template_relation_count={template.count('[RELATION]')}"],
                    template,
                )
            )

        # One in instruction + one in expression.
        if q.count("[RELATION]") != 2:
            bad.append(
                (i, side, [f"question_relation_count={q.count('[RELATION]')}"], q)
            )

if bad:
    for x in bad[:20]:
        print(x)
    fail(f"Leakage/slot failures: {len(bad)}")

print("checked questions:", len(df) * 2)
print("failures: 0")
print("PASS no MC leakage / slot structure")


print("\n" + "=" * 80)
print("5. PAIRWISE A GOLD RECONSTRUCTION")
print("=" * 80)

recon_fail = []
checked = 0

for i, r in df.iterrows():
    fam = r["A_family"]

    if fam not in PAIRWISE:
        continue

    selected = int(r["selected_a"])
    target = r["A_option1"] if selected == 1 else r["A_option2"]

    reconstructed = str(r["A_template"]).replace(
        "[RELATION]", str(r["A_gold_semantic"])
    )

    checked += 1

    if norm(reconstructed) != norm(target):
        recon_fail.append(
            {
                "row": i,
                "family": fam,
                "selected": selected,
                "target": target,
                "reconstructed": reconstructed,
                "gold": r["A_gold_semantic"],
            }
        )

print("checked:", checked)
print("failures:", len(recon_fail))

if recon_fail:
    for x in recon_fail[:20]:
        print("\n", x)
    fail("Pairwise reconstruction failed")

print("PASS pairwise reconstruction")


print("\n" + "=" * 80)
print("6. BETWEEN/OUTSIDE A GOLD VALIDATION")
print("=" * 80)

bo = df[df["A_family"] == "between_outside"]

bo_fail = []

for i, r in bo.iterrows():
    selected = int(r["selected_a"])
    gold = str(r["A_gold_semantic"]).lower()

    if selected == 1:
        if gold != "between":
            bo_fail.append((i, selected, gold, r["A_original"]))

    elif selected == 2:
        if gold != "outside the interval between":
            bo_fail.append((i, selected, gold, r["A_original"]))

    else:
        bo_fail.append((i, selected, gold, r["A_original"]))

print("checked:", len(bo))
print("failures:", len(bo_fail))

if bo_fail:
    for x in bo_fail[:20]:
        print(x)
    fail("between/outside gold validation failed")

print("PASS between/outside gold mapping")


print("\n" + "=" * 80)
print("7. GOLD DISTRIBUTION")
print("=" * 80)

print(df["A_gold_semantic"].value_counts().to_string())


print("\n" + "=" * 80)
print("8. EXAMPLE RENDERING")
print("=" * 80)

sample = df.sample(10, random_state=17)

for _, r in sample.iterrows():
    print("\n---")
    print("split:", r["split"])
    print("A family:", r["A_family"])
    print("A:", r["A_question_open"])
    print("A gold:", r["A_gold_semantic"])
    print("U family:", r["U_family"])
    print("U:", r["U_question_open"])
    print("U gold:", r["U_gold_semantic"])


print("\n" + "=" * 80)
print("ALL STRUCTURAL QC PASSED")
print("=" * 80)
print("2300 pairs / 4600 questions")
print("No multiple-choice markers")
print("No explicit impossible-to-decide option")
print("All relation slots valid")
print("All pairwise A golds reconstruct the selected original option")
print("All between/outside A gold mappings valid")
