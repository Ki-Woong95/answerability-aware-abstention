"""Audit-only paired adapter; uses unchanged upstream A generation/templates."""

import hashlib
import importlib
import json
import random
import sys
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "vendor/treecut_math/treecut"


def upstream_modules():
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    return importlib.import_module("gen_data"), importlib.import_module("gen_questions")


def reduce_system(matrix, rhs):
    a = [[Fraction(v) for v in row] + [Fraction(b)] for row, b in zip(matrix, rhs)]
    n = len(matrix[0])
    pivots = []
    i = 0
    for j in range(n):
        k = next((k for k in range(i, len(a)) if a[k][j]), None)
        if k is None:
            continue
        a[i], a[k] = a[k], a[i]
        v = a[i][j]
        a[i] = [x / v for x in a[i]]
        for k in range(len(a)):
            if k != i:
                v = a[k][j]
                a[k] = [x - v * y for x, y in zip(a[k], a[i])]
        pivots.append(j)
        i += 1
        if i == len(a):
            break
    if any(not any(row[:n]) and row[n] for row in a):
        raise ValueError("Inconsistent linear system")
    null = []
    for free in sorted(set(range(n)) - set(pivots)):
        vector = [Fraction(0)] * n
        vector[free] = 1
        for k, j in enumerate(pivots):
            vector[j] = -a[k][free]
        null.append(vector)
    solution = [Fraction(0)] * n
    for k, j in enumerate(pivots):
        solution[j] = a[k][n]
    return len(pivots), solution, null


def build_pair(seed, num_vars, ans_depth, composite_names, cut_depth):
    if (
        num_vars not in (2, 4)
        or not 2 <= ans_depth <= num_vars
        or not 1 <= cut_depth < ans_depth
    ):
        raise ValueError("Invalid pilot structure/cut")
    gd, gq = upstream_modules()
    saved = {}
    native_formulas = gq.gen_formulas
    native_question = gd.gen_question

    def capture_formulas(*args):
        result = native_formulas(*args)
        saved["formulas"] = result
        return result

    def capture_question(edges, target, names, values, plurals):
        result = native_question(edges, target, names, values, plurals)
        saved.update(
            edges=edges,
            target=target,
            names=names,
            values=values,
            plurals=plurals,
            question=result[1],
            sentences=result[3],
        )
        return result

    state = random.getstate()
    try:
        random.seed(seed)
        with (
            patch.object(gq, "gen_formulas", side_effect=capture_formulas),
            patch.object(gd, "gen_question", side_effect=capture_question),
        ):
            answerable = gd.generate_qa(
                "food", composite_names, num_vars, ans_depth, "random", False
            )
    finally:
        random.setstate(state)
    child = str(ans_depth - cut_depth)
    cut = ("ROOT" if child == "1" else str(int(child) - 1), child)
    assert cut in saved["edges"]
    removed = saved["sentences"][cut]
    before = answerable["problem"].rsplit(saved["question"], 1)[0].strip()
    ordered = before.split(". ")
    ordered = [s if s.endswith(".") else s + "." for s in ordered]
    assert len(ordered) == num_vars and ordered.count(removed) == 1
    remaining = [s for s in ordered if s != removed]
    matrix = []
    rhs = []
    exported = []
    for edge in saved["edges"]:
        formula = saved["formulas"][edge]
        row = [0] * num_vars
        if edge[0] == "ROOT":
            row[int(edge[1]) - 1] = formula.para[0]
        else:
            row[int(edge[0]) - 1], row[int(edge[1]) - 1] = formula.para
        matrix.append(row)
        rhs.append(formula.res)
        exported.append(
            dict(
                edge=list(edge),
                coefficients=row,
                rhs=formula.res,
                sentence=saved["sentences"][edge],
            )
        )
    rank, solution, null = reduce_system(matrix, rhs)
    expected = [saved["values"][saved["names"][str(i + 1)]] for i in range(num_vars)]
    assert (
        rank == num_vars
        and not null
        and solution == expected
        and str(solution[ans_depth - 1]) == answerable["answer"]
    )
    cut_index = saved["edges"].index(cut)
    u_matrix = [r for i, r in enumerate(matrix) if i != cut_index]
    u_rhs = [r for i, r in enumerate(rhs) if i != cut_index]
    u_rank, _, u_null = reduce_system(u_matrix, u_rhs)
    witness = next((v for v in u_null if v[ans_depth - 1]), None)
    assert u_rank == num_vars - 1 and witness is not None
    # Explicit two positive solutions, independent of the model or proof string.
    epsilon = min(Fraction(v, 2) / abs(z) for v, z in zip(expected, witness) if z)
    alternative = [Fraction(v) + epsilon * z for v, z in zip(expected, witness)]
    assert (
        all(x > 0 for x in alternative)
        and alternative[ans_depth - 1] != expected[ans_depth - 1]
    )
    for row, b in zip(u_matrix, u_rhs):
        assert sum(x * c for x, c in zip(alternative, row)) == b
    latent = dict(
        numVars=num_vars,
        ansDepth=ans_depth,
        values=expected,
        equations=[dict(coefficients=r, rhs=b) for r, b in zip(matrix, rhs)],
    )
    fingerprint = hashlib.sha256(
        json.dumps(latent, sort_keys=True).encode()
    ).hexdigest()
    pair_id = f"treecut_pilot_{seed}_{num_vars}_{ans_depth}_{int(composite_names)}_{cut_depth}"
    return dict(
        pair_id=pair_id,
        latent_group_id=fingerprint,
        seed=seed,
        numVars=num_vars,
        ansDepth=ans_depth,
        compositeName=composite_names,
        cutDepth=cut_depth,
        cut_edge=list(cut),
        deleted_condition=removed,
        variable_names={k: v for k, v in saved["names"].items() if k != "ROOT"},
        equations=exported,
        answerable=answerable,
        unanswerable=dict(
            problem=" ".join(remaining + [saved["question"]]), answer="unknown"
        ),
        proof_audit=dict(
            A_rank=rank,
            U_rank=u_rank,
            target_null_direction=[str(v) for v in witness],
            A_solution=[str(v) for v in solution],
            U_alternative_positive_solution=[str(v) for v in alternative],
        ),
    )
