"""Inspect TreeCut public samples; no TTS or experimental changes."""

import collections, hashlib, json, re, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/audits/treecut_math"


def parts(problem):
    body, q = problem.rsplit("Question:", 1)
    return [s.strip() for s in body.strip().split(". ") if s.strip()], q.strip()


def normalize(s):
    return " ".join(s.lower().split()).rstrip(".")


def key(sentences, question):
    return hashlib.sha256(
        json.dumps(
            [normalize(question), sorted(normalize(s) for s in sentences)]
        ).encode()
    ).hexdigest()


def main():
    rows = []
    files = []
    for p in sorted((OUT / "upstream").glob("*.jsonl")):
        parsed = re.search(
            r"numVars-(\d+)_ansDepth-(\d+).*hallu-(True|False)(?:_cutDepth-(\d+))?",
            p.name,
        )
        n, depth, label, cut = parsed.groups()
        contents = [json.loads(l) for l in p.open() if l.strip()]
        files.append(
            dict(
                file=p.name,
                rows=len(contents),
                numVars=int(n),
                ansDepth=int(depth),
                U=label == "True",
                sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            )
        )
        for i, r in enumerate(contents):
            s, q = parts(r["problem"])
            words = len(r["problem"].split())
            rows.append(
                dict(
                    r,
                    file=p.name,
                    index=i,
                    numVars=int(n),
                    ansDepth=int(depth),
                    U=label == "True",
                    cutDepth=int(cut) if cut else None,
                    words=words,
                    estimated_seconds_170wpm=2 + words * 60 / 170,
                    nonascii=any(ord(c) > 127 for c in r["problem"]),
                    unusual_characters=sorted(
                        set(c for c in r["problem"] if ord(c) > 127)
                    ),
                    condition_count=len(s),
                    key=key(s, q),
                )
            )
    # Exact shared evidence modulo sentence order, with one condition removed.
    variants = collections.defaultdict(set)
    for r in rows:
        if not r["U"]:
            s, q = parts(r["problem"])
            for i in range(len(s)):
                variants[(r["numVars"], r["ansDepth"], key(s[:i] + s[i + 1 :], q))].add(
                    (r["file"], r["index"])
                )
    matched = []
    for r in rows:
        if r["U"]:
            candidates = variants.get((r["numVars"], r["ansDepth"], r["key"]), set())
            if candidates:
                matched.append(
                    dict(
                        U_file=r["file"],
                        U_index=r["index"],
                        A_candidates=sorted(candidates),
                    )
                )
    groups = {}
    for n in sorted({r["numVars"] for r in rows}):
        selected = [r for r in rows if r["numVars"] == n]
        w = [r["words"] for r in selected]
        groups[str(n)] = dict(
            rows=len(selected),
            A=sum(not r["U"] for r in selected),
            U=sum(r["U"] for r in selected),
            mean_words=statistics.mean(w),
            min_words=min(w),
            max_words=max(w),
            estimated_under30=sum(r["estimated_seconds_170wpm"] < 30 for r in selected),
        )
    summary = dict(
        revision=json.loads((OUT / "upstream_files.json").read_text())["revision"],
        files=len(files),
        rows=len(rows),
        A=sum(not r["U"] for r in rows),
        U=sum(r["U"] for r in rows),
        nonascii=sum(r["nonascii"] for r in rows),
        unique_problem_texts=len({r["problem"] for r in rows}),
        label_mismatches=sum((r["answer"] == "unknown") != r["U"] for r in rows),
        numeric_A_answers=sum(
            bool(re.fullmatch(r"-?\d+(?:\.\d+)?", str(r["answer"])))
            for r in rows
            if not r["U"]
        ),
        exact_A_U_matches_after_one_condition_deletion=len(matched),
        groups_by_numVars=groups,
        label_interpretation="hallu-True in filename means unanswerable construction, not an observed model hallucination",
        limitations=[
            "Word-count durations are estimates; no TTS yet.",
            "Exact matching ignores sentence order but does not prove no semantic pairs exist.",
            "Released fields are problem, answer, proof; no explicit tree or pair IDs.",
            "Evidence/answer correctness is not yet independently verified from linear equations.",
        ],
    )
    (OUT / "records_audit.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )
    (OUT / "file_inventory.json").write_text(json.dumps(files, indent=2) + "\n")
    (OUT / "paired_matches.json").write_text(json.dumps(matched, indent=2) + "\n")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
