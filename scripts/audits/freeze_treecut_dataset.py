"""Freeze retained paired TreeCut candidates after complete duration QC."""

import argparse, collections, hashlib, json, random, wave
from pathlib import Path
from fractions import Fraction
from scripts.audits.treecut_pairs import reduce_system
from src.data.frozen_treecut import ROOT, SPLIT_PATH, TRIALS_PATH, RECORD_PATH, sha256
from src.data.tts_text import normalize_tts_text


def allocate(total):
    weights = dict(
        gate_train=1423 / 2556, calibration=360 / 2556, val=386 / 2556, test=387 / 2556
    )
    counts = {k: int(total * w) for k, w in weights.items()}
    for k in sorted(
        weights, key=lambda k: total * weights[k] - counts[k], reverse=True
    )[: total - sum(counts.values())]:
        counts[k] += 1
    return counts


def validate_pair(pair):
    n = pair["numVars"]
    target = pair["ansDepth"] - 1
    eq = pair["equations"]
    matrix = [e["coefficients"] for e in eq]
    rhs = [e["rhs"] for e in eq]
    rank, solution, null = reduce_system(matrix, rhs)
    assert (
        rank == n
        and not null
        and str(solution[target]) == str(pair["answerable"]["answer"])
    )
    cut = [e for e in eq if e["edge"] == pair["cut_edge"]]
    assert len(cut) == 1 and cut[0]["sentence"] == pair["deleted_condition"]
    surviving = [e for e in eq if e["edge"] != pair["cut_edge"]]
    u_rank, _, directions = reduce_system(
        [e["coefficients"] for e in surviving], [e["rhs"] for e in surviving]
    )
    assert u_rank == n - 1 and any(v[target] for v in directions)
    a, q = pair["answerable"]["problem"].rsplit("Question:", 1)
    u, uq = pair["unanswerable"]["problem"].rsplit("Question:", 1)
    assert q == uq
    a = [s.rstrip(".") for s in a.strip().split(". ")]
    u = [s.rstrip(".") for s in u.strip().split(". ")]
    assert collections.Counter(a) == collections.Counter(
        e["sentence"].rstrip(".") for e in eq
    )
    assert collections.Counter(u) == collections.Counter(
        e["sentence"].rstrip(".") for e in surviving
    )
    assert (
        [s for s in a if s != pair["deleted_condition"].rstrip(".")] == u
        and len(a) == n
        and len(u) == n - 1
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, required=True)
    args = ap.parse_args()
    pool = args.candidates
    if any(p.exists() for p in [SPLIT_PATH, TRIALS_PATH, RECORD_PATH]):
        raise FileExistsError("Frozen TreeCut files already exist; no overwrite")
    pairs = [json.loads(l) for l in (pool / "passed_pairs.jsonl").open() if l.strip()]
    allpairs = [json.loads(l) for l in (pool / "pairs.jsonl").open() if l.strip()]
    audio_rows = list(map(json.loads, (pool / "manifest.jsonl").open()))
    audios = {(r["pair_id"], r["condition"]): r for r in audio_rows}
    assert len(audios) == len(audio_rows), "Duplicate audio manifest identities"
    assert pairs and len(audios) == 2 * len(allpairs)
    for p in allpairs:
        validate_pair(p)
    assert len({p["pair_id"] for p in allpairs}) == len(allpairs) and len(
        {p["latent_group_id"] for p in allpairs}
    ) == len(allpairs)
    # Check exact text overlap against all UMWP input pairs, independent of outcomes.
    existing = set()
    for r in map(json.loads, (ROOT / "data/processed/umwp_pairs.jsonl").open()):
        for c in ["answerable", "unanswerable"]:
            existing.add(" ".join(r[c]["question"].casefold().split()))
    assert not any(
        " ".join(p[c]["problem"].casefold().split()) in existing
        for p in allpairs
        for c in ["answerable", "unanswerable"]
    )
    strata = collections.defaultdict(list)
    for p in pairs:
        strata[(p["numVars"], p["ansDepth"], p["compositeName"])].append(p)
    splits = {k: [] for k in ["gate_train", "calibration", "val", "test"]}
    counts = {}
    lookup = {p["pair_id"]: p for p in pairs}
    rng = random.Random(17)
    for cfg, group in sorted(strata.items()):
        group = sorted(group, key=lambda p: p["pair_id"])
        rng.shuffle(group)
        counts[str(cfg)] = allocate(len(group))
        offset = 0
        for split, n in counts[str(cfg)].items():
            splits[split].extend(p["pair_id"] for p in group[offset : offset + n])
            offset += n
    assert len({x for ids in splits.values() for x in ids}) == len(pairs)
    trials = []
    for split, ids in splits.items():
        for pid in ids:
            pair = lookup[pid]
            for condition, suffix in [("answerable", "A"), ("unanswerable", "U")]:
                audio = audios[pid, condition]
                path = ROOT / audio["audio_path"]
                assert sha256(path) == audio["audio_sha256"]
                with wave.open(str(path)) as wav:
                    assert (
                        wav.getnchannels() == 1
                        and wav.getframerate() == audio["sample_rate"]
                        and wav.getnframes() == audio["frames"]
                        and wav.getnframes() / wav.getframerate() < 30
                    )
                raw = pair[condition]["problem"]
                assert audio["raw_text"] == raw and audio[
                    "tts_text"
                ] == normalize_tts_text(raw)
                assert "".join(audio["tts_text"].split()) == "".join(
                    "".join(c["graphemes"].split()) for c in audio["tts_chunks"]
                )
                assert not any(
                    "❓" in c["phonemes"] for c in audio["tts_chunks"]
                ), "Unknown TTS phoneme marker"
                trials.append(
                    dict(
                        dataset="treecut",
                        trial_id=pid + "_" + suffix,
                        pair_id=pid,
                        latent_group_id=pair["latent_group_id"],
                        condition=condition,
                        split=split,
                        source="treecut",
                        category=None,
                        voice="af_heart",
                        audio_path=audio["audio_path"],
                        raw_text=raw,
                        tts_text=audio["tts_text"],
                        gold_answer=(
                            [pair["answerable"]["answer"]]
                            if condition == "answerable"
                            else []
                        ),
                        numVars=pair["numVars"],
                        ansDepth=pair["ansDepth"],
                        cutDepth=pair["cutDepth"],
                        compositeName=pair["compositeName"],
                        audio_sha256=audio["audio_sha256"],
                        audio_duration=audio["duration"],
                    )
                )
    split_record = dict(
        dataset="treecut",
        status="frozen",
        seed=17,
        grouping="unique latent family; one A/one U per family",
        splits=splits,
        counts={k: len(v) for k, v in splits.items()},
        stratum_counts=counts,
        retained_pairs=len(pairs),
        candidate_pairs=len(allpairs),
        excluded_duration_pairs=len(allpairs) - len(pairs),
        raw_recordings=len(trials),
        active_trials=2 * (len(pairs) - len(splits["calibration"]))
        + len(splits["calibration"]),
        umwp_exact_text_overlap=0,
        scope="fresh matched generation using pinned native TreeCut A templates",
    )
    for p in [SPLIT_PATH, TRIALS_PATH, RECORD_PATH]:
        p.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_PATH.write_text(json.dumps(split_record, indent=2) + "\n")
    TRIALS_PATH.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in trials)
    )
    protected = [
        SPLIT_PATH,
        TRIALS_PATH,
        pool / "pairs.jsonl",
        pool / "manifest.jsonl",
        pool / "passed_pairs.jsonl",
        pool / "failed_pairs.jsonl",
        pool / "provenance.json",
    ]
    record = dict(
        **{k: v for k, v in split_record.items() if k != "splits"},
        protected_sha256={
            str(p.resolve().relative_to(ROOT)): sha256(p) for p in protected
        },
        candidate_pool=str(pool),
        manual_listening="representative review pending; prior pilot delivered for review",
        full_audio="No condition or waveform truncated; both complete WAVs strictly <30s; full grapheme coverage verified",
    )
    RECORD_PATH.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
