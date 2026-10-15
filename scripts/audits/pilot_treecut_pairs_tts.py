"""Configurable audit-only matched pairs and full WAVs; no reasoning inference."""

import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from scripts.audits.treecut_pairs import build_pair, ROOT
from src.data.tts_text import normalize_tts_text

OUT = ROOT / "data/audits/treecut_math/paired_tts_pilot_24"
SNAPSHOT = Path(
    os.environ.get(
        "KOKORO_MODEL_PATH",
        str(
            Path.home()
            / ".cache/huggingface/hub/models--hexgrad--Kokoro-82M/snapshots/f3ff3571791e39611d31c381e3a41a3af07b4987"
        ),
    )
)


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=24)
    parser.add_argument("--seed-base", type=int, default=170000)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.pairs < 6:
        raise ValueError("At least six pairs are required to cover all pilot strata")
    OUT = args.output.resolve()
    if not OUT.is_relative_to(ROOT):
        raise ValueError("Pilot output must be inside this project")
    if OUT.exists():
        raise FileExistsError("Preserve existing pilot output")
    OUT.mkdir(parents=True)
    pairs = []
    seen_families = set()
    seen_texts = set()
    attempted = 0
    # Keep the expanded pilot independent of the prior 24-pair feasibility sample.
    for size in [24, 100]:
        prior = ROOT / f"data/audits/treecut_math/paired_tts_pilot_{size}/pairs.jsonl"
        if args.pairs != size and prior.exists():
            for row in map(json.loads, prior.open()):
                seen_families.add(row["latent_group_id"])
                seen_texts.update(
                    row[c]["problem"] for c in ["answerable", "unanswerable"]
                )
    for i, (n, d) in enumerate([(2, 2), (4, 2), (4, 4)]):
        for names in [False, True]:
            stratum = i * 2 + int(names)
            count = (
                args.pairs // 6 + int(stratum >= 6 - args.pairs % 6)
                if args.pairs % 6
                else args.pairs // 6
            )
            accepted = 0
            candidate = 0
            while accepted < count:
                if candidate > 10000:
                    raise RuntimeError("Excessive duplicate sampling")
                seed = args.seed_base + i * 10000 + int(names) * 1000 + candidate
                pair = build_pair(seed, n, d, names, 1 + accepted % (d - 1))
                candidate += 1
                attempted += 1
                if pair["latent_group_id"] in seen_families or any(
                    pair[c]["problem"] in seen_texts
                    for c in ["answerable", "unanswerable"]
                ):
                    continue
                seen_families.add(pair["latent_group_id"])
                seen_texts.update(
                    pair[c]["problem"] for c in ["answerable", "unanswerable"]
                )
                pairs.append(pair)
                accepted += 1
    assert len(pairs) == args.pairs
    (OUT / "pairs.jsonl").write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs)
    )
    provenance = dict(
        scope=f"{args.pairs} matched pairs; TTS/input audit only; not an experimental split",
        seed_base=args.seed_base,
        device=args.device,
        attempted_candidates=attempted,
        excluded_prior_pilot=args.pairs != 24,
        upstream_commit="b48f3de5586d00fffaf3ae2392c6d16497b664ea",
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT / "data/audits/treecut_math/generator_source/treecut").glob(
                "*.py"
            )
        },
        snapshot=SNAPSHOT.name,
        voice="af_heart",
        speed=1.0,
        sample_rate=24000,
        adaptation="Capture native A generation once; remove precisely one necessary edge sentence without changing surviving text/order.",
        label_validation="Exact rational rank, A solution and U positive alternative solution witnesses.",
        text_fidelity="No speech listening or semantic pronunciation validation yet.",
    )
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    from kokoro import KModel, KPipeline

    torch.set_num_threads(4)
    model = (
        KModel(
            repo_id="hexgrad/Kokoro-82M",
            config=str(SNAPSHOT / "config.json"),
            model=str(SNAPSHOT / "kokoro-v1_0.pth"),
        )
        .to(args.device)
        .eval()
    )
    pipeline = KPipeline(
        lang_code="a", repo_id="hexgrad/Kokoro-82M", model=model, device=args.device
    )
    rows = []
    for i, pair in enumerate(pairs, 1):
        for condition, suffix in [("answerable", "A"), ("unanswerable", "U")]:
            text = normalize_tts_text(pair[condition]["problem"])
            parts = list(
                pipeline(text, voice=str(SNAPSHOT / "voices/af_heart.pt"), speed=1.0)
            )
            assert parts and all(p[2] is not None for p in parts)
            assert "".join(text.split()) == "".join(
                "".join(p[0].split()) for p in parts
            )
            audio = np.concatenate([p[2].detach().cpu().numpy() for p in parts])
            assert len(audio) and np.isfinite(audio).all()
            path = OUT / (pair["pair_id"] + "_" + suffix + ".wav")
            sf.write(path, audio, 24000)
            info = sf.info(path)
            row = dict(
                pair_id=pair["pair_id"],
                condition=condition,
                numVars=pair["numVars"],
                ansDepth=pair["ansDepth"],
                compositeName=pair["compositeName"],
                cutDepth=pair["cutDepth"],
                raw_text=pair[condition]["problem"],
                tts_text=text,
                duration=info.duration,
                frames=info.frames,
                sample_rate=info.samplerate,
                audio_path=str(path.relative_to(ROOT)),
                audio_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                tts_chunks=[dict(graphemes=p[0], phonemes=p[1]) for p in parts],
            )
            with (OUT / "manifest.jsonl").open("a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            print(
                f'[{len(rows)}/{2*args.pairs}] n={pair["numVars"]} depth={pair["ansDepth"]} composite={pair["compositeName"]} {suffix} {info.duration:.2f}s',
                flush=True,
            )
    summary = dict(
        pairs=args.pairs,
        recordings=len(rows),
        groups=[],
        scope=provenance["scope"],
        manual_listening="pending",
    )
    for n, d in [(2, 2), (4, 2), (4, 4)]:
        for names in [False, True]:
            selected = [
                p
                for p in pairs
                if (p["numVars"], p["ansDepth"], p["compositeName"]) == (n, d, names)
            ]
            group_rows = [
                r
                for r in rows
                if (r["numVars"], r["ansDepth"], r["compositeName"]) == (n, d, names)
            ]
            dur = {
                r["pair_id"]: {
                    v["condition"]: v["duration"]
                    for v in group_rows
                    if v["pair_id"] == r["pair_id"]
                }
                for r in group_rows
            }
            summary["groups"].append(
                dict(
                    numVars=n,
                    ansDepth=d,
                    compositeName=names,
                    pairs=len(selected),
                    both_under30=sum(max(v.values()) < 30 for v in dur.values()),
                    both_under29=sum(max(v.values()) < 29 for v in dur.values()),
                    A_mean=float(
                        np.mean(
                            [
                                r["duration"]
                                for r in group_rows
                                if r["condition"] == "answerable"
                            ]
                        )
                    ),
                    U_mean=float(
                        np.mean(
                            [
                                r["duration"]
                                for r in group_rows
                                if r["condition"] == "unanswerable"
                            ]
                        )
                    ),
                    max_duration=max(r["duration"] for r in group_rows),
                )
            )
    audio_map = {(r["pair_id"], r["condition"]): r for r in rows}
    passed = []
    failed = []
    for pair in pairs:
        a, u = [audio_map[(pair["pair_id"], c)] for c in ["answerable", "unanswerable"]]
        record = dict(
            pair,
            answerable_duration=a["duration"],
            unanswerable_duration=u["duration"],
            answerable_audio_path=a["audio_path"],
            unanswerable_audio_path=u["audio_path"],
            both_under30=max(a["duration"], u["duration"]) < 30,
            both_under29=max(a["duration"], u["duration"]) < 29,
        )
        (passed if record["both_under30"] else failed).append(record)
    for name, records in [
        ("passed_pairs.jsonl", passed),
        ("failed_pairs.jsonl", failed),
    ]:
        (OUT / name).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        )
    summary.update(
        passed_pairs_under30=len(passed),
        failed_pairs_under30=len(failed),
        passed_pairs_under29=sum(r["both_under29"] for r in passed),
        max_duration=max(r["duration"] for r in rows),
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
