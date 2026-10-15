"""Report frozen inputs and duration cues; never score model behavior."""

import collections, json, statistics
from pathlib import Path
from src.data.frozen_treecut import ROOT, verify_dataset, load_trials


def duration_auc(a, u):
    return sum((y < x) + 0.5 * (y == x) for x in a for y in u) / (len(a) * len(u))


def main():
    record = verify_dataset()
    trials = load_trials()
    groups = collections.defaultdict(list)
    for row in trials:
        if row["split"] == "gate_train":
            groups[(row["numVars"], row["ansDepth"], row["compositeName"])].append(row)
    diag = []
    for cfg, rows in sorted(groups.items()):
        a = [r["audio_duration"] for r in rows if r["condition"] == "answerable"]
        u = [r["audio_duration"] for r in rows if r["condition"] == "unanswerable"]
        diag.append(
            dict(
                numVars=cfg[0],
                ansDepth=cfg[1],
                compositeName=cfg[2],
                pairs=len(a),
                A_mean=statistics.mean(a),
                U_mean=statistics.mean(u),
                duration_only_auc=duration_auc(a, u),
                max_duration=max(a + u),
            )
        )
    folder = ROOT / "data/treecut"
    (folder / "duration_diagnostics.json").write_text(json.dumps(diag, indent=2) + "\n")
    lines = [
        "# Frozen TreeCut data report",
        "",
        f"Candidates: {record['candidate_pairs']} pairs. Retained: {record['retained_pairs']}. Excluded for complete-pair duration QC: {record['excluded_duration_pairs']}.",
        "",
        f"Split SHA-256: `{record['protected_sha256']['data/splits/treecut_final_split.json']}`",
        "",
        "| Split | Retained pair IDs | Active A | Active U |",
        "| --- | ---: | ---: | ---: |",
    ]
    for split, n in record["counts"].items():
        lines.append(f"| {split} | {n} | {n} | {0 if split=='calibration' else n} |")
    lines += [
        "",
        "One full A and one U created by deleting a single necessary condition share a latent family. All retained recordings are strictly below 30 seconds. Exact rational target verification, complete text coverage, waveform hashes and latent grouping passed. Model labels and thresholds were not used to select data.",
        "",
        "| Variables | Depth | Composite names | Gate-training pairs | A mean s | U mean s | Duration-only AUROC |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for d in diag:
        lines.append(
            f"| {d['numVars']} | {d['ansDepth']} | {d['compositeName']} | {d['pairs']} | {d['A_mean']:.2f} | {d['U_mean']:.2f} | {d['duration_only_auc']:.3f} |"
        )
    lines += [
        "",
        "Duration-only AUROC uses the fixed score −duration on gate-training input labels only, with no fitted model or test-score decisions. It is a descriptive data diagnostic, not a held-out gate result. Deletion creates length and condition-count cues; behavioral intervention effects and reasoning-awareness interpretations must be separated. Formal calibration claims remain conditional on the stated assumptions.",
        "",
        "Manual semantic listening of all recordings has not been performed. Previous representative paired samples were provided for review; full grapheme/phoneme coverage is not a pronunciation guarantee. No TreeCut test generation or behavioral evaluation is included in the current capture/gate chain.",
    ]
    (folder / "DATA_REPORT.md").write_text("\n".join(lines) + "\n")
    print("Written data/treecut/DATA_REPORT.md and duration_diagnostics.json")


if __name__ == "__main__":
    main()
