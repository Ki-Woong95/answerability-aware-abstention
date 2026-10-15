from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.gate.datasets import get_dataset_config

MODELS = ("af_next", "audio-reasoner", "moss")
MODEL_LABELS = {
    "af_next": "AF-Next",
    "audio-reasoner": "Audio-Reasoner",
    "moss": "MOSS",
}
PREFERRED_SOURCES = ("GSM8K", "SVAMP", "MultiArith", "ASDiv")


def ms(values) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan, np.nan
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else np.nan


def _model_x():
    return np.arange(len(MODELS))


def _save(fig, figure_root: Path, stem: str):
    fig.tight_layout()
    fig.savefig(figure_root / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(figure_root / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_before_after(
    seed_metrics, vanilla_metric, system_metric, ylabel, title, figure_root, stem
):
    x, width = _model_x(), 0.36
    v_means, v_sds, s_means, s_sds = [], [], [], []
    for model in MODELS:
        d = seed_metrics[seed_metrics.model == model]
        vm, vs = ms(d[vanilla_metric])
        sm, ss = ms(d[system_metric])
        v_means.append(vm * 100)
        v_sds.append((0 if np.isnan(vs) else vs) * 100)
        s_means.append(sm * 100)
        s_sds.append((0 if np.isnan(ss) else ss) * 100)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.bar(x - width / 2, v_means, width, yerr=v_sds, capsize=4, label="Vanilla")
    ax.bar(x + width / 2, s_means, width, yerr=s_sds, capsize=4, label="Self-braking")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False)
    _save(fig, figure_root, stem)


def plot_gate(seed_metrics, figure_root, split_label):
    x, width = _model_x(), 0.36
    a_means, a_sds, u_means, u_sds = [], [], [], []
    for model in MODELS:
        d = seed_metrics[seed_metrics.model == model]
        am, a_sd = ms(d.a_false_rate)
        um, u_sd = ms(d.u_brake_rate)
        a_means.append(am * 100)
        a_sds.append(a_sd * 100)
        u_means.append(um * 100)
        u_sds.append(u_sd * 100)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.bar(
        x - width / 2,
        a_means,
        width,
        yerr=a_sds,
        capsize=4,
        label="Answerable false brake",
    )
    ax.bar(
        x + width / 2, u_means, width, yerr=u_sds, capsize=4, label="Unanswerable brake"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS])
    ax.set_ylabel("Rate (%)")
    ax.set_title(f"{split_label} gate behavior at calibrated 5% operating point")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False)
    _save(fig, figure_root, "fig1_gate_detection")


def plot_triggered_u(seed_metrics, figure_root):
    x, width = _model_x(), 0.24
    specs = (
        ("u_triggered_post_abstain", "Correct abstention"),
        ("u_triggered_post_hallucinated", "Hallucinated answer"),
        ("u_triggered_post_nonanswer", "Non-answer"),
    )
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for i, (metric, label) in enumerate(specs):
        means, sds = [], []
        for model in MODELS:
            m, s = ms(seed_metrics[seed_metrics.model == model][metric])
            means.append(m * 100)
            sds.append((0 if np.isnan(s) else s) * 100)
        ax.bar(x + (i - 1) * width, means, width, yerr=sds, capsize=3, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS])
    ax.set_ylabel("Triggered U trials (%)")
    ax.set_title("Post-intervention outcome on triggered unanswerable trials")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False)
    _save(fig, figure_root, "fig5_triggered_u_outcomes")


def plot_triggered_a(seed_metrics, figure_root):
    x, width = _model_x(), 0.21
    specs = (
        ("a_correct_to_correct", "Correct retained"),
        ("a_correct_to_abstain", "Became abstention"),
        ("a_correct_to_incorrect", "Became incorrect"),
        ("a_correct_to_nonanswer", "Became non-answer"),
    )
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    offsets = np.arange(len(specs)) - (len(specs) - 1) / 2
    for offset, (metric, label) in zip(offsets, specs):
        means, sds = [], []
        for model in MODELS:
            m, s = ms(seed_metrics[seed_metrics.model == model][metric])
            means.append(m * 100)
            sds.append((0 if np.isnan(s) else s) * 100)
        ax.bar(x + offset * width, means, width, yerr=sds, capsize=3, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS])
    ax.set_ylabel("Previously-correct, false-triggered A trials (%)")
    ax.set_title("Behavior after false braking on previously correct A trials")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=9)
    _save(fig, figure_root, "fig6_triggered_a_correct_outcomes")


def plot_brake_fraction(seed_metrics, figure_root):
    x, width = _model_x(), 0.36
    a_means, a_sds, u_means, u_sds = [], [], [], []
    for model in MODELS:
        d = seed_metrics[seed_metrics.model == model]
        am, a_sd = ms(d.a_brake_fraction_mean)
        um, u_sd = ms(d.u_brake_fraction_mean)
        a_means.append(am * 100)
        a_sds.append(a_sd * 100)
        u_means.append(um * 100)
        u_sds.append(u_sd * 100)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.bar(
        x - width / 2,
        a_means,
        width,
        yerr=a_sds,
        capsize=4,
        label="Answerable false brake",
    )
    ax.bar(
        x + width / 2, u_means, width, yerr=u_sds, capsize=4, label="Unanswerable brake"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS])
    ax.set_ylabel("Brake position (% of vanilla trajectory)")
    ax.set_title("When does sentence-boundary self-braking trigger?")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False)
    _save(fig, figure_root, "fig7_brake_timing")


def plot_source_heatmap(source_metrics, metric, title, figure_root, stem):
    if source_metrics.empty or metric not in source_metrics.columns:
        print(f"WARNING: skipping {stem}: source metrics unavailable")
        return
    available = [str(x) for x in source_metrics.source.dropna().unique()]
    present = [s for s in PREFERRED_SOURCES if s in available]
    present += sorted(s for s in available if s not in present)
    if not present:
        print(f"WARNING: skipping {stem}: no source metadata")
        return
    summary = (
        source_metrics.groupby(["model", "source"])[metric]
        .mean()
        .unstack("source")
        .reindex(index=MODELS, columns=present)
    )
    values = summary.to_numpy() * 100
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    image = ax.imshow(values, aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(present)))
    ax.set_xticklabels(present, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(MODELS)))
    ax.set_yticklabels([MODEL_LABELS[m] for m in MODELS])
    ax.set_title(title)
    for i in range(len(MODELS)):
        for j in range(len(present)):
            if not np.isnan(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.1f}", ha="center", va="center")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Percent")
    _save(fig, figure_root, stem)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="umwp", choices=("umwp", "worldsense"))
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test:
        raise RuntimeError("Refusing to touch test split without --allow-test")
    dataset_config = get_dataset_config(args.dataset)
    analysis_root = dataset_config.analysis_root / args.split
    figure_root = analysis_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    seed_path = analysis_root / "seed_metrics.csv"
    source_path = analysis_root / "source_metrics.csv"
    if not seed_path.exists():
        raise FileNotFoundError(f"Run analyze_validation first: {seed_path}")
    seed_metrics = pd.read_csv(seed_path)
    source_metrics = (
        pd.read_csv(source_path) if source_path.exists() else pd.DataFrame()
    )
    split_label = f"{dataset_config.name} {args.split}"
    plot_gate(seed_metrics, figure_root, split_label)
    plot_before_after(
        seed_metrics,
        "a_vanilla_accuracy",
        "a_system_accuracy",
        "Accuracy (%)",
        f"{split_label}: answerable trials — vanilla vs self-braking",
        figure_root,
        "fig2_answerable_accuracy",
    )
    plot_before_after(
        seed_metrics,
        "u_vanilla_abstain",
        "u_system_abstain",
        "Correct abstention rate (%)",
        f"{split_label}: unanswerable trials — vanilla vs self-braking",
        figure_root,
        "fig3_unanswerable_abstention",
    )
    plot_before_after(
        seed_metrics,
        "u_vanilla_hallucinated",
        "u_system_hallucinated",
        "Hallucinated-answer rate (%)",
        f"{split_label}: unanswerable hallucinated answers",
        figure_root,
        "fig4_unanswerable_hallucination",
    )
    plot_triggered_u(seed_metrics, figure_root)
    plot_triggered_a(seed_metrics, figure_root)
    plot_brake_fraction(seed_metrics, figure_root)
    plot_source_heatmap(
        source_metrics,
        "u_system_abstain",
        "Self-braking correct abstention by dataset source",
        figure_root,
        "fig8_source_u_abstention",
    )
    plot_source_heatmap(
        source_metrics,
        "a_system_accuracy",
        "Self-braking answerable accuracy by dataset source",
        figure_root,
        "fig9_source_a_accuracy",
    )
    print(f"Saved figures to: {figure_root}")


if __name__ == "__main__":
    main()
