#!/usr/bin/env python
"""Create thesis-ready figures from the final QC-aligned outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analyse_gaze_pupil_isc import fdr


BLUE = "#2F6B9A"
LIGHT_BLUE = "#86B3D1"
GREY = "#A7ADB4"
DARK_GREY = "#626970"
RED = "#B44C4C"
GRID = "#E3E6E8"


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def panel_label(ax, letter):
    ax.text(-0.16, 1.06, letter, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top")


def significance_bar(ax, left=0, right=1, label="*"):
    """Add a compact comparison bracket without obscuring the plotted data."""
    low, high = ax.get_ylim()
    span = high - low if high > low else 1.0
    y = high + 0.08 * span
    tick = 0.025 * span
    ax.plot([left, left, right, right], [y - tick, y, y, y - tick],
            color="black", linewidth=0.9, clip_on=False)
    ax.text((left + right) / 2, y + 0.01 * span, label,
            ha="center", va="bottom", fontsize=10, fontweight="bold",
            clip_on=False)
    ax.set_ylim(low, high + 0.20 * span)


def significance_label(p):
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return None


def save_both(fig, out: Path, stem: str):
    fig.savefig(out / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight", facecolor="white")


def condition_dots(ax, data, value, ylabel, title, rng, significance=None):
    for index, condition in enumerate(("flop", "viral")):
        values = data.loc[data.condition == condition, value].dropna().to_numpy()
        jitter = rng.uniform(-0.075, 0.075, len(values))
        color = GREY if condition == "flop" else BLUE
        ax.scatter(np.full(len(values), index) + jitter, values, s=28,
                   facecolor=color, edgecolor="white", linewidth=0.7, alpha=0.9, zorder=3)
        mean = float(np.mean(values))
        boot = np.array([
            np.mean(rng.choice(values, len(values), replace=True)) for _ in range(5000)
        ])
        low, high = np.percentile(boot, [2.5, 97.5])
        ax.errorbar(index, mean, yerr=[[mean-low], [high-mean]], fmt="D",
                    color="black", mfc="white", mec="black", ms=5, capsize=3,
                    linewidth=1.2, zorder=5)
    ax.set_xticks([0, 1], ["Flop", "Viral"])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9, pad=7)
    style_axis(ax)
    if significance:
        significance_bar(ax, label=significance)


def paired_panel(ax, data, participant, condition, value, title, ylabel,
                 significance=None):
    pivot = data.groupby([participant, condition])[value].mean().unstack().dropna()
    for _, row in pivot.iterrows():
        ax.plot([0, 1], [row["flop"], row["viral"]], color=GREY, alpha=0.65,
                linewidth=0.9, marker="o", markersize=3.2,
                markerfacecolor="white", markeredgecolor=GREY, zorder=2)
    means = [pivot["flop"].mean(), pivot["viral"].mean()]
    ax.plot([0, 1], means, color=BLUE, linewidth=2.3, marker="o", markersize=5.5,
            markerfacecolor=BLUE, markeredgecolor="white", zorder=4)
    ax.set_xticks([0, 1], ["Flop", "Viral"])
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n(n = {len(pivot)})", fontsize=9, pad=6)
    style_axis(ax)
    if significance:
        significance_bar(ax, label=significance)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    rng = np.random.default_rng(20260831)

    qc = pd.read_csv(args.final / "participant_modality_qc_final.csv")
    metrics = pd.read_csv(args.final / "participant_video_metrics_final.csv")
    isc = pd.read_csv(args.final / "gaze_pupil_isc_final.csv")
    clip_isc = pd.read_csv(args.final / "clip_level_isc_final.csv")
    ratings = pd.read_csv(args.previous / "ratings_long.csv")
    rt = pd.read_csv(args.previous / "rt_by_task_participant.csv")
    targeting = pd.read_csv(args.previous / "mouse_targeting_by_participant.csv")
    gonogo = pd.read_csv(args.previous / "mouse_gonogo_by_participant.csv")

    # Figure 1: quality-control outcome.
    qc_plot = qc.sort_values(
        ["gaze_passed_clips", "pupil_passed_clips", "participant"],
        ascending=[False, False, True],
    )
    y = np.arange(len(qc_plot))
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.scatter(qc_plot.gaze_passed_clips, y - 0.12, s=35, color=BLUE,
               edgecolor="white", linewidth=0.7, label="Gaze", zorder=3)
    ax.scatter(qc_plot.pupil_passed_clips, y + 0.12, s=35, color=LIGHT_BLUE,
               edgecolor="white", linewidth=0.7, label="Pupil", zorder=3)
    ax.axvline(23, color=RED, linestyle=(0, (4, 3)), linewidth=1.2,
               label="Complete-session reference (23 of 26)")
    ax.set_yticks(y, qc_plot.participant)
    ax.invert_yaxis()
    ax.set_xlim(-0.5, 26.8)
    ax.set_xticks([0, 5, 10, 15, 20, 23, 26])
    ax.set_xlabel("Clips passing modality-specific quality control (out of 26)")
    ax.set_ylabel("Participant code")
    incomplete = qc.loc[~qc.complete_session]
    for row in incomplete.itertuples():
        position = int(np.flatnonzero(qc_plot.participant.eq(row.participant))[0])
        ax.annotate(
            f"{row.gaze_passed_clips}/{row.observed_clips} observed; "
            f"{row.missing_clips} missing",
            (row.gaze_passed_clips, position - 0.12),
            xytext=(-8, 8), textcoords="offset points", ha="right", va="bottom",
            fontsize=6.5, color=DARK_GREY,
        )
    ax.legend(frameon=False, ncol=3, loc="lower right", fontsize=7)
    style_axis(ax)
    fig.tight_layout()
    save_both(fig, args.out, "Figure_1_data_quality")
    plt.close(fig)

    # Figure 2: primary participant-level paired gaze-synchrony result.
    from scipy import stats
    raw_p = []
    for channel in ("gaze_x", "gaze_y", "pupil"):
        pivot = (isc.loc[isc.channel.eq(channel)]
                   .groupby(["participant", "condition"])["isc_loo"]
                   .mean().unstack().dropna())
        raw_p.append(stats.wilcoxon(pivot.viral, pivot.flop).pvalue)
    adjusted_p = dict(zip(("gaze_x", "gaze_y", "pupil"), fdr(raw_p)))

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))
    for ax, channel, title, letter in zip(
        axes, ("gaze_x", "gaze_y"),
        ("Horizontal gaze synchrony", "Vertical gaze synchrony"), ("A", "B")
    ):
        subset = isc[isc.channel == channel]
        significance = significance_label(adjusted_p[channel])
        paired_panel(ax, subset, "participant", "condition", "isc_loo",
                     title, "Mean gaze synchrony\n(correlation, r)",
                     significance=significance)
        panel_label(ax, letter)
    fig.text(0.99, 0.01, "* p < .05; ** p < .01 (FDR-adjusted)",
             ha="right", va="bottom", fontsize=7, color=DARK_GREY)
    fig.tight_layout(w_pad=2.0)
    save_both(fig, args.out, "Figure_2_primary_gaze_synchrony")
    plt.close(fig)

    # Figure 3: secondary ocular outcomes, paired within participant.
    gaze_metrics = metrics[metrics.gaze_eligible & metrics.gaze_clip_pass]
    pupil_metrics = metrics[metrics.pupil_eligible & metrics.pupil_clip_pass]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75))
    specifications = [
        (axes[0], gaze_metrics, "fixation_rate_hz", "Fixation rate", "Fixations per second"),
        (axes[1], gaze_metrics, "gaze_dispersion", "Gaze dispersion", "Normalised screen units"),
        (axes[2], pupil_metrics, "pupil_mean_bc", "Pupil response", "Baseline-corrected diameter (mm)"),
    ]
    for letter, (ax, frame, value, title, ylabel) in zip("ABC", specifications):
        paired_panel(ax, frame, "participant", "condition", value, title, ylabel)
        panel_label(ax, letter)
    fig.tight_layout(w_pad=1.7)
    save_both(fig, args.out, "Figure_3_secondary_ocular_measures")
    plt.close(fig)

    # Figure 4: subjective ratings.
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75))
    rating_specs = [
        ("rating_liking", "Liking", "Mean rating (-3 to +3)"),
        ("rating_purchase_intent", "Purchase intention", "Mean rating (0-10)"),
        ("rating_virality", "Perceived virality", "Mean rating (-3 to +3)"),
    ]
    for letter, ax, (value, title, ylabel) in zip("ABC", axes, rating_specs):
        significance = "**" if value == "rating_liking" else None
        paired_panel(ax, ratings, "participant", "video_condition", value,
                     title, ylabel, significance=significance)
        panel_label(ax, letter)
    fig.text(0.99, 0.01, "* p < .05; ** p < .01 (FDR-adjusted)",
             ha="right", va="bottom", fontsize=7, color=DARK_GREY)
    fig.tight_layout(w_pad=1.7)
    save_both(fig, args.out, "Figure_4_subjective_ratings")
    plt.close(fig)

    # Figure 5: mouse-task performance.
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75))
    ax = axes[0]
    task_cols = ["targeting", "gonogo_GO", "gonogo_NOGO"]
    for _, row in rt.iterrows():
        ax.plot(range(3), row[task_cols], color=GREY, alpha=0.65, linewidth=0.9,
                marker="o", markersize=3, markerfacecolor="white", markeredgecolor=GREY)
    ax.plot(range(3), rt[task_cols].mean(), color=BLUE, linewidth=2.3,
            marker="o", markersize=5.5, markeredgecolor="white")
    ax.set_xticks(range(3), ["Targeting", "Go", "No-go"])
    ax.set_ylabel("Mean response time (s)")
    ax.set_title(f"Response time\n(n = {len(rt)})", fontsize=9)
    style_axis(ax); panel_label(ax, "A")

    ax = axes[1]
    jitter = rng.uniform(-0.06, 0.06, len(targeting))
    ax.scatter(jitter, targeting.click_error, s=30, color=BLUE,
               edgecolor="white", linewidth=0.7)
    ax.scatter([0], [targeting.click_error.mean()], marker="D", s=40,
               facecolor="white", edgecolor="black", zorder=4)
    ax.set_xticks([0], ["Targeting"])
    ax.set_ylabel("Mean click error\n(normalised screen units)")
    ax.set_title(f"Targeting precision\n(n = {len(targeting)})", fontsize=9)
    style_axis(ax); panel_label(ax, "B")

    ax = axes[2]
    jitter = rng.uniform(-0.06, 0.06, len(gonogo))
    ax.scatter(jitter, gonogo.commission_errors, s=30, color=BLUE,
               edgecolor="white", linewidth=0.7)
    ax.set_xticks([0], ["No-go trials"])
    ax.set_yticks(sorted(gonogo.commission_errors.unique()))
    ax.set_ylabel("Commission errors (out of 10)")
    ax.set_title(f"Response-control errors\n(n = {len(gonogo)})", fontsize=9)
    style_axis(ax); panel_label(ax, "C")

    fig.tight_layout(w_pad=1.7)
    save_both(fig, args.out, "Figure_5_mouse_task_performance")
    plt.close(fig)

    print("Created 5 thesis figures (PNG and PDF)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
