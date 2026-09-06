#!/usr/bin/env python
"""Reproducible 95% confidence intervals added to the submitted pilot TFG.

Paired condition contrasts use a participant bootstrap of the mean paired
difference. Clip-level Spearman correlations use a clip bootstrap. OLS
coefficient intervals use the t distribution. The product-application odds
ratio uses the standard log-odds (Woolf) interval.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent
FINAL = ROOT / "final_results_output"
OLD = ROOT / "real_data_analysis_output"
OUT = FINAL / "thesis_confidence_intervals.csv"
SEED = 20260903
N_BOOT = 10_000
rng = np.random.default_rng(SEED)


def boot_mean(values):
    x = np.asarray(values, float)
    draws = rng.choice(x, size=(N_BOOT, len(x)), replace=True).mean(axis=1)
    return np.quantile(draws, [0.025, 0.975])


rows = []

# Final-QC synchrony and ocular contrasts: viral minus flop.
isc = pd.read_csv(FINAL / "gaze_pupil_isc_final.csv")
for channel in ("gaze_x", "gaze_y", "pupil"):
    d = isc[isc.channel.eq(channel)]
    pivot = d.groupby(["participant", "condition"]).isc_loo.mean().unstack().dropna()
    lo, hi = boot_mean(pivot.viral - pivot.flop)
    rows.append([channel, "participant", len(pivot), lo, hi, "paired bootstrap mean difference"])

metrics = pd.read_csv(FINAL / "participant_video_metrics_final.csv")
for measure, modality in (("fixation_rate_hz", "gaze"), ("mean_fix_dur", "gaze"),
                          ("gaze_dispersion", "gaze"), ("blink_rate_per_min", "gaze"),
                          ("pupil_mean_bc", "pupil")):
    d = metrics.copy()
    if modality == "gaze":
        d = d[d.gaze_eligible & d.gaze_clip_pass]
    else:
        d = d[d.pupil_eligible & d.pupil_clip_pass]
    pivot = d.groupby(["participant", "condition"])[measure].mean().unstack().dropna()
    lo, hi = boot_mean(pivot.viral - pivot.flop)
    rows.append([measure, "participant", len(pivot), lo, hi, "paired bootstrap mean difference"])

# Ratings: viral minus flop.
ratings = pd.read_csv(OLD / "ratings_long.csv")
for measure in ("rating_liking", "rating_purchase_intent", "rating_virality", "rating_brand_familiarity"):
    pivot = ratings.groupby(["participant", "video_condition"])[measure].mean().unstack().dropna()
    lo, hi = boot_mean(pivot.viral - pivot.flop)
    rows.append([measure, "participant", len(pivot), lo, hi, "paired bootstrap mean difference"])

# Mouse paired contrasts from the saved participant summaries.
wide = pd.read_csv(OLD / "rt_by_task_participant.csv").set_index("participant")
for name, diff in (("gonogo_go_minus_targeting", wide["gonogo_GO"] - wide["targeting"]),
                   ("gonogo_nogo_minus_gonogo_go", wide["gonogo_NOGO"] - wide["gonogo_GO"])):
    diff = diff.dropna()
    lo, hi = boot_mean(diff)
    rows.append([name, "participant", len(diff), lo, hi, "paired bootstrap mean difference"])

# Clip-level editing correlations and OLS coefficients.
edit = pd.read_csv(FINAL / "editing_rhythm_clip_data_final_qc.csv")
for predictor in ("cut_rate_per_s", "median_shot_s", "n_cuts"):
    vals = []
    for _ in range(N_BOOT):
        ix = rng.integers(0, len(edit), len(edit))
        rho = stats.spearmanr(edit[predictor].to_numpy()[ix], edit.mean_isc.to_numpy()[ix]).statistic
        if np.isfinite(rho):
            vals.append(rho)
    lo, hi = np.quantile(vals, [0.025, 0.975])
    rows.append([f"spearman_{predictor}", "advertisement", len(edit), lo, hi, "clip bootstrap Spearman rho"])

edit["viral_condition"] = edit.condition.eq("viral").astype(int)
for predictor in ("cut_rate_per_s", "n_cuts"):
    X = np.column_stack([np.ones(len(edit)), edit.viral_condition, edit[predictor]])
    y = edit.mean_isc.to_numpy()
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    df = len(y) - X.shape[1]
    cov = (resid @ resid / df) * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    crit = stats.t.ppf(0.975, df)
    for j, term in ((1, "viral_condition"), (2, predictor)):
        rows.append([f"ols_{predictor}_{term}", "advertisement", len(edit),
                     beta[j] - crit * se[j], beta[j] + crit * se[j], "t interval for OLS coefficient"])

# Product application, using the completed row-by-row verification.
content = pd.read_csv(OLD / "content_coding.csv")
viral = content[content.condition.eq("viral")].product_applied.astype(int)
flop = content[content.condition.eq("flop")].product_applied.astype(int)
a, b = int(viral.sum()), int((1 - viral).sum())
c, d = int(flop.sum()), int((1 - flop).sum())
odds_ratio = (a * d) / (b * c)
se_log = np.sqrt(1/a + 1/b + 1/c + 1/d)
lo, hi = np.exp(np.log(odds_ratio) + np.array([-1, 1]) * 1.96 * se_log)
rows.append(["product_application_odds_ratio", "advertisement", 26, lo, hi, "Woolf log-odds interval"])

# One-second cut-censoring sensitivity contrasts (viral minus flop ISC).
censored = pd.read_csv(FINAL / "isc_cut_censored.csv")
censored = censored[censored.window_s.eq(1.0)]
for channel in ("gaze_x", "gaze_y"):
    d = censored[censored.channel.eq(channel)]
    for column in ("isc_full", "isc_cut_censored", "isc_random_censored"):
        pivot = d.groupby(["participant", "condition"])[column].mean().unstack().dropna()
        lo, hi = boot_mean(pivot.viral - pivot.flop)
        rows.append([f"one_second_{channel}_{column}", "participant", len(pivot), lo, hi,
                     "paired bootstrap mean difference"])

out = pd.DataFrame(rows, columns=["estimate", "unit", "n", "ci95_low", "ci95_high", "method"])
out.to_csv(OUT, index=False)
print(out.to_string(index=False))
print(OUT)
