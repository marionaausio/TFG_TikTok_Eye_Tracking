#!/usr/bin/env python
"""Editing-rhythm associations on the current final QC-aligned gaze sample."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent
FINAL = ROOT / "final_results_output"
PREVIOUS = ROOT / "real_data_analysis_output"


def norm_key(value: str) -> str:
    return "".join(sorted(re.sub(r"[^a-z0-9]", "", str(value).lower())))


isc = pd.read_csv(FINAL / "clip_level_isc_final.csv")
quality = pd.read_csv(PREVIOUS / "production_quality.csv")

vertical = isc.loc[isc.channel.eq("gaze_y")].copy()
vertical["key"] = vertical.video_id.map(norm_key)
quality["key"] = quality.filename.str.replace(".mp4", "", regex=False).map(norm_key)
columns = ["key", "n_cuts", "cut_rate_per_s", "median_shot_s"]
data = vertical.merge(quality[columns], on="key", how="left", validate="one_to_one")
if data[columns[1:]].isna().any().any():
    raise RuntimeError("Unmatched editing-rhythm values")

rows = []
for predictor in ("cut_rate_per_s", "median_shot_s", "n_cuts"):
    rho, p = stats.spearmanr(data[predictor], data.mean_isc)
    rows.append({
        "analysis": "spearman",
        "predictor": predictor,
        "n_clips": len(data),
        "estimate": rho,
        "p": p,
    })

data["viral_condition"] = data.condition.eq("viral").astype(int)
for predictor in ("cut_rate_per_s", "n_cuts"):
    terms = ["intercept", "viral_condition", predictor]
    design = np.column_stack([
        np.ones(len(data)),
        data["viral_condition"].to_numpy(float),
        data[predictor].to_numpy(float),
    ])
    outcome = data["mean_isc"].to_numpy(float)
    beta, _, _, _ = np.linalg.lstsq(design, outcome, rcond=None)
    residual = outcome - design @ beta
    df_resid = len(outcome) - design.shape[1]
    sigma2 = float(residual @ residual / df_resid)
    covariance = sigma2 * np.linalg.inv(design.T @ design)
    standard_error = np.sqrt(np.diag(covariance))
    t_values = beta / standard_error
    p_values = 2 * stats.t.sf(np.abs(t_values), df_resid)
    for index, term in enumerate(terms[1:], start=1):
        rows.append({
            "analysis": f"ols_with_{predictor}",
            "predictor": term,
            "n_clips": len(data),
            "estimate": beta[index],
            "p": p_values[index],
        })

result = pd.DataFrame(rows)
result.to_csv(FINAL / "editing_rhythm_final_qc.csv", index=False)
data.to_csv(FINAL / "editing_rhythm_clip_data_final_qc.csv", index=False)
print(result.to_string(index=False))
