#!/usr/bin/env python
"""Complete post-classifier outputs after author verification of product use.

The classifier performance file has already been written by
ml_feature_importance.py. This script reproduces the remaining single-feature,
SHAP and liking-model stages without rerunning the expensive permutation loop.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
import xgboost as xgb
import shap

import ml_feature_importance as ml


OUT = Path(__file__).resolve().parent / "real_data_analysis_output"


def bh_fdr(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


clip = ml.load_clip_table(OUT)
clip.to_csv(OUT / "ml_clip_table.csv", index=False)
features = [c for c in [
    "n_cuts", "cut_rate_per_s", "median_shot_s", "mean_shot_s", "shake",
    "shake_p90", "sharpness", "sharpness_stability", "exposure_stability",
    "luminance_mean", "contrast", "saturation_mean", "saturation_sd",
    "edge_density", "face_present_frac", "face_screen_share", "duration_s",
    "bitrate_mbps", "isc_gaze_x", "isc_gaze_y", "isc_pupil", "fix_rate",
    "mean_fix_dur", "gaze_dispersion", "pupil_mean_bc", "mean_rating_liking",
    "mean_rating_purchase_intent", "mean_rating_virality", "face_visible",
    "product_applied",
] if c in clip.columns]

rows = []
for feature in features:
    viral = clip.loc[clip.y.eq(1), feature].dropna()
    flop = clip.loc[clip.y.eq(0), feature].dropna()
    if len(viral) < 5 or len(flop) < 5:
        continue
    u, p = stats.mannwhitneyu(viral, flop)
    raw_auc = u / (len(viral) * len(flop))
    rows.append({
        "feature": feature,
        "viral_mean": viral.mean(),
        "flop_mean": flop.mean(),
        "auc": max(raw_auc, 1 - raw_auc),
        "direction": "viral>flop" if viral.median() > flop.median() else "flop>viral",
        "p": p,
    })
single = pd.DataFrame(rows).sort_values("auc", ascending=False)
single["p_fdr"] = bh_fdr(single.p)
single.to_csv(OUT / "ml_modelA_single_feature.csv", index=False)

X = clip[features]
y = clip.y.to_numpy()
Xi = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=features)
classifier = xgb.XGBClassifier(
    n_estimators=300, max_depth=2, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.6, reg_lambda=5.0, random_state=ml.SEED,
    eval_metric="logloss", n_jobs=-1,
).fit(Xi, y)
try:
    sv = shap.TreeExplainer(classifier).shap_values(Xi)
except (ValueError, TypeError):
    sv = classifier.get_booster().predict(xgb.DMatrix(Xi), pred_contribs=True)[:, :-1]
importance = pd.DataFrame({
    "feature": features,
    "mean_abs_shap": np.abs(sv).mean(0),
}).sort_values("mean_abs_shap", ascending=False)
importance.to_csv(OUT / "ml_modelA_shap.csv", index=False)

print("Classifier performance:")
print(pd.read_csv(OUT / "ml_modelA_performance.csv").to_string(index=False))
print("\nSingle-feature ranking:")
print(single.head(10).to_string(index=False))
print("\nClassifier SHAP ranking:")
print(importance.head(10).to_string(index=False))

ml.model_b(ml.load_pair_table(OUT), OUT)
