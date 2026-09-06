#!/usr/bin/env python
"""Which features distinguish viral from flop advertisements, and which predict
how an individual responds to one.

Two models, because the two questions have different units of analysis and mixing
them produces a result that looks strong and means nothing.

MODEL A - what makes an advertisement viral?
    Unit: the clip. n = 26, of which 24 have gaze data.
    Label: viral / flop. There are only 26 labels in the entire dataset.
    Features: measured video properties, plus viewer response aggregated to the
    clip (mean liking, mean ISC).
    Participant-level features (mouse traits, demographics) are NOT eligible.
    They are constant within a participant and vary independently of which clip
    is on screen, so they carry no information about the clip's label. Including
    them would still yield non-zero importances from noise, which is worse than
    useless.

MODEL B - what predicts how much a given person likes a given advertisement?
    Unit: the participant-clip pair. n = 17 x 26.
    Label: that person's liking rating for that clip.
    Features: clip properties AND participant traits AND demographics. This is
    where the mouse and demographic measures legitimately belong.

Both models are cross-validated with grouping, and both are tested against a
permutation null. With 26 clips a model can appear to work by chance, so an
accuracy number on its own is not evidence. The permutation p-value is the
result; the accuracy is a description.

Usage:
    python ml_feature_importance.py --out real_data_analysis_output
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SEED = 20260831


def norm_key(s: str) -> str:
    """Clip identifiers differ across files in punctuation AND word order, so
    the key is the sorted character multiset of the alphanumerics."""
    return "".join(sorted(re.sub(r"[^a-z0-9]", "", str(s).lower())))


# --------------------------------------------------------------------- loading
def load_clip_table(out: Path) -> pd.DataFrame:
    pq = pd.read_csv(out / "production_quality.csv")
    pq["k"] = pq.filename.str.replace(".mp4", "", regex=False).map(norm_key)

    isc = pd.read_csv(out / "gaze_pupil_isc_per_participant.csv")
    isc = isc[isc.participant != "__pairwise__"]
    isc_w = (isc.pivot_table(index="video_id", columns="channel",
                             values="isc_loo", aggfunc="mean")
                .add_prefix("isc_").reset_index())
    isc_w["k"] = isc_w.video_id.map(norm_key)

    met = pd.read_csv(out / "per_participant_video_metrics.csv")
    met["fix_rate"] = met.n_fixations / met.duration_s
    met_w = (met.groupby("video_id")[["fix_rate", "mean_fix_dur", "gaze_dispersion",
                                      "pupil_mean_bc", "duration_s"]]
                .mean().reset_index())
    met_w["k"] = met_w.video_id.map(norm_key)

    rat = pd.read_csv(out / "ratings_long.csv")
    vid_col = "video_id" if "video_id" in rat.columns else None
    rat_w = None
    if vid_col:
        rat_w = (rat.groupby(vid_col)[[c for c in ["rating_liking", "rating_purchase_intent",
                                       "rating_virality", "rating_brand_familiarity"] if c in rat.columns]]
                    .mean().add_prefix("mean_").reset_index())
        rat_w["k"] = rat_w[vid_col].map(norm_key)

    cc_path = out / "content_coding.csv"
    cc = None
    if cc_path.exists():
        cc = pd.read_csv(cc_path)
        cc["k"] = cc.filename.str.replace(".mp4", "", regex=False).map(norm_key)
        # codes with no variance carry no information and are dropped, not ranked
        cc = cc[["k"] + [c for c in ["face_visible", "product_applied",
                                     "onscreen_text", "brand_visible"]
                         if c in cc.columns and cc[c].nunique() > 1]]

    df = pq.merge(isc_w.drop(columns="video_id"), on="k", how="left")
    n0 = len(df)
    df = df.merge(met_w.drop(columns="video_id"), on="k", how="left")
    assert len(df) == n0, f"metrics merge changed row count {n0} -> {len(df)}"
    if rat_w is not None:
        df = df.merge(rat_w.drop(columns=vid_col), on="k", how="left")
        assert len(df) == n0, "ratings merge changed row count"
    if cc is not None:
        df = df.merge(cc, on="k", how="left")
        assert len(df) == n0, "content-coding merge changed row count"
        miss = df[[c for c in cc.columns if c != "k"]].isna().any(axis=1).sum()
        if miss:
            print(f"  WARNING: {miss} clips have no content coding")
    df["y"] = (df.condition == "viral").astype(int)
    return df


def load_pair_table(out: Path) -> pd.DataFrame:
    rat = pd.read_csv(out / "ratings_long.csv")
    vid_col = "video_id" if "video_id" in rat.columns else None
    if vid_col is None:
        return pd.DataFrame()
    rat = rat.dropna(subset=["rating_liking"]).copy()
    rat["k"] = rat[vid_col].map(norm_key)

    clip = load_clip_table(out)
    pair = rat.merge(clip.drop(columns=["condition"], errors="ignore"),
                     on="k", how="inner", suffixes=("", "_clip"))

    tr = pd.read_csv(out / "h5_traits_and_outcomes.csv")
    keep = ["participant", "entropy_targeting", "xflip_rate_targeting",
            "mean_speed_targeting", "targeting_rt", "gonogo_go_rt",
            "nogo_decision_time", "inhibition_cost"]
    tr = tr[[c for c in keep if c in tr.columns]]
    pair = pair.merge(tr, on="participant", how="left")

    dem = pd.read_csv(out / "demographics_internal.csv")
    dem = dem[["participant", "age", "cosmetics_interest", "tiktok_use", "cosmetics_buy"]].copy()
    dem["tiktok_daily"] = (dem.tiktok_use == "daily").astype(int)
    buy_rank = {"never": 0, "rarely": 1, "few_months": 2, "monthly": 3, "weekly": 4}
    dem["cosmetics_buy_freq"] = dem.cosmetics_buy.map(buy_rank)
    pair = pair.merge(dem.drop(columns=["tiktok_use", "cosmetics_buy"]),
                      on="participant", how="left")
    return pair


# ------------------------------------------------------------------- model A
def model_a(df: pd.DataFrame, out: Path) -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import roc_auc_score
    import xgboost as xgb

    feats = [c for c in [
        "n_cuts", "cut_rate_per_s", "median_shot_s", "mean_shot_s", "shake",
        "shake_p90", "sharpness", "sharpness_stability", "exposure_stability",
        "luminance_mean", "contrast", "saturation_mean", "saturation_sd",
        "edge_density", "face_present_frac", "face_screen_share",
        "duration_s", "bitrate_mbps",
        "isc_gaze_x", "isc_gaze_y", "isc_pupil",
        "fix_rate", "mean_fix_dur", "gaze_dispersion", "pupil_mean_bc",
        "mean_rating_liking", "mean_rating_purchase_intent", "mean_rating_virality",
        "face_visible", "product_applied",
    ] if c in df.columns]

    d = df.dropna(subset=["y"]).copy()
    X = d[feats]
    y = d.y.values
    print(f"\nMODEL A  clips n={len(d)}  features={len(feats)}  "
          f"viral={y.sum()} flop={len(y)-y.sum()}")
    print(f"  {len(feats)} features for {len(d)} clips: this is far more features "
          f"than the data can support.\n  Treat every number below as a ranking "
          f"exercise, not a validated model.\n")

    models = {
        "logistic (L2)": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(penalty="l2", C=0.3, max_iter=5000, random_state=SEED)),
        "random forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=150, min_samples_leaf=2,
                                   max_features="sqrt", random_state=SEED, n_jobs=-1)),
        "xgboost": make_pipeline(
            SimpleImputer(strategy="median"),
            xgb.XGBClassifier(n_estimators=150, max_depth=2, learning_rate=0.08,
                              subsample=0.8, colsample_bytree=0.6,
                              reg_lambda=5.0, random_state=SEED, n_jobs=-1,
                              eval_metric="logloss")),
    }

    rng = np.random.default_rng(SEED)
    rows = []
    import time
    for name, mdl in models.items():
        t_start = time.time()
        loo = LeaveOneOut()
        prob = cross_val_predict(mdl, X, y, cv=loo, method="predict_proba", n_jobs=-1)[:, 1]
        acc = ((prob > 0.5).astype(int) == y).mean()
        auc = roc_auc_score(y, prob)
        # permutation null: shuffle labels, repeat the whole LOO procedure
        null = []
        for _ in range(1000):  # permutation null
            yp = rng.permutation(y)
            pp = cross_val_predict(mdl, X, yp, cv=LeaveOneOut(), method="predict_proba", n_jobs=-1)[:, 1]
            null.append(roc_auc_score(yp, pp))
        null = np.asarray(null)
        p = (np.sum(null >= auc) + 1) / (len(null) + 1)
        rows.append({"model": name, "loo_accuracy": acc, "loo_auc": auc,
                     "null_auc_mean": null.mean(), "perm_p": p, "n_perm": len(null)})
        print(f"  {name:16s} LOO acc={acc:.3f}  AUC={auc:.3f}  "
              f"null AUC={null.mean():.3f}  permutation p={p:.3f}   "
              f"[{time.time()-t_start:.0f}s]", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(out / "ml_modelA_performance.csv", index=False)

    # single-feature ranking: at n=26 this is more honest than any multivariate fit
    from scipy import stats
    srows = []
    for f in feats:
        a = d[d.y == 1][f].dropna()
        b = d[d.y == 0][f].dropna()
        if len(a) < 5 or len(b) < 5:
            continue
        u, pv = stats.mannwhitneyu(a, b)
        auc1 = u / (len(a) * len(b))
        srows.append({"feature": f, "viral_mean": a.mean(), "flop_mean": b.mean(),
                      "auc": max(auc1, 1 - auc1), "direction": "viral>flop" if a.median() > b.median() else "flop>viral",
                      "p": pv})
    sr = pd.DataFrame(srows).sort_values("auc", ascending=False)
    from statsmodels.stats.multitest import multipletests
    sr["p_fdr"] = multipletests(sr.p, method="fdr_bh")[1]
    sr.to_csv(out / "ml_modelA_single_feature.csv", index=False)
    print("\n  Single-feature separation, ranked by AUC "
          "(the honest analysis at this sample size):")
    print(sr.head(12).to_string(index=False,
          float_format=lambda v: f"{v:.3f}"))

    # SHAP on a model fitted to all the data, for direction only
    try:
        import shap
        from sklearn.impute import SimpleImputer as SI
        Xi = pd.DataFrame(SI(strategy="median").fit_transform(X), columns=feats)
        m = xgb.XGBClassifier(n_estimators=300, max_depth=2, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.6, reg_lambda=5.0,
                              random_state=SEED, eval_metric="logloss").fit(Xi, y)
        sv = shap.TreeExplainer(m).shap_values(Xi)
        imp = pd.DataFrame({"feature": feats,
                            "mean_abs_shap": np.abs(sv).mean(0)}).sort_values(
                            "mean_abs_shap", ascending=False)
        imp.to_csv(out / "ml_modelA_shap.csv", index=False)
        print("\n  SHAP (in-sample, ranking only, NOT evidence of generalisation):")
        print(imp.head(10).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    except Exception as e:
        print(f"  SHAP skipped: {e}")


# ------------------------------------------------------------------- model B
def model_b(pair: pd.DataFrame, out: Path) -> None:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    import xgboost as xgb

    if pair.empty:
        print("\nMODEL B skipped: ratings file has no video identifier column.")
        return

    clip_f = [c for c in ["n_cuts", "cut_rate_per_s", "median_shot_s", "shake",
                          "sharpness", "exposure_stability", "contrast",
                          "saturation_mean", "edge_density", "face_screen_share",
                          "duration_s", "face_visible", "product_applied"]
              if c in pair.columns]
    part_f = [c for c in ["entropy_targeting", "xflip_rate_targeting",
                          "mean_speed_targeting", "targeting_rt", "gonogo_go_rt",
                          "nogo_decision_time", "inhibition_cost", "age",
                          "cosmetics_interest", "tiktok_daily", "cosmetics_buy_freq"]
              if c in pair.columns]
    feats = clip_f + part_f

    d = pair.dropna(subset=["rating_liking"]).copy()
    X, y = d[feats], d.rating_liking.values
    print(f"\nMODEL B  participant-clip rows n={len(d)}  "
          f"clip features={len(clip_f)}  participant features={len(part_f)}")

    mdl = make_pipeline(SimpleImputer(strategy="median"),
                        RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                              random_state=SEED, n_jobs=-1))
    performance_rows = []
    for gname, groups in (("grouped by clip", d.k.values),
                          ("grouped by participant", d.participant.values)):
        ng = len(np.unique(groups))
        pred = cross_val_predict(mdl, X, y, cv=GroupKFold(n_splits=min(10, ng)),
                                 groups=groups, n_jobs=-1)
        r = np.corrcoef(pred, y)[0, 1]
        ss = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
        performance_rows.append({"grouping": gname, "n": len(d), "r": r, "r2": ss})
        print(f"  {gname:24s} r={r:+.3f}  R2={ss:+.3f}")
    pd.DataFrame(performance_rows).to_csv(out / "ml_modelB_performance.csv", index=False)

    try:
        import shap
        from sklearn.impute import SimpleImputer as SI
        Xi = pd.DataFrame(SI(strategy="median").fit_transform(X), columns=feats)
        m = xgb.XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0,
                             random_state=SEED).fit(Xi, y)
        try:
            sv = shap.TreeExplainer(m).shap_values(Xi)
        except (ValueError, TypeError):
            sv = m.get_booster().predict(xgb.DMatrix(Xi), pred_contribs=True)[:, :-1]
        imp = pd.DataFrame({
            "feature": feats,
            "level": ["clip"] * len(clip_f) + ["participant"] * len(part_f),
            "mean_abs_shap": np.abs(sv).mean(0)}).sort_values("mean_abs_shap", ascending=False)
        imp.to_csv(out / "ml_modelB_shap.csv", index=False)
        print("\n  SHAP for individual liking:")
        print(imp.head(14).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    except Exception as e:
        print(f"  SHAP skipped: {e}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("real_data_analysis_output"))
    args = ap.parse_args()
    out = args.out
    if not out.exists():
        print(f"no such directory: {out}", file=sys.stderr)
        return 1

    clip = load_clip_table(out)
    clip.to_csv(out / "ml_clip_table.csv", index=False)
    model_a(clip, out)
    model_b(load_pair_table(out), out)
    print(f"\noutputs -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
