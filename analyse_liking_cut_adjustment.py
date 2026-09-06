#!/usr/bin/env python
"""Does the viral-versus-flop liking difference survive adjustment for cut count?

Part A reproduces the original adjustment used for the descriptive thesis
result. Part B is a separate sensitivity analysis added on 2026-09-04 that
addresses the statistical limitation identified in Part A.

PART A - ORIGINAL PROCEDURE (exact reproduction)
  A1  raw participant-level paired difference in liking, viral minus flop
  A2  clip-level OLS   mean_liking ~ n_cuts   -> one residual per clip
      merge each clip's residual back onto every rating of that clip
      per participant, mean residual for viral minus mean residual for flop
      "% retained" = adjusted mean difference / raw mean difference
  A3  clip-level OLS   mean_liking ~ viral + n_cuts

THE PROBLEM WITH A2
  Every participant rated (almost) all 26 clips, so every participant's
  "mean viral residual" is the mean of the SAME 13 numbers, and likewise for
  flop. The adjusted difference is therefore nearly identical across all 17
  participants; the only variation comes from the six missing ratings. That is
  why its SD collapsed from 0.416 to 0.054 and d_z inflated to -1.47 while the
  actual effect shrank by 80%. A Wilcoxon test on a near-constant is not a test
  of anything. The 80% figure is a legitimate DESCRIPTIVE quantity; the adjusted
  d_z and p-value reported alongside it are not valid inferential statistics.

PART B - SENSITIVITY ANALYSIS (added, not a replacement)
  Linear mixed model on all 436 participant-by-clip ratings with crossed random
  intercepts for participant and for advertisement:
      rating_liking ~ viral                     (B1)
      rating_liking ~ viral + n_cuts            (B2)
  The condition coefficient is estimated once, with uncertainty that respects
  both sources of repeated measurement. The change in the viral coefficient from
  B1 to B2 is the defensible analogue of the "80% reduction".

Usage:
    python analyse_liking_cut_adjustment.py --data real_data_analysis_output
"""
from __future__ import annotations

import argparse
import platform
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def norm_key(s: str) -> str:
    """Clip identifiers differ across files in punctuation and word order."""
    return "".join(sorted(re.sub(r"[^a-z0-9]", "", str(s).lower())))


def load(data: Path):
    r = pd.read_csv(data / "ratings_long.csv")
    r["k"] = r.video_id.map(norm_key)
    p = pd.read_csv(data / "production_quality.csv")
    p["k"] = p.filename.str.replace(".mp4", "", regex=False).map(norm_key)
    m = pd.read_csv(data / "per_participant_video_metrics.csv")
    m["k"] = m.video_id.map(norm_key)
    p = p.merge(m.groupby("k").duration_s.mean(), on="k", how="left")
    d = r.merge(p[["k", "n_cuts", "cut_rate_per_s", "duration_s"]], on="k")
    assert len(d) == len(r), f"merge changed row count {len(r)} -> {len(d)}"
    assert d.n_cuts.notna().all(), "some ratings have no cut count"
    return d, p


# ============================================================ PART A (original)
def part_a(d: pd.DataFrame, p: pd.DataFrame, rows: list) -> None:
    from scipy import stats
    import statsmodels.api as sm

    # A1 raw paired difference
    piv = d.groupby(["participant", "video_condition"]).rating_liking.mean().unstack()
    raw = piv.viral - piv.flop
    w_raw, p_raw = stats.wilcoxon(piv.viral, piv.flop)
    dz_raw = raw.mean() / raw.std(ddof=1)
    rows.append(dict(part="A1", analysis="raw paired difference (viral - flop)",
                     unit="participant", n=len(raw), estimate=raw.mean(),
                     sd=raw.std(ddof=1), dz=dz_raw, stat=w_raw, p=p_raw))
    print(f"A1  raw difference: mean {raw.mean():+.6f}  sd {raw.std(ddof=1):.6f}  "
          f"dz {dz_raw:+.6f}  W={w_raw:.0f}  p={p_raw:.6f}  n={len(raw)}")

    # A2 clip-level residualisation, for each covariate the original tried
    for cov in ["n_cuts", "duration_s", "cut_rate_per_s"]:
        cl = (d.groupby(["k", "video_condition"]).rating_liking.mean()
                .reset_index().merge(p[["k", cov]], on="k"))
        X = sm.add_constant(cl[[cov]].astype(float))
        res = sm.OLS(cl.rating_liking, X).fit()
        cl["resid"] = res.resid
        dd = d.merge(cl[["k", "resid"]], on="k")
        pv = dd.groupby(["participant", "video_condition"]).resid.mean().unstack()
        adj = pv.viral - pv.flop
        w_adj, p_adj = stats.wilcoxon(pv.viral, pv.flop)
        dz_adj = adj.mean() / adj.std(ddof=1)
        retained = 100 * adj.mean() / raw.mean()
        rows.append(dict(part="A2", analysis=f"clip-residualised on {cov}, paired difference",
                         unit="participant", n=len(adj), estimate=adj.mean(),
                         sd=adj.std(ddof=1), dz=dz_adj, stat=w_adj, p=p_adj,
                         pct_retained=retained, pct_removed=100 - retained,
                         covariate_slope=res.params[cov], covariate_slope_p=res.pvalues[cov],
                         note="dz and p NOT valid: adjusted differences are near-constant "
                              "across participants (see docstring)"))
        print(f"A2  adjusted for {cov:15s}: mean {adj.mean():+.6f}  sd {adj.std(ddof=1):.6f}  "
              f"dz {dz_adj:+.6f}  p={p_adj:.6f}  retained {retained:.3f}%  removed {100-retained:.3f}%")

    # A3 clip-level OLS with both terms
    c = (d.groupby(["k", "video_condition"]).rating_liking.mean().reset_index()
           .merge(p[["k", "n_cuts", "cut_rate_per_s", "duration_s"]], on="k"))
    c["viral"] = (c.video_condition == "viral").astype(int)
    for cov in ["n_cuts", "cut_rate_per_s", "duration_s"]:
        X = sm.add_constant(c[["viral", cov]].astype(float))
        mo = sm.OLS(c.rating_liking, X).fit()
        for term in ("viral", cov):
            rows.append(dict(part="A3", analysis=f"clip OLS liking ~ viral + {cov}",
                             unit="advertisement", n=len(c), term=term,
                             estimate=mo.params[term], se=mo.bse[term], p=mo.pvalues[term],
                             ci_low=mo.conf_int().loc[term, 0], ci_high=mo.conf_int().loc[term, 1],
                             r2=mo.rsquared))
        print(f"A3  OLS liking ~ viral + {cov:15s}: viral b={mo.params['viral']:+.6f} "
              f"p={mo.pvalues['viral']:.6f} | {cov} b={mo.params[cov]:+.6f} "
              f"p={mo.pvalues[cov]:.6f} | R2={mo.rsquared:.6f} n={len(c)}")

    # Why A2's inference fails: how much does the adjusted difference actually vary?
    cl = (d.groupby(["k", "video_condition"]).rating_liking.mean().reset_index()
            .merge(p[["k", "n_cuts"]], on="k"))
    res = sm.OLS(cl.rating_liking, sm.add_constant(cl[["n_cuts"]].astype(float))).fit()
    cl["resid"] = res.resid
    dd = d.merge(cl[["k", "resid"]], on="k")
    pv = dd.groupby(["participant", "video_condition"]).resid.mean().unstack()
    adj = pv.viral - pv.flop
    n_rated = d.groupby("participant").size()
    print(f"\n    diagnostic: adjusted differences range {adj.min():+.4f} to {adj.max():+.4f}; "
          f"{(n_rated == 26).sum()} of {len(n_rated)} participants rated all 26 clips, "
          f"so their adjusted difference is identical by construction "
          f"({adj[n_rated[n_rated == 26].index].nunique()} distinct value(s) among them)")


# ======================================================== PART B (sensitivity)
def part_b(d: pd.DataFrame, rows: list) -> None:
    import statsmodels.formula.api as smf
    import warnings
    warnings.filterwarnings("ignore")

    df = d.copy()
    df["viral"] = (df.video_condition == "viral").astype(int)
    df["one"] = 1
    n_p, n_c, n_obs = df.participant.nunique(), df.k.nunique(), len(df)
    print(f"\nB   mixed model: {n_obs} ratings, {n_p} participants, {n_c} advertisements, "
          f"crossed random intercepts")

    fits = {}
    for label, formula in [("B1", "rating_liking ~ viral"),
                           ("B2", "rating_liking ~ viral + n_cuts")]:
        md = smf.mixedlm(formula, df, groups="one",
                         vc_formula={"participant": "0 + C(participant)",
                                     "advertisement": "0 + C(k)"})
        fit = md.fit(reml=True, method=["lbfgs", "powell"], maxiter=2000)
        fits[label] = fit
        for term in fit.fe_params.index:
            if term == "Intercept":
                continue
            ci = fit.conf_int().loc[term]
            rows.append(dict(part=label, analysis=f"mixed model {formula}",
                             unit="rating (crossed RE)", n=n_obs, term=term,
                             estimate=fit.fe_params[term], se=fit.bse_fe[term],
                             p=fit.pvalues[term], ci_low=ci[0], ci_high=ci[1],
                             converged=fit.converged))
        vc = fit.vcomp
        rows.append(dict(part=label, analysis=f"mixed model {formula} variance components",
                         unit="rating (crossed RE)", n=n_obs,
                         var_participant=float(vc[0]), var_advertisement=float(vc[1]),
                         var_residual=float(fit.scale), converged=fit.converged))
        print(f"{label}  {formula:34s} converged={fit.converged}")
        for term in fit.fe_params.index:
            if term != "Intercept":
                ci = fit.conf_int().loc[term]
                print(f"      {term:8s} b={fit.fe_params[term]:+.4f}  SE={fit.bse_fe[term]:.4f}  "
                      f"95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  p={fit.pvalues[term]:.4f}")
        print(f"      var: participant {vc[0]:.4f}  advertisement {vc[1]:.4f}  residual {fit.scale:.4f}")

    b1, b2 = fits["B1"].fe_params["viral"], fits["B2"].fe_params["viral"]
    reduction = 100 * (1 - b2 / b1)
    rows.append(dict(part="B", analysis="change in viral coefficient when n_cuts is added",
                     unit="rating (crossed RE)", estimate_b1=b1, estimate_b2=b2,
                     pct_retained=100 * b2 / b1, pct_removed=reduction))
    print(f"\nB   viral coefficient {b1:+.4f} (B1) -> {b2:+.4f} (B2): "
          f"{100*b2/b1:.1f}% retained, {reduction:.1f}% removed")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("real_data_analysis_output"))
    args = ap.parse_args()
    d, p = load(args.data)
    print(f"inputs: {len(d)} ratings, {d.participant.nunique()} participants, "
          f"{d.k.nunique()} advertisements\n")
    rows: list = []
    part_a(d, p, rows)
    part_b(d, rows)
    out = pd.DataFrame(rows)
    import numpy, pandas, scipy, statsmodels
    out["python"] = platform.python_version()
    out["numpy"], out["pandas"] = numpy.__version__, pandas.__version__
    out["scipy"], out["statsmodels"] = scipy.__version__, statsmodels.__version__
    dest = args.data / "liking_cut_adjustment_results.csv"
    out.to_csv(dest, index=False)
    print(f"\nwritten -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
