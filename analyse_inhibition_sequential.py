#!/usr/bin/env python
"""Does trial history matter for the inhibition measure?

The inhibition cost used so far is a block-level contrast:

    inhibition_cost = mean(all Go/No-Go go trials) - mean(all targeting trials)

That pools every go trial regardless of what preceded it. Two different things
are mixed together in it:

  tonic    the sustained cost of working in a task where a no-go might appear,
           so every response has to wait on a luminance check
  phasic   post-no-go slowing, the well-documented tendency to respond more
           slowly on the trial immediately after successfully withholding

This splits them:

    go_after_go        go trials whose predecessor was also a go trial
    go_after_nogo      go trials whose predecessor was a no-go trial
    post_nogo_slowing  go_after_nogo - go_after_go          (phasic)
    tonic_cost         go_after_go - targeting              (tonic, uncontaminated)

and re-runs the gaze correlations against each, so it is clear which component
the saccade result actually rests on.

The first trial of the block has no predecessor and is dropped. Trials following
an incorrect no-go are reported separately, since an error carries its own
slowing.

Usage:
    python analyse_inhibition_sequential.py --data-dir <dir> --out <dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from scipy import stats

    rows = []
    for f in sorted(args.data_dir.glob("sub-*/ses-*/*.csv")):
        pid = f.name.split("_")[0]
        d = pd.read_csv(f, encoding="utf-8-sig", low_memory=False)
        if "gonogo_luminance" not in d.columns:
            continue
        g = d[["gonogo_trial", "gonogo_luminance", "gonogo_rt",
               "gonogo_correct", "gonogo_clicked"]].dropna(subset=["gonogo_luminance"])
        if g.empty:
            continue
        g = g.sort_values("gonogo_trial").reset_index(drop=True)
        g["is_nogo"] = g.gonogo_luminance.astype(str).str.lower().eq("white")
        g["prev_nogo"] = g.is_nogo.shift(1)
        g["prev_correct"] = g.gonogo_correct.shift(1)
        g["participant"] = pid
        g["trial_idx"] = np.arange(len(g))
        rows.append(g)

    if not rows:
        print("no Go/No-Go trials found", file=sys.stderr)
        return 1
    T = pd.concat(rows, ignore_index=True)
    T.to_csv(args.out / "gonogo_trials_long.csv", index=False)

    n_nogo = T.groupby("participant").is_nogo.sum()
    print(f"participants={T.participant.nunique()}  trials/participant={len(T)//T.participant.nunique()}"
          f"  no-go per participant: {n_nogo.min()}-{n_nogo.max()}")

    go = T[(~T.is_nogo) & T.prev_nogo.notna()].copy()
    go["after"] = np.where(go.prev_nogo, "after_nogo", "after_go")
    counts = go.groupby(["participant", "after"]).size().unstack(fill_value=0)
    print(f"\ngo trials per participant  after_go: {counts.get('after_go', pd.Series()).median():.0f} "
          f"(median)   after_nogo: {counts.get('after_nogo', pd.Series()).median():.0f}")

    piv = go.groupby(["participant", "after"]).gonogo_rt.mean().unstack()
    beh = pd.read_csv(args.out / "rt_by_task_participant.csv", index_col=0)
    piv = piv.join(beh[["targeting", "gonogo_GO", "GO - targeting"]])
    piv = piv.rename(columns={"GO - targeting": "inhibition_cost_pooled"})
    piv["post_nogo_slowing"] = piv.after_nogo - piv.after_go
    piv["tonic_cost"] = piv.after_go - piv.targeting
    piv.to_csv(args.out / "inhibition_components.csv")

    print("\n" + "=" * 74)
    print("SPLITTING THE INHIBITION COST BY WHAT CAME BEFORE")
    print("=" * 74)
    d = piv.dropna(subset=["after_go", "after_nogo"])
    print(f"n = {len(d)}")
    print(f"  targeting block RT          {d.targeting.mean():.3f} s")
    print(f"  go trial after a go trial   {d.after_go.mean():.3f} s")
    print(f"  go trial after a no-go      {d.after_nogo.mean():.3f} s")
    w, p = stats.wilcoxon(d.after_nogo, d.after_go)
    dd = d.after_nogo - d.after_go
    print(f"\n  POST-NO-GO SLOWING  {dd.mean()*1000:+.0f} ms  "
          f"dz={dd.mean()/dd.std(ddof=1):+.2f}  p={p:.4f}  "
          f"({int((dd>0).sum())}/{len(dd)} slower after a no-go)")
    w2, p2 = stats.wilcoxon(d.after_go, d.targeting)
    dd2 = d.after_go - d.targeting
    print(f"  TONIC COST          {dd2.mean()*1000:+.0f} ms  "
          f"dz={dd2.mean()/dd2.std(ddof=1):+.2f}  p={p2:.4f}")
    print(f"  pooled cost (what was used) {d.inhibition_cost_pooled.mean()*1000:+.0f} ms")
    r, pr = stats.spearmanr(d.tonic_cost, d.inhibition_cost_pooled)
    print(f"\n  tonic vs pooled     rho={r:+.3f} p={pr:.4f}")
    r, pr = stats.spearmanr(d.post_nogo_slowing, d.inhibition_cost_pooled)
    print(f"  post-no-go vs pooled rho={r:+.3f} p={pr:.4f}")
    r, pr = stats.spearmanr(d.tonic_cost, d.post_nogo_slowing)
    print(f"  tonic vs post-no-go  rho={r:+.3f} p={pr:.4f}   "
          f"{'(independent)' if pr > .05 else '(related)'}")

    # ---------------------------------------------------- against gaze outcomes
    h = pd.read_csv(args.out / "h5_traits_and_outcomes.csv")
    m = h.merge(d.reset_index()[["participant", "tonic_cost", "post_nogo_slowing",
                                 "after_go", "after_nogo"]], on="participant")
    outs = [c for c in ["sacc_amp_iqr", "sacc_amp_mad", "sacc_amp_cv", "sacc_amp_median",
                        "isc_gaze_y", "isc_gaze_x", "gaze_dispersion", "fix_rate"]
            if c in m.columns]
    print("\n" + "=" * 74)
    print("WHICH COMPONENT CARRIES THE SACCADE RESULT?")
    print("=" * 74)
    hdr = f"{'outcome':18s}" + "".join(f"{t:>22s}" for t in
                                       ["pooled cost", "tonic only", "post-no-go slowing"])
    print(hdr); print("-" * len(hdr))
    res = []
    for o in outs:
        line = f"{o:18s}"
        for trait in ["inhibition_cost", "tonic_cost", "post_nogo_slowing"]:
            sub = m.dropna(subset=[trait, o])
            if len(sub) < 8:
                line += f"{'n/a':>22s}"
                continue
            r, p = stats.spearmanr(sub[trait], sub[o])
            line += f"{f'{r:+.3f} (p={p:.3f})':>22s}"
            res.append({"outcome": o, "trait": trait, "n": len(sub), "rho": r, "p": p})
        print(line)
    pd.DataFrame(res).to_csv(args.out / "inhibition_components_vs_gaze.csv", index=False)

    print("\nAlso worth knowing: are no-go trials evenly spaced, or do runs occur?")
    runs = T.groupby("participant").apply(
        lambda g: int((g.is_nogo & g.is_nogo.shift(1)).sum()), include_groups=False)
    print(f"  consecutive no-go pairs per participant: median {runs.median():.0f} "
          f"(range {runs.min()}-{runs.max()})")
    err = T[T.is_nogo & (T.gonogo_correct == 0)]
    print(f"  incorrect no-go trials in the whole dataset: {len(err)} "
          f"of {int(T.is_nogo.sum())}")
    print(f"\noutputs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
