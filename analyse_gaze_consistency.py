#!/usr/bin/env python
"""How consistently do viewers look at the same place, moment by moment?

The inter-subject correlation in the main analysis is one number per clip. It
says whether people moved their eyes in step, but not how tightly clustered they
were, not when during the clip they agreed, and not what agreement looks like in
units anyone can picture.

This produces the descriptive version, which is worth reporting whether or not
anything reaches significance:

  agreement_radius     the median distance from each viewer to the group centre
                       at each moment, in screen-width units. Small = everyone
                       looking at the same spot.
  pct_within_10        the percentage of the clip during which at least three
                       quarters of viewers were within 10% of screen width of
                       the group centre. A plain-language measure of consensus.
  time course          agreement as a function of time through the clip, so a
                       clip can be described as starting together and drifting
                       apart, or the reverse.
  onset                how long after the clip starts before viewers converge.

Gaze is resampled onto a common time grid per clip so that participants with
different sampling phases are comparable.

Usage:
    python analyse_gaze_consistency.py --data-dir <dir> --out real_data_analysis_output
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from analyse_h5_traits import load, video_epochs  # noqa: E402

GRID_HZ = 30.0          # common time grid
MIN_VIEWERS = 6         # a moment needs this many valid viewers to be scored
CONSENSUS_FRAC = 0.75   # "most viewers" = this fraction
CONSENSUS_RADIUS = 0.10  # within 10% of screen width of the group centre


def resample_epoch(ts, x, y, valid, t0, t1):
    """Gaze on a common grid for one participant-clip, NaN where invalid."""
    n = int(round((t1 - t0) * GRID_HZ))
    if n < 10:
        return None, None, None
    grid = t0 + np.arange(n) / GRID_HZ
    m = (ts >= t0) & (ts <= t1)
    if m.sum() < 10:
        return None, None, None
    tt, xx, yy = ts[m], x[m], y[m]
    vv = valid[m] if valid is not None else np.ones(m.sum(), bool)
    ok = vv.astype(bool) & np.isfinite(xx) & np.isfinite(yy)
    if ok.sum() < 10:
        return None, None, None
    gx = np.interp(grid, tt[ok], xx[ok], left=np.nan, right=np.nan)
    gy = np.interp(grid, tt[ok], yy[ok], left=np.nan, right=np.nan)
    # blank the grid points that fall in a long gap rather than interpolating over it
    gaps = np.searchsorted(tt[ok], grid)
    gaps = np.clip(gaps, 1, len(tt[ok]) - 1)
    near = np.minimum(np.abs(grid - tt[ok][gaps - 1]), np.abs(tt[ok][gaps] - grid))
    bad = near > 0.2
    gx[bad] = np.nan
    gy[bad] = np.nan
    return grid, gx, gy


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from scipy import stats

    # -------------------------------------------------- collect resampled gaze
    store: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for f in sorted(args.data_dir.glob("sub-*/ses-*/*.xdf")):
        pid = f.name.split("_")[0]
        sess = load(f)
        if not sess or sess.get("fx") is None:
            continue
        ts = sess["gaze_ts"]
        valid = sess.get("fv")          # FPOGV, the tracker's own validity flag
        for vid, cond, t0, t1 in video_epochs(sess["events"]):
            g, gx, gy = resample_epoch(ts, sess["fx"], sess["fy"], valid, t0, t1)
            if g is None:
                continue
            store.setdefault((vid, cond), {})[pid] = np.column_stack([gx, gy])

    if not store:
        print("no epochs found", file=sys.stderr)
        return 1

    rows, tc_rows = [], []
    for (vid, cond), byp in sorted(store.items()):
        n = min(len(v) for v in byp.values())
        M = np.stack([v[:n] for v in byp.values()])       # viewers x time x 2
        viewers = M.shape[0]
        if viewers < MIN_VIEWERS:
            continue
        # group centre at each moment, ignoring missing viewers
        centre = np.nanmedian(M, axis=0)                   # time x 2
        d = np.sqrt(np.nansum((M - centre) ** 2, axis=2))  # viewers x time
        d[~np.isfinite(M).all(axis=2)] = np.nan
        n_valid = np.isfinite(d).sum(axis=0)
        scoreable = n_valid >= MIN_VIEWERS

        med_r = np.nanmedian(d, axis=0)
        med_r[~scoreable] = np.nan
        within = np.nanmean(d <= CONSENSUS_RADIUS, axis=0)
        within[~scoreable] = np.nan
        consensus = np.nanmean(within >= CONSENSUS_FRAC) * 100

        t_rel = np.arange(n) / GRID_HZ
        # time to first consensus moment
        idx = np.flatnonzero(np.nan_to_num(within) >= CONSENSUS_FRAC)
        onset = t_rel[idx[0]] if idx.size else np.nan

        rows.append({
            "video_id": vid, "condition": cond, "n_viewers": viewers,
            "dur_s": n / GRID_HZ,
            "agreement_radius": float(np.nanmedian(med_r)),
            "agreement_radius_first3s": float(np.nanmedian(med_r[t_rel <= 3])),
            "agreement_radius_last3s": float(np.nanmedian(med_r[t_rel >= t_rel[-1] - 3])),
            "pct_time_consensus": float(consensus),
            "consensus_onset_s": float(onset),
            "spread_over_time": float(np.nanmedian(med_r[t_rel >= t_rel[-1] - 3])
                                      - np.nanmedian(med_r[t_rel <= 3])),
        })
        for i in range(0, n, int(GRID_HZ / 2)):      # every 0.5 s
            if scoreable[i]:
                tc_rows.append({"video_id": vid, "condition": cond,
                                "t_s": t_rel[i], "t_frac": t_rel[i] / t_rel[-1],
                                "agreement_radius": med_r[i],
                                "frac_within": within[i]})

    clip = pd.DataFrame(rows)
    tc = pd.DataFrame(tc_rows)
    clip.to_csv(args.out / "gaze_consistency_per_clip.csv", index=False)
    tc.to_csv(args.out / "gaze_consistency_timecourse.csv", index=False)

    print(f"clips scored: {len(clip)}   "
          f"viral={(clip.condition=='viral').sum()} flop={(clip.condition=='flop').sum()}")
    print("\nAll units are fractions of screen width. Smaller radius = viewers "
          "looking at more nearly the same place.\n")

    print("=" * 78)
    print("HOW CONSISTENT IS GAZE, DESCRIPTIVELY")
    print("=" * 78)
    hdr = f"{'measure':28s} {'viral':>10s} {'flop':>10s} {'diff':>9s} {'p':>8s}"
    print(hdr); print("-" * len(hdr))
    for m, lab in [("agreement_radius", "agreement radius"),
                   ("pct_time_consensus", "% time in consensus"),
                   ("consensus_onset_s", "time to consensus (s)"),
                   ("agreement_radius_first3s", "radius, first 3 s"),
                   ("agreement_radius_last3s", "radius, last 3 s"),
                   ("spread_over_time", "drift (last - first)")]:
        a = clip[clip.condition == "viral"][m].dropna()
        b = clip[clip.condition == "flop"][m].dropna()
        if len(a) < 3 or len(b) < 3:
            continue
        p = stats.mannwhitneyu(a, b)[1]
        star = " *" if p < .05 else "  "
        print(f"{lab:28s} {a.mean():10.4f} {b.mean():10.4f} "
              f"{a.mean()-b.mean():+9.4f} {p:8.4f}{star}")

    print("\nRange across clips (this is the descriptive point):")
    print(f"  agreement radius   {clip.agreement_radius.min():.3f} to "
          f"{clip.agreement_radius.max():.3f}  (median {clip.agreement_radius.median():.3f})")
    print(f"  % time consensus   {clip.pct_time_consensus.min():.1f}% to "
          f"{clip.pct_time_consensus.max():.1f}%  (median {clip.pct_time_consensus.median():.1f}%)")

    print("\nMost and least consistently watched clips:")
    o = clip.sort_values("agreement_radius")
    for _, r in pd.concat([o.head(4), o.tail(4)]).iterrows():
        print(f"  {r.video_id:28s} {r.condition:5s} radius={r.agreement_radius:.3f} "
              f"consensus={r.pct_time_consensus:5.1f}%  n={int(r.n_viewers)}")

    print("\nDoes agreement change over the course of a clip?")
    for cond in ["viral", "flop"]:
        sub = tc[tc.condition == cond]
        q = sub.groupby(pd.cut(sub.t_frac, [0, .25, .5, .75, 1.0]),
                        observed=True).agreement_radius.mean()
        print(f"  {cond:6s} " + "  ".join(f"{k}: {v:.3f}" for k, v in q.items()))

    print(f"\noutputs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
