#!/usr/bin/env python
"""H5: do pre-video motor and inhibition traits relate to gaze and pupil during video?

Includes a direct replication test of the one pilot result that survived
correction: mouse trajectory entropy against saccade-amplitude variability
(pilot rho = -0.85, FDR-significant, N = 13).

Two things are done properly here that the quick first pass did not do:
  * trajectory metrics are computed from the TARGETING BLOCK ONLY, using the
    mouse_trial_start / mouse_trial_end markers, rather than the whole session.
    The whole-session version partly measures how much someone fidgets during
    videos, which is not a pre-video trait.
  * saccade amplitude is derived from consecutive fixation positions during the
    video epochs, matching the pilot's outcome variable.

With nine participants a correlation needs |rho| of roughly 0.67 to reach
p < .05, so this is a test of large effects only. Everything here is
exploratory and is reported with that caveat.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

GAZE_STREAM = "GazepointEyeTracker"
MOUSE_STREAM = "PsychoPyStream"
MARKER_STREAM = "PsychoPyMarkers"


def fields(text):
    parts = text.split(",")
    out = {"_tag": parts[0]}
    for p in parts[1:]:
        if ":" in p:
            k, v = p.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def load(xdf_path: Path):
    import pyxdf
    streams, _ = pyxdf.load_xdf(str(xdf_path), dejitter_timestamps=False, verbose=False)
    got = {}
    for s in streams:
        got[s["info"]["name"][0]] = s
    if MARKER_STREAM not in got:
        return None

    ev = [(float(t), fields(v[0]))
          for t, v in zip(got[MARKER_STREAM]["time_stamps"], got[MARKER_STREAM]["time_series"])]

    out = {"events": ev}

    if MOUSE_STREAM in got:
        m = got[MOUSE_STREAM]
        desc = m["info"]["desc"][0]
        labels = [c["label"][0] for c in desc["channels"][0]["channel"]]
        d = np.asarray(m["time_series"], float)
        out["mouse_ts"] = np.asarray(m["time_stamps"], float)
        out["mouse_x"] = d[:, labels.index("x")] if "x" in labels else None
        out["mouse_y"] = d[:, labels.index("y")] if "y" in labels else None

    if GAZE_STREAM in got:
        g = got[GAZE_STREAM]
        desc = g["info"]["desc"][0]
        labels = [c["label"][0] for c in desc["channels"][0]["channel"]]
        d = np.asarray(g["time_series"], float)
        out["gaze_ts"] = np.asarray(g["time_stamps"], float)
        col = lambda n: d[:, labels.index(n)] if n in labels else None
        out["fx"], out["fy"] = col("FPOGX"), col("FPOGY")
        out["fid"], out["fv"] = col("FPOGID"), col("FPOGV")
    return out


def targeting_window(events):
    """Start/end of the analysed targeting block (excludes practice)."""
    starts = [t for t, f in events if f["_tag"] == "mouse_trial_start"]
    ends = [t for t, f in events if f["_tag"] == "mouse_trial_end"]
    if not starts or not ends:
        return None
    return min(starts), max(ends)


def video_epochs(events):
    st, en, cond = {}, {}, {}
    for t, f in events:
        vid = f.get("id")
        if not vid:
            continue
        if f["_tag"] == "video_start":
            st[vid] = t
            cond[vid] = f.get("condition", "").lower()
        elif f["_tag"] == "video_end":
            en[vid] = t
    return [(v, cond.get(v, ""), st[v], en[v]) for v in sorted(set(st) & set(en))]


def traj_metrics(ts, x, y, t0, t1):
    m = (ts >= t0) & (ts <= t1)
    if m.sum() < 50:
        return {}
    xs, ys, tt = x[m], y[m], ts[m]
    ok = np.isfinite(xs) & np.isfinite(ys)
    xs, ys, tt = xs[ok], ys[ok], tt[ok]
    if len(xs) < 50:
        return {}
    h, _, _ = np.histogram2d(xs, ys, bins=20)
    p = h.ravel()[h.ravel() > 0]
    p = p / p.sum()
    ent = float(-(p * np.log(p)).sum())
    dx, dy, dt = np.diff(xs), np.diff(ys), np.diff(tt)
    step = np.hypot(dx, dy)
    dur = tt[-1] - tt[0]
    sign = np.sign(dx)
    sign = sign[sign != 0]
    xflips = int((np.diff(sign) != 0).sum()) if len(sign) > 1 else 0
    good = dt > 0
    return {
        "entropy_targeting": ent,
        "path_len_targeting": float(step.sum()),
        # rate, not count: the mouse stream rate changes between participants
        "xflip_rate_targeting": xflips / dur if dur > 0 else np.nan,
        "mean_speed_targeting": float(np.mean(step[good] / dt[good])) if good.any() else np.nan,
    }


def saccade_amplitudes(sess, t0, t1):
    """Distance between consecutive fixation centroids within a video epoch."""
    if sess.get("fid") is None:
        return np.array([])
    ts = sess["gaze_ts"]
    m = (ts >= t0) & (ts <= t1)
    if sess.get("fv") is not None:
        m &= sess["fv"] > 0
    if m.sum() < 20:
        return np.array([])
    fid, fx, fy = sess["fid"][m], sess["fx"][m], sess["fy"][m]
    cx, cy = [], []
    for u in pd.unique(fid):
        sel = fid == u
        if sel.sum() >= 2:
            cx.append(np.nanmean(fx[sel]))
            cy.append(np.nanmean(fy[sel]))
    if len(cx) < 3:
        return np.array([])
    cx, cy = np.asarray(cx), np.asarray(cy)
    return np.hypot(np.diff(cx), np.diff(cy))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for f in sorted(args.data_dir.glob("sub-*/ses-*/*.xdf")):
        pid = f.name.split("_")[0]
        sess = load(f)
        if not sess:
            continue
        r = {"participant": pid}

        tw = targeting_window(sess["events"])
        if tw and sess.get("mouse_x") is not None:
            r.update(traj_metrics(sess["mouse_ts"], sess["mouse_x"], sess["mouse_y"], *tw))

        amps_all, amps_by_cond = [], {"viral": [], "flop": []}
        for vid, cond, t0, t1 in video_epochs(sess["events"]):
            a = saccade_amplitudes(sess, t0, t1)
            if a.size:
                amps_all.append(a)
                if cond in amps_by_cond:
                    amps_by_cond[cond].append(a)
        if amps_all:
            allamp = np.concatenate(amps_all)
            r["sacc_amp_mean"] = float(np.mean(allamp))
            r["sacc_amp_sd"] = float(np.std(allamp, ddof=1))
            r["sacc_amp_cv"] = float(np.std(allamp, ddof=1) / np.mean(allamp))
            # Robust dispersion. The SD is destroyed by a handful of spurious
            # very large 'saccades' produced by tracking dropout (one participant
            # shows an SD an order of magnitude above the rest while their MEAN
            # amplitude is normal). The IQR and MAD are the honest versions of
            # the pilot's 'saccade amplitude variability'.
            q75, q25 = np.percentile(allamp, [75, 25])
            med = float(np.median(allamp))
            r["sacc_amp_median"] = med
            r["sacc_amp_iqr"] = float(q75 - q25)
            r["sacc_amp_mad"] = float(np.median(np.abs(allamp - med)))
            r["sacc_amp_rcv"] = float((q75 - q25) / med) if med > 0 else np.nan
            r["sacc_n"] = int(allamp.size)
        rows.append(r)

    traits = pd.DataFrame(rows).set_index("participant")

    # bring in the behavioural traits and the gaze outcomes already computed
    beh = pd.read_csv(args.out / "rt_by_task_participant.csv", index_col=0) \
        if (args.out / "rt_by_task_participant.csv").exists() else None
    if beh is not None:
        traits["targeting_rt"] = beh["targeting"]
        traits["gonogo_go_rt"] = beh["gonogo_GO"]
        traits["nogo_decision_time"] = beh["gonogo_NOGO"]
        traits["inhibition_cost"] = beh["GO - targeting"]

    iscf = args.out / "gaze_pupil_isc_per_participant.csv"
    if iscf.exists():
        isc = pd.read_csv(iscf)
        isc = isc[isc.participant != "__pairwise__"]
        for ch in ("gaze_x", "gaze_y", "pupil"):
            traits[f"isc_{ch}"] = isc[isc.channel == ch].groupby("participant")["isc_loo"].mean()

    metf = args.out / "per_participant_video_metrics.csv"
    if metf.exists():
        met = pd.read_csv(metf)
        met["fix_rate"] = met["n_fixations"] / met["duration_s"]
        g = met.groupby("participant")
        traits["fix_rate"] = g["fix_rate"].mean()
        traits["mean_fix_dur"] = g["mean_fix_dur"].mean()
        traits["gaze_dispersion"] = g["gaze_dispersion"].mean()
        traits["pupil_mean_bc"] = g["pupil_mean_bc"].mean()

    traits.to_csv(args.out / "h5_traits_and_outcomes.csv")

    print("=" * 90)
    print("H5  pre-video motor/inhibition traits vs gaze and pupil during video")
    print("=" * 90)
    print(f"N = {len(traits)}   (|rho| ~ 0.67 needed for p < .05)")
    print()
    show = [c for c in ("targeting_rt", "inhibition_cost", "nogo_decision_time",
                        "entropy_targeting", "xflip_rate_targeting",
                        "sacc_amp_sd", "sacc_amp_cv", "isc_gaze_y", "fix_rate")
            if c in traits.columns]
    print(traits[show].to_string(float_format=lambda v: f"{v:9.4f}"))
    print()

    TRAITS = [c for c in ("targeting_rt", "inhibition_cost", "nogo_decision_time",
                          "entropy_targeting", "xflip_rate_targeting") if c in traits.columns]
    OUTCOMES = [c for c in ("sacc_amp_sd", "sacc_amp_cv", "sacc_amp_iqr", "sacc_amp_mad", "sacc_amp_rcv", "isc_gaze_x", "isc_gaze_y",
                            "isc_pupil", "fix_rate", "mean_fix_dur", "gaze_dispersion",
                            "pupil_mean_bc") if c in traits.columns]

    res = []
    for t in TRAITS:
        for o in OUTCOMES:
            s = traits[[t, o]].dropna()
            if len(s) < 5:
                continue
            rho, p = stats.spearmanr(s[t], s[o])
            res.append({"trait": t, "outcome": o, "n": len(s),
                        "rho": float(rho), "p": float(p)})
    rd = pd.DataFrame(res)
    if not rd.empty:
        # BH across the whole matrix
        p = rd["p"].values
        order = np.argsort(p)
        adj = p[order] * len(p) / (np.arange(len(p)) + 1)
        adj = np.clip(np.minimum.accumulate(adj[::-1])[::-1], 0, 1)
        q = np.empty(len(p)); q[order] = adj
        rd["p_fdr"] = q
        rd = rd.sort_values("p")
        print("--- all trait x outcome correlations, sorted by p ---")
        print(rd.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
        rd.to_csv(args.out / "h5_correlations.csv", index=False)
        print()
        sig = rd[rd["p"] < 0.05]
        print(f"uncorrected p < .05: {len(sig)} of {len(rd)}   "
              f"(expected by chance: {0.05*len(rd):.1f})")
        print(f"surviving FDR      : {(rd['p_fdr'] < 0.05).sum()}")

        # Jackknife every nominally-significant correlation. With N=9 a single
        # participant can manufacture or destroy a correlation, so a result that
        # does not survive all leave-one-out refits should not be reported.
        if len(sig):
            print()
            print("--- leave-one-participant-out jackknife of every p < .05 result ---")
            jk_rows = []
            for _, row in sig.iterrows():
                s = traits[[row["trait"], row["outcome"]]].dropna()
                rhos, ps = [], []
                for pid in s.index:
                    r_, p_ = stats.spearmanr(s.drop(pid).iloc[:, 0], s.drop(pid).iloc[:, 1])
                    rhos.append(r_); ps.append(p_)
                n_hold = int(np.sum(np.asarray(ps) < 0.05))
                verdict = "ROBUST" if n_hold == len(ps) else f"fails {len(ps)-n_hold}/{len(ps)}"
                jk_rows.append({"trait": row["trait"], "outcome": row["outcome"],
                                "rho": row["rho"], "p": row["p"],
                                "rho_min": float(np.min(rhos)), "rho_max": float(np.max(rhos)),
                                "n_refits_p<.05": n_hold, "verdict": verdict})
            jk = pd.DataFrame(jk_rows)
            print(jk.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
            jk.to_csv(args.out / "h5_jackknife.csv", index=False)
            print()
            print("Report only rows marked ROBUST, and even those as exploratory.")
    print()

    print("--- PILOT REPLICATION: trajectory entropy vs saccade-amplitude variability ---")
    print("    pilot: rho = -0.846, FDR-significant, N = 13")
    for out_col in ("sacc_amp_sd", "sacc_amp_cv", "sacc_amp_iqr", "sacc_amp_mad"):
        if "entropy_targeting" in traits and out_col in traits:
            s = traits[["entropy_targeting", out_col]].dropna()
            if len(s) >= 5:
                rho, p = stats.spearmanr(s["entropy_targeting"], s[out_col])
                print(f"    here : entropy vs {out_col:12s} rho = {rho:+.3f}  p = {p:.4f}  N = {len(s)}")
    print()
    print(f"outputs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
