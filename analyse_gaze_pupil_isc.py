#!/usr/bin/env python
"""Gaze and pupil inter-subject correlation, viral vs flop (H1, H2, H3).

Metric structure is taken from the earlier pilot (PsychoPy workspace,
analyse_eeg_gaze_crossmodal.py), which computed ISC per video on three
channels: gaze_x, gaze_y and pupil. This reproduces those three on the new
data and adds the improvements the pilot could not make:

  * leave-one-out ISC as the primary estimator, because the pilot's pairwise
    correlations are not independent across participants (each participant
    appears in many pairs). The pilot's pairwise mean is also reported so the
    two datasets can be compared on a like-for-like basis.
  * pupil baseline correction against the per-video grey-screen baseline,
    which did not exist in the pilot recordings.
  * a validity filter on gaze and pupil rather than treating all samples as good.

Also computes the non-ISC attention measures for H3: fixation count and
duration, gaze dispersion, and valid-gaze proportion.

Usage:
    python analyse_gaze_pupil_isc.py --data-dir <real_experiment_data> --out <dir>
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

GAZE_STREAM = "GazepointEyeTracker"
MARKER_STREAM = "PsychoPyMarkers"

# resample grid for ISC; well below the 149 Hz tracker rate, above fixation dynamics
GRID_HZ = 50.0


# ------------------------------------------------------------------ parsing

def marker_fields(text: str) -> dict:
    """'video_start,id:x,condition:viral' -> {'_tag':'video_start','id':'x',...}"""
    parts = text.split(",")
    out = {"_tag": parts[0]}
    for p in parts[1:]:
        if ":" in p:
            k, v = p.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def load_session(xdf_path: Path):
    import pyxdf
    streams, _ = pyxdf.load_xdf(str(xdf_path), dejitter_timestamps=False, verbose=False)
    gaze = markers = None
    for s in streams:
        nm = s["info"]["name"][0]
        if nm == GAZE_STREAM:
            gaze = s
        elif nm == MARKER_STREAM:
            markers = s
    if gaze is None or markers is None:
        return None

    desc = gaze["info"]["desc"][0]
    labels = [c["label"][0] for c in desc["channels"][0]["channel"]]
    data = np.asarray(gaze["time_series"], dtype=float)
    ts = np.asarray(gaze["time_stamps"], dtype=float)

    def col(name):
        return data[:, labels.index(name)] if name in labels else None

    # pupil in mm, averaged over eyes where both are valid
    lpmm, rpmm = col("LPMM"), col("RPMM")
    # Use the validity flags that correspond to the millimetre pupil channels.
    # Older files may not contain them, in which case the general per-eye flags
    # remain an explicitly documented fallback.
    lpv = col("LPMMV") if col("LPMMV") is not None else col("LPV")
    rpv = col("RPMMV") if col("RPMMV") is not None else col("RPV")
    if lpmm is not None and rpmm is not None:
        lv = np.ones_like(lpmm, bool) if lpv is None else (lpv > 0)
        rv = np.ones_like(rpmm, bool) if rpv is None else (rpv > 0)
        pupil = np.where(lv & rv, (lpmm + rpmm) / 2.0,
                         np.where(lv, lpmm, np.where(rv, rpmm, np.nan)))
        pupil_valid = lv | rv
    else:
        pupil = col("PUPILMM")
        pupil_valid = np.isfinite(pupil) if pupil is not None else None

    fpogv = col("FPOGV")
    gaze_valid = (fpogv > 0) if fpogv is not None else np.ones(len(ts), bool)

    ev = []
    for t, v in zip(markers["time_stamps"], markers["time_series"]):
        ev.append((float(t), marker_fields(v[0])))

    return {
        "ts": ts,
        "x": col("FPOGX"), "y": col("FPOGY"),
        "pupil": pupil,
        "gaze_valid": gaze_valid,
        "pupil_valid": pupil_valid if pupil_valid is not None else np.ones(len(ts), bool),
        "fix_id": col("FPOGID"), "fix_dur": col("FPOGD"),
        "blink": col("BLINK"),
        "events": ev,
    }


def video_windows(events):
    """[(video_id, condition, t_start, t_end, t_base_start, t_base_end)]"""
    starts, ends, bstart, bend = {}, {}, {}, {}
    cond = {}
    for t, f in events:
        vid = f.get("id")
        if not vid:
            continue
        tag = f["_tag"]
        if tag == "video_start":
            starts[vid] = t
            cond[vid] = f.get("condition", "").lower()
        elif tag == "video_end":
            ends[vid] = t
        elif tag == "pre_video_baseline_start":
            bstart[vid] = t
        elif tag == "pre_video_baseline_end":
            bend[vid] = t
    out = []
    for vid in sorted(set(starts) & set(ends)):
        out.append((vid, cond.get(vid, ""), starts[vid], ends[vid],
                    bstart.get(vid), bend.get(vid)))
    return out


# ------------------------------------------------------------------ per-video

def resample(t, v, grid):
    ok = np.isfinite(t) & np.isfinite(v)
    if ok.sum() < 10:
        return np.full(len(grid), np.nan)
    return np.interp(grid, t[ok], v[ok], left=np.nan, right=np.nan)


def epoch(sess, t0, t1, grid_hz=GRID_HZ):
    ts = sess["ts"]
    m = (ts >= t0) & (ts <= t1)
    if m.sum() < 20:
        return None
    rel = ts[m] - t0
    dur = t1 - t0
    grid = np.arange(0, dur, 1.0 / grid_hz)

    gv = sess["gaze_valid"][m]
    pv = sess["pupil_valid"][m]

    x = np.where(gv, sess["x"][m], np.nan)
    y = np.where(gv, sess["y"][m], np.nan)
    p = np.where(pv, sess["pupil"][m], np.nan) if sess["pupil"] is not None else None

    out = {
        "grid": grid,
        "gaze_x": resample(rel, x, grid),
        "gaze_y": resample(rel, y, grid),
        "pupil": resample(rel, p, grid) if p is not None else np.full(len(grid), np.nan),
        "valid_gaze_pct": float(np.mean(gv) * 100),
        "valid_pupil_pct": float(np.mean(pv) * 100),
        "duration_s": float(dur),
    }

    # H3 attention measures, from the raw (unresampled) samples
    if sess["fix_id"] is not None:
        fid = sess["fix_id"][m][gv]
        out["n_fixations"] = int(len(np.unique(fid))) if len(fid) else 0
    if sess["fix_dur"] is not None:
        fd = sess["fix_dur"][m][gv]
        out["mean_fix_dur"] = float(np.nanmean(fd)) if len(fd) else np.nan
    xv, yv = x[np.isfinite(x)], y[np.isfinite(y)]
    if len(xv) > 10:
        out["gaze_dispersion"] = float(np.sqrt(np.nanvar(xv) + np.nanvar(yv)))
    if sess["blink"] is not None:
        b = sess["blink"][m]
        out["blink_rate_per_min"] = float(np.sum(np.diff(b) > 0) / (dur / 60.0)) if dur > 0 else np.nan
    return out


def baseline_correct(pupil_trace, sess, tb0, tb1):
    """Subtract the median pupil during the pre-video grey screen."""
    if tb0 is None or tb1 is None:
        return pupil_trace, np.nan
    ts = sess["ts"]
    m = (ts >= tb0) & (ts <= tb1) & sess["pupil_valid"]
    if m.sum() < 5:
        return pupil_trace, np.nan
    base = float(np.nanmedian(sess["pupil"][m]))
    return pupil_trace - base, base


# ------------------------------------------------------------------ ISC

def isc_leave_one_out(traces):
    """traces: {pid: 1-D array on a common grid} -> {pid: r}"""
    pids = [p for p, v in traces.items() if np.isfinite(v).sum() > 20]
    if len(pids) < 3:
        return {}
    M = np.vstack([traces[p] for p in pids])
    out = {}
    for i, p in enumerate(pids):
        others = np.delete(M, i, axis=0)
        mean_other = np.nanmean(others, axis=0)
        a, b = M[i], mean_other
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() > 20 and np.nanstd(a[ok]) > 0 and np.nanstd(b[ok]) > 0:
            out[p] = float(np.corrcoef(a[ok], b[ok])[0, 1])
    return out


def isc_pairwise_mean(traces):
    """The pilot's estimator: mean of all pairwise correlations."""
    pids = [p for p, v in traces.items() if np.isfinite(v).sum() > 20]
    if len(pids) < 2:
        return np.nan, 0
    rs = []
    for i in range(len(pids)):
        for j in range(i + 1, len(pids)):
            a, b = traces[pids[i]], traces[pids[j]]
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() > 20 and np.nanstd(a[ok]) > 0 and np.nanstd(b[ok]) > 0:
                rs.append(np.corrcoef(a[ok], b[ok])[0, 1])
    return (float(np.mean(rs)) if rs else np.nan), len(rs)


# ------------------------------------------------------------------ stats

def paired_test(df, value_col, unit="participant"):
    piv = df.groupby([unit, "condition"])[value_col].mean().unstack("condition")
    if not {"viral", "flop"} <= set(piv.columns):
        return None
    piv = piv.dropna()
    if len(piv) < 3:
        return None
    v, f = piv["viral"].values, piv["flop"].values
    d = v - f
    res = {"n": len(v), "viral": float(np.mean(v)), "flop": float(np.mean(f)),
           "diff": float(np.mean(d)), "n_higher_viral": int((d > 0).sum())}
    sd = np.std(d, ddof=1)
    res["cohen_dz"] = float(np.mean(d) / sd) if sd > 0 else np.nan
    if len(v) >= 5 and np.any(d != 0):
        w, p = stats.wilcoxon(v, f)
        res["p"] = float(p)
    return res


def fdr(p):
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    out = np.full_like(p, np.nan)
    if ok.sum() == 0:
        return out
    q = p[ok]; order = np.argsort(q); n = len(q)
    adj = q[order] * n / (np.arange(n) + 1)
    adj = np.clip(np.minimum.accumulate(adj[::-1])[::-1], 0, 1)
    r = np.empty(n); r[order] = adj; out[ok] = r
    return out


# ------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--min-valid-gaze", type=float, default=50.0,
                    help="drop a participant-video below this %% valid gaze")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    xdfs = sorted(args.data_dir.glob("sub-*/ses-*/*.xdf"))
    sessions = {}
    for f in xdfs:
        pid = f.name.split("_")[0]
        if pid in args.exclude:
            continue
        s = load_session(f)
        if s:
            sessions[pid] = s
    if not sessions:
        print("no sessions loaded", file=sys.stderr)
        return 1

    print("=" * 84)
    print("GAZE AND PUPIL ISC - viral vs flop")
    print("=" * 84)
    print(f"participants: {len(sessions)} -> {', '.join(sorted(sessions))}")
    print(f"min valid gaze per video: {args.min_valid_gaze}%")
    print()

    # ---- epoch every participant x video
    per_video = {}   # vid -> {'condition':c, 'traces':{chan:{pid:arr}}, 'metrics':[...]}
    rows = []
    for pid, sess in sessions.items():
        for vid, cond, t0, t1, tb0, tb1 in video_windows(sess["events"]):
            if cond not in ("viral", "flop"):
                continue
            ep = epoch(sess, t0, t1)
            if ep is None:
                continue
            if ep["valid_gaze_pct"] < args.min_valid_gaze:
                continue
            pcorr, base = baseline_correct(ep["pupil"], sess, tb0, tb1)
            ep["pupil_bc"] = pcorr
            ep["pupil_baseline_mm"] = base

            slot = per_video.setdefault(vid, {"condition": cond, "traces": {
                "gaze_x": {}, "gaze_y": {}, "pupil": {}}})
            slot["traces"]["gaze_x"][pid] = ep["gaze_x"]
            slot["traces"]["gaze_y"][pid] = ep["gaze_y"]
            slot["traces"]["pupil"][pid] = ep["pupil_bc"]

            rows.append({
                "participant": pid, "video_id": vid, "condition": cond,
                "duration_s": ep["duration_s"],
                "valid_gaze_pct": ep["valid_gaze_pct"],
                "valid_pupil_pct": ep["valid_pupil_pct"],
                "n_fixations": ep.get("n_fixations", np.nan),
                "mean_fix_dur": ep.get("mean_fix_dur", np.nan),
                "gaze_dispersion": ep.get("gaze_dispersion", np.nan),
                "blink_rate_per_min": ep.get("blink_rate_per_min", np.nan),
                "pupil_baseline_mm": base,
                "pupil_mean_bc": float(np.nanmean(pcorr)),
            })

    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.out / "per_participant_video_metrics.csv", index=False)
    print(f"epochs kept: {len(metrics)}  "
          f"(mean valid gaze {metrics['valid_gaze_pct'].mean():.1f}%)")
    print()

    # ---- ISC per video
    isc_rows = []
    for vid, slot in per_video.items():
        for chan in ("gaze_x", "gaze_y", "pupil"):
            tr = slot["traces"][chan]
            # trim to shortest common length
            if len(tr) < 3:
                continue
            L = min(len(v) for v in tr.values())
            tr = {p: v[:L] for p, v in tr.items()}
            loo = isc_leave_one_out(tr)
            pw, npairs = isc_pairwise_mean(tr)
            for pid, r in loo.items():
                isc_rows.append({"video_id": vid, "condition": slot["condition"],
                                 "channel": chan, "participant": pid, "isc_loo": r})
            isc_rows.append({"video_id": vid, "condition": slot["condition"],
                             "channel": chan, "participant": "__pairwise__",
                             "isc_loo": np.nan, "isc_pairwise": pw, "n_pairs": npairs})
    isc = pd.DataFrame(isc_rows)
    isc.to_csv(args.out / "gaze_pupil_isc_per_participant.csv", index=False)

    loo = isc[isc["participant"] != "__pairwise__"]
    pw = isc[isc["participant"] == "__pairwise__"]

    print("-" * 84)
    print("H1 / H2   ISC by condition  (leave-one-out, per participant per video)")
    print("-" * 84)
    res_rows = []
    for chan in ("gaze_x", "gaze_y", "pupil"):
        sub = loo[loo["channel"] == chan]
        if sub.empty:
            continue
        r = paired_test(sub, "isc_loo")
        if r:
            r["channel"] = chan
            res_rows.append(r)
    if res_rows:
        rd = pd.DataFrame(res_rows)
        rd["p_fdr"] = fdr(rd.get("p", pd.Series([np.nan] * len(rd))).values)
        cols = [c for c in ("channel", "n", "viral", "flop", "diff",
                            "n_higher_viral", "cohen_dz", "p", "p_fdr") if c in rd.columns]
        print(rd[cols].to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
        rd.to_csv(args.out / "h1_h2_isc_condition_tests.csv", index=False)
    print()

    print("--- pilot-style pairwise ISC means, for comparison with the March pilot ---")
    if not pw.empty:
        print(pw.pivot_table(index="condition", columns="channel",
                             values="isc_pairwise", aggfunc="mean")
                .round(4).to_string())
    print()

    print("-" * 84)
    print("H3   attention measures by condition")
    print("-" * 84)
    h3 = []
    for col in ("n_fixations", "mean_fix_dur", "gaze_dispersion",
                "blink_rate_per_min", "pupil_mean_bc", "valid_gaze_pct"):
        if col not in metrics.columns or metrics[col].isna().all():
            continue
        r = paired_test(metrics, col)
        if r:
            r["measure"] = col
            h3.append(r)
    if h3:
        hd = pd.DataFrame(h3)
        hd["p_fdr"] = fdr(hd.get("p", pd.Series([np.nan] * len(hd))).values)
        cols = [c for c in ("measure", "n", "viral", "flop", "diff",
                            "n_higher_viral", "cohen_dz", "p", "p_fdr") if c in hd.columns]
        print(hd[cols].to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
        hd.to_csv(args.out / "h3_attention_condition_tests.csv", index=False)
    print()
    print(f"outputs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
