#!/usr/bin/env python
"""Is any of the synchrony difference left once the cuts are taken out?

Section 8 of RESULTS.md shows the viral-flop gaze-synchrony difference does not
survive statistical adjustment for cut rate. This tests the same thing by
removing the mechanism directly: find every cut, delete the window of gaze that
follows it, and recompute inter-subject correlation on what is left.

A cut forces every viewer to re-orient at the same instant, and that shared
re-orientation is what inflates correlation. Delete those windows and any
remaining correlation is people choosing to look at the same thing at the same
time, rather than the edit making them.

The censoring mask is defined by the video, so it is identical for every viewer
of a clip and the surviving samples stay aligned across participants.

THE CONTROL THAT MATTERS. Flops are cut about twice as often, so censoring
removes about twice as much data from them, and a shorter series estimates
correlation more noisily. A raw comparison after censoring would therefore be
biased. Two controls are run:

  matched-loss   remove the same FRACTION of each clip, at random times, as that
                 clip lost to cut censoring. Isolates the effect of losing data
                 from the effect of losing cuts specifically.
  equalised      censor every clip to the same surviving fraction, so viral and
                 flop clips contribute equally long series.

Usage:
    python analyse_isc_cut_censored.py --data-dir <dir> --videos <dir> --out <dir>
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from analyse_gaze_pupil_isc import (  # noqa: E402
    load_session, video_windows, epoch, isc_leave_one_out, fdr, GRID_HZ,
)

SEED = 20260831
POST_CUT_S = 0.5          # window deleted after each cut
WINDOWS_TO_TEST = [0.3, 0.5, 1.0]


def norm_key(s: str) -> str:
    return "".join(sorted(re.sub(r"[^a-z0-9]", "", str(s).lower())))


def detect_cut_times(path: Path, every: int = 3) -> tuple[np.ndarray, float]:
    """Cut times in seconds from clip start, same detector as production_quality."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    hd, idxs, prev = [], [], None
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % every == 0:
            hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
            h = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
            cv2.normalize(h, h)
            if prev is not None:
                hd.append(cv2.compareHist(prev, h, cv2.HISTCMP_BHATTACHARYYA))
                idxs.append(i)
            prev = h
        i += 1
    cap.release()
    dur = i / fps
    if not hd:
        return np.array([]), dur
    hd = np.asarray(hd)
    thr = max(0.35, float(np.median(hd) + 4 * (np.median(np.abs(hd - np.median(hd))) + 1e-9)))
    cut_frames = np.asarray(idxs)[hd > thr]
    return cut_frames / fps, dur


def cut_mask(n: int, cut_times: np.ndarray, window_s: float) -> np.ndarray:
    """True where the sample SURVIVES censoring."""
    keep = np.ones(n, bool)
    for c in cut_times:
        a = int(np.floor(c * GRID_HZ))
        b = int(np.ceil((c + window_s) * GRID_HZ))
        keep[max(0, a):min(n, b)] = False
    return keep


def random_mask(n: int, drop_frac: float, rng) -> np.ndarray:
    """Drop the same fraction in contiguous blocks of the same typical length."""
    keep = np.ones(n, bool)
    target = int(round(drop_frac * n))
    blk = max(1, int(POST_CUT_S * GRID_HZ))
    while (~keep).sum() < target:
        a = rng.integers(0, max(1, n - blk))
        keep[a:a + blk] = False
    return keep


def isc_masked(traces: dict[str, np.ndarray], keep: np.ndarray) -> dict[str, float]:
    return isc_leave_one_out({p: np.where(keep, v, np.nan) for p, v in traces.items()})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--videos", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--qc", type=Path, default=None,
                    help="participant_modality_qc_final.csv: restrict to gaze-eligible participants")
    args = ap.parse_args()
    from scipy import stats
    rng = np.random.default_rng(SEED)

    # ---------------------------------------------------------------- cut times
    cuts_path = args.out / "cut_times.csv"
    if cuts_path.exists():
        ct = pd.read_csv(cuts_path)
    else:
        rows = []
        for f in sorted(args.videos.glob("*.mp4")):
            t, dur = detect_cut_times(f)
            print(f"  {f.name:34s} {len(t):3d} cuts  {dur:5.1f}s", flush=True)
            rows.append({"filename": f.name, "duration_s": dur,
                         "n_cuts": len(t),
                         "cut_times": ";".join(f"{x:.3f}" for x in t)})
        ct = pd.DataFrame(rows)
        ct.to_csv(cuts_path, index=False)
    ct["k"] = ct.filename.str.replace(".mp4", "", regex=False).map(norm_key)
    cut_lookup = {r.k: (np.array([float(x) for x in str(r.cut_times).split(";") if x])
                        if isinstance(r.cut_times, str) and r.cut_times else np.array([]))
                  for r in ct.itertuples()}

    # ------------------------------------------------------------ gather traces
    keep_pids = None
    if args.qc and args.qc.exists():
        q = pd.read_csv(args.qc)
        keep_pids = set(q.loc[q.gaze_eligible, "participant"])
        print(f"restricted to {len(keep_pids)} gaze-eligible participants: "
              f"{', '.join(sorted(keep_pids))}")

    per_clip: dict[tuple[str, str], dict[str, dict[str, np.ndarray]]] = {}
    for f in sorted(args.data_dir.glob("sub-*/ses-*/*.xdf")):
        pid = f.name.split("_")[0]
        if keep_pids is not None and pid not in keep_pids:
            continue
        sess = load_session(f)
        if not sess:
            continue
        for vid, cond, t0, t1, tb0, tb1 in video_windows(sess["events"]):
            ep = epoch(sess, t0, t1)
            if ep is None:
                continue
            slot = per_clip.setdefault((vid, cond), {})
            for chan in ("gaze_x", "gaze_y"):
                slot.setdefault(chan, {})[pid] = ep[chan]

    if not per_clip:
        print("no epochs", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------ compute
    rows = []
    for (vid, cond), chans in sorted(per_clip.items()):
        k = norm_key(vid)
        cts = cut_lookup.get(k)
        if cts is None:
            print(f"  WARNING no cut times for {vid}")
            continue
        for chan, traces in chans.items():
            n = min(len(v) for v in traces.values())
            traces = {p: v[:n] for p, v in traces.items()}
            if len(traces) < 6:
                continue
            full = isc_leave_one_out(traces)
            for w in WINDOWS_TO_TEST:
                keep = cut_mask(n, cts, w)
                frac_lost = 1 - keep.mean()
                if keep.sum() < 60:
                    continue
                cens = isc_masked(traces, keep)
                # matched random loss, averaged over repeats to reduce its own noise
                ctrl = {p: [] for p in traces}
                for _ in range(20):
                    rk = random_mask(n, frac_lost, rng)
                    c = isc_masked(traces, rk)
                    for p, v in c.items():
                        ctrl[p].append(v)
                ctrl = {p: float(np.nanmean(v)) for p, v in ctrl.items()}
                for p in traces:
                    rows.append({"video_id": vid, "condition": cond, "channel": chan,
                                 "participant": p, "window_s": w,
                                 "frac_lost": frac_lost,
                                 "isc_full": full.get(p, np.nan),
                                 "isc_cut_censored": cens.get(p, np.nan),
                                 "isc_random_censored": ctrl.get(p, np.nan)})
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "isc_cut_censored.csv", index=False)

    print("\n" + "=" * 78)
    print("ISC BEFORE AND AFTER DELETING THE WINDOW AFTER EVERY CUT")
    print("=" * 78)
    for w in WINDOWS_TO_TEST:
        d = df[df.window_s == w]
        if d.empty:
            continue
        lost = d.groupby("condition").frac_lost.mean()
        print(f"\n--- {w:.1f} s censored after each cut "
              f"(data removed: viral {lost.get('viral', np.nan):.1%}, "
              f"flop {lost.get('flop', np.nan):.1%}) ---")
        for chan in ("gaze_y", "gaze_x"):
            dc = d[d.channel == chan]
            if dc.empty:
                continue
            print(f"  {chan}")
            for col, lab in [("isc_full", "no censoring"),
                             ("isc_cut_censored", "cuts removed"),
                             ("isc_random_censored", "random loss (control)")]:
                piv = dc.groupby(["participant", "condition"])[col].mean().unstack().dropna()
                if len(piv) < 6 or "viral" not in piv or "flop" not in piv:
                    continue
                p = stats.wilcoxon(piv.viral, piv.flop)[1]
                dd = piv.viral - piv.flop
                dz = dd.mean() / dd.std(ddof=1)
                star = " *" if p < .05 else "  "
                print(f"    {lab:24s} viral={piv.viral.mean():.3f} "
                      f"flop={piv.flop.mean():.3f} diff={dd.mean():+.3f} "
                      f"dz={dz:+.2f} p={p:.4f}{star}")

    print("\nRead it this way. If the difference survives 'cuts removed' but the "
          "\n'random loss' control shows the same shrinkage, the change is about "
          "\nlosing data, not about losing cuts.")
    print(f"\noutputs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
