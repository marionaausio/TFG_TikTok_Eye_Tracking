#!/usr/bin/env python
"""Reaction-time comparison between the two mouse tasks.

Three contrasts, all within participant:

  A. Targeting RT              vs  Go/No-Go GO RT
     Both are 'click the circle'. Task 2 adds a colour discrimination and the
     standing possibility of having to withhold, so any slowing is the cost of
     holding an inhibitory set, independent of whether errors are ever made.

  B. Go/No-Go GO RT            vs  Go/No-Go NO-GO RT
     The no-go RT is the time to redirect and click the centre cross. The
     difference is the cost of cancelling the trained response and re-aiming.

  C. Targeting RT              vs  Go/No-Go NO-GO RT
     Total cost relative to the untrained simple movement.

Zero commission errors does not mean the manipulation failed. If the inhibitory
set is doing work it shows up in RT rather than in accuracy.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "real_experiment_data"


def load(data_dir: Path):
    targ, gng = [], []
    for csv_path in sorted(data_dir.glob("sub-*/ses-*/*.csv")):
        pid = csv_path.name.split("_")[0]
        d = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)

        if {"trial", "rt"} <= set(d.columns):
            t = d[["trial", "rt"]].copy()
            t = t[pd.to_numeric(t["trial"], errors="coerce").notna()]
            t["rt"] = pd.to_numeric(t["rt"], errors="coerce")
            t = t.dropna(subset=["rt"])
            t["participant"] = pid
            targ.append(t)

        if {"gonogo_luminance", "gonogo_rt"} <= set(d.columns):
            g = d[["gonogo_luminance", "gonogo_rt", "gonogo_correct"]].copy()
            g = g.dropna(subset=["gonogo_luminance"])
            g["gonogo_rt"] = pd.to_numeric(g["gonogo_rt"], errors="coerce")
            g["gonogo_correct"] = pd.to_numeric(g["gonogo_correct"], errors="coerce")
            g = g.dropna(subset=["gonogo_rt"])
            g["trial_type"] = np.where(
                g["gonogo_luminance"].astype(str).str.lower().eq("white"), "no-go", "go")
            g["participant"] = pid
            gng.append(g)

    return (pd.concat(targ, ignore_index=True) if targ else pd.DataFrame(),
            pd.concat(gng, ignore_index=True) if gng else pd.DataFrame())


def paired(a: pd.Series, b: pd.Series, label_a: str, label_b: str):
    """a and b indexed by participant."""
    idx = a.index.intersection(b.index)
    x, y = a.loc[idx].values, b.loc[idx].values
    d = y - x
    out = {
        "contrast": f"{label_b} - {label_a}",
        "n": len(idx),
        f"mean_{label_a}": float(np.mean(x)),
        f"mean_{label_b}": float(np.mean(y)),
        "mean_diff_s": float(np.mean(d)),
        "median_diff_s": float(np.median(d)),
        "n_slower": int((d > 0).sum()),
    }
    sd = np.std(d, ddof=1)
    out["cohen_dz"] = float(np.mean(d) / sd) if sd > 0 else np.nan
    if len(idx) >= 5 and np.any(d != 0):
        w, p = stats.wilcoxon(y, x)
        out["W"] = float(w)
        out["p"] = float(p)
    return out, idx, x, y, d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--correct-only", action="store_true", default=True,
                    help="use only correct trials (default)")
    args = ap.parse_args()

    targ, gng = load(args.data_dir)
    if targ.empty or gng.empty:
        print("missing data")
        return 1

    if args.correct_only:
        gng = gng[gng["gonogo_correct"] == 1]

    t_rt = targ.groupby("participant")["rt"].mean()
    go_rt = gng[gng["trial_type"] == "go"].groupby("participant")["gonogo_rt"].mean()
    nogo_rt = gng[gng["trial_type"] == "no-go"].groupby("participant")["gonogo_rt"].mean()

    print("=" * 84)
    print("REACTION TIME: targeting task vs Go/No-Go task")
    print("=" * 84)
    print(f"correct trials only: {args.correct_only}")
    print()

    tbl = pd.DataFrame({
        "targeting": t_rt,
        "gonogo_GO": go_rt,
        "gonogo_NOGO": nogo_rt,
    })
    tbl["GO - targeting"] = tbl["gonogo_GO"] - tbl["targeting"]
    tbl["NOGO - GO"] = tbl["gonogo_NOGO"] - tbl["gonogo_GO"]
    tbl["NOGO - targeting"] = tbl["gonogo_NOGO"] - tbl["targeting"]
    print("--- per participant (seconds) ---")
    print(tbl.to_string(float_format=lambda v: f"{v:9.4f}"))
    print()

    results = []
    for a, b, la, lb in (
        (t_rt, go_rt, "targeting", "gonogo_GO"),
        (go_rt, nogo_rt, "gonogo_GO", "gonogo_NOGO"),
        (t_rt, nogo_rt, "targeting", "gonogo_NOGO"),
    ):
        res, *_ = paired(a, b, la, lb)
        results.append(res)

    print("--- paired contrasts (Wilcoxon, within participant) ---")
    rd = pd.DataFrame(results)
    cols = ["contrast", "n", "mean_diff_s", "median_diff_s", "n_slower", "cohen_dz", "W", "p"]
    cols = [c for c in cols if c in rd.columns]
    print(rd[cols].to_string(index=False, float_format=lambda v: f"{v:9.4f}"))
    print()

    # trial-level distribution check
    print("--- trial-level RT distributions (all participants pooled) ---")
    pool = pd.concat([
        targ[["rt"]].assign(task="targeting").rename(columns={"rt": "RT"}),
        gng[gng["trial_type"] == "go"][["gonogo_rt"]].assign(task="gonogo_GO").rename(columns={"gonogo_rt": "RT"}),
        gng[gng["trial_type"] == "no-go"][["gonogo_rt"]].assign(task="gonogo_NOGO").rename(columns={"gonogo_rt": "RT"}),
    ])
    print(pool.groupby("task")["RT"].describe()[["count", "mean", "std", "25%", "50%", "75%"]]
          .to_string(float_format=lambda v: f"{v:9.4f}"))

    out = args.data_dir.parent / "real_data_analysis_output"
    out.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(out / "rt_by_task_participant.csv")
    rd.to_csv(out / "rt_paired_contrasts.csv", index=False)
    print()
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
