"""
Quick per-participant QC figure for the mouse-tracking block.

Reads the latest PsychoPy CSV + LabRecorder XDF for a participant and saves a
4-panel figure showing:

  (0,0) Gaze  - raw 2D trajectories vs targets (PsychoPy norm space)
  (0,1) Mouse - raw 2D trajectories vs targets (PsychoPy norm space)
  (1,0) Gaze  - each trial rotated so target direction -> +x (single-vector view)
  (1,1) Mouse - each trial rotated so target direction -> +x (single-vector view)

Coordinate frame: PsychoPy norm units ([-1, 1], centre-origin). Aspect ratio for
gazepoint-to-PsychoPy conversion defaults to 1.6 (2560x1600). Swap to cm or
degrees of visual angle later by multiplying by half-width / tan(half-FOV).

Intended to run automatically at the end of mouse_lsl_task_stable.py; can also
be run standalone:

    python quick_qc.py --participant VC-01
    python quick_qc.py --participant VC-01 --xdf path/to/file.xdf
    python quick_qc.py --participant VC-01 --wait-for-xdf
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

try:
    import pyxdf
except ImportError:
    print("[ERROR] pyxdf not installed in this environment.", file=sys.stderr)
    sys.exit(2)

try:
    from scipy.ndimage import gaussian_filter
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = DATA_DIR / "results_by_participant"
CSV_DIR = SCRIPT_DIR / "data"

DEFAULT_LABREC_DIR = Path.home() / "Documents" / "CurrentStudy" / "sub-P001" / "ses-S001" / "eeg"

MOUSE_CH_X = 1
MOUSE_CH_Y = 2

GAZE_CH_X = 0
GAZE_CH_Y = 1
GAZE_CH_V = 5

DEFAULT_ASPECT = 1.6

MOUSE_COLOR = "#FF6B35"
GAZE_COLOR = "#4A90D9"
TARGET_COLOR = "black"
IDEAL_COLOR = "#888888"

WAIT_POLL_S = 1.0
WAIT_STABLE_S = 2.0
WAIT_MAX_S = 120.0


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_latest_csv(participant: str) -> Path | None:
    pattern = str(CSV_DIR / f"{participant}_*_mouseclick_*.csv")
    matches = sorted(glob.glob(pattern))
    if not matches:
        pattern = str(CSV_DIR / f"{participant}_*.csv")
        matches = sorted(glob.glob(pattern))
    return Path(matches[-1]) if matches else None


def find_latest_xdf(labrec_dir: Path) -> Path | None:
    if not labrec_dir.exists():
        return None
    xdfs = sorted(labrec_dir.glob("*.xdf"), key=lambda p: p.stat().st_mtime)
    return xdfs[-1] if xdfs else None


def wait_for_new_xdf(labrec_dir: Path, since_mtime: float) -> Path | None:
    """Poll for a newer-than-`since_mtime` XDF whose size is stable for WAIT_STABLE_S."""
    deadline = time.time() + WAIT_MAX_S
    candidate: Path | None = None
    stable_since: float | None = None
    last_size = -1

    while time.time() < deadline:
        newest = find_latest_xdf(labrec_dir)
        if newest and newest.stat().st_mtime > since_mtime:
            size = newest.stat().st_size
            if candidate != newest or size != last_size:
                candidate = newest
                last_size = size
                stable_since = time.time()
            elif stable_since and (time.time() - stable_since) >= WAIT_STABLE_S:
                return newest
        time.sleep(WAIT_POLL_S)

    return candidate


# ---------------------------------------------------------------------------
# XDF + CSV loading
# ---------------------------------------------------------------------------

def load_csv(csv_path: Path) -> pd.DataFrame | None:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "trial" not in df.columns:
        return None
    df = df[df["trial"].notna() & (df["trial"].astype(str).str.strip() != "")].copy()
    for col in ["trial", "target_x", "target_y", "click_x", "click_y", "rt"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["target_x", "target_y", "click_x", "click_y"])
    return df.sort_values("trial").reset_index(drop=True)


def load_xdf(xdf_path: Path):
    streams, _ = pyxdf.load_xdf(str(xdf_path))

    def find(name):
        return next((s for s in streams if s["info"]["name"][0] == name), None)

    mkr = find("PsychoPyMarkers")
    mouse = find("PsychoPyStream")
    gaze = find("GazepointEyeTracker")

    marker_df = None
    if mkr is not None:
        rows = []
        prev = None
        for t, s in zip(mkr["time_stamps"], mkr["time_series"]):
            txt = str(s[0])
            if txt != prev:
                rows.append({"time": float(t), "text": txt})
                prev = txt
        marker_df = pd.DataFrame(rows)

    mouse_times = mouse_xy = None
    if mouse is not None:
        mouse_times = np.array(mouse["time_stamps"])
        mouse_arr = np.array(mouse["time_series"], dtype=float)
        if mouse_arr.shape[1] > max(MOUSE_CH_X, MOUSE_CH_Y):
            mouse_xy = mouse_arr[:, [MOUSE_CH_X, MOUSE_CH_Y]]

    gaze_times = gaze_arr = None
    if gaze is not None:
        gaze_times = np.array(gaze["time_stamps"])
        gaze_arr = np.array(gaze["time_series"], dtype=float)

    return marker_df, mouse_times, mouse_xy, gaze_times, gaze_arr


# ---------------------------------------------------------------------------
# Per-trial extraction
# ---------------------------------------------------------------------------

def _segment(times, data, t0, t1):
    if times is None or data is None:
        return None
    mask = (times >= t0) & (times <= t1)
    return data[mask] if mask.any() else None


def _marker_value(label, key):
    prefix = f"{key}:"
    for part in str(label).split(","):
        if part.startswith(prefix):
            return part[len(prefix):]
    return None


def _marker_trial(label):
    raw = _marker_value(label, "trial")
    try:
        return int(raw) if raw is not None else None
    except ValueError:
        return None


def build_trials(csv_df, marker_df, mouse_times, mouse_xy, gaze_times, gaze_arr, aspect=DEFAULT_ASPECT):
    start_by_trial = {}
    end_by_trial = {}
    outcome_by_trial = {}
    if marker_df is not None and not marker_df.empty:
        starts = marker_df[marker_df["text"].str.startswith("mouse_trial_start")]
        ends = marker_df[marker_df["text"].str.startswith("mouse_trial_end")]
        for _, marker in starts.iterrows():
            trial_num = _marker_trial(marker["text"])
            if trial_num is not None:
                start_by_trial[trial_num] = float(marker["time"])
        for _, marker in ends.iterrows():
            text = marker["text"]
            trial_num = _marker_trial(text)
            if trial_num is None:
                continue
            if _marker_value(text, "success") == "1":
                outcome = "success"
            elif _marker_value(text, "timeout") == "1":
                outcome = "timeout"
            elif _marker_value(text, "aborted") == "1":
                outcome = "aborted"
            else:
                outcome = "unknown"
            outcome_by_trial[trial_num] = outcome
            if outcome == "success":
                end_by_trial[trial_num] = float(marker["time"])

    trials = []
    prev_cx, prev_cy = 0.0, 0.0

    for _, row in csv_df.iterrows():
        trial_num = int(row["trial"])
        tx, ty = float(row["target_x"]), float(row["target_y"])
        cx, cy = float(row["click_x"]), float(row["click_y"])

        mouse_path = gaze_path = None

        onset_time = start_by_trial.get(trial_num)
        click_time = end_by_trial.get(trial_num)
        if onset_time is not None and click_time is not None:

            seg = _segment(mouse_times, mouse_xy, onset_time, click_time)
            if seg is not None and len(seg) >= 2:
                mouse_path = seg

            seg = _segment(gaze_times, gaze_arr, onset_time, click_time)
            if seg is not None and len(seg) >= 2:
                gx_pp = (seg[:, GAZE_CH_X] - 0.5) * aspect
                gy_pp = 0.5 - seg[:, GAZE_CH_Y]
                if seg.shape[1] > GAZE_CH_V:
                    valid = seg[:, GAZE_CH_V] > 0.5
                    if valid.any():
                        gaze_path = np.column_stack([gx_pp[valid], gy_pp[valid]])
                else:
                    gaze_path = np.column_stack([gx_pp, gy_pp])

        trials.append({
            "trial":        trial_num,
            "outcome":      outcome_by_trial.get(trial_num, "unknown"),
            "target_x":     tx,
            "target_y":     ty,
            "click_x":      cx,
            "click_y":      cy,
            "prev_click_x": prev_cx,
            "prev_click_y": prev_cy,
            "mouse_path":   mouse_path,
            "gaze_path":    gaze_path,
        })
        prev_cx, prev_cy = cx, cy

    return trials


# ---------------------------------------------------------------------------
# Rotation helper
# ---------------------------------------------------------------------------

def rotate_to_right(pts, angle_rad):
    """Rotate (N,2) points by -angle_rad so that direction angle_rad -> (+x).

    Matches the convention used in the PsychoPy workspace analyse_trajectories.py:
    trial origin is the screen centre (0,0); target is at polar angle
    atan2(target_y, target_x) from the centre; after rotation the target lands
    at (+target_r, 0) and the straight-line "ideal" path lies along y = 0.
    No translation is applied.
    """
    if pts is None or len(pts) < 1:
        return None
    c = math.cos(-angle_rad)
    s = math.sin(-angle_rad)
    R = np.array([[c, -s], [s, c]])
    return np.asarray(pts, dtype=float) @ R.T


def resample_by_arclength(pts, n=200):
    """Resample a 2D path to `n` equally-spaced points by arc length.

    Removes time-in-place bias: every unit of *distance* contributes equally to
    the pooled density, regardless of how long the participant lingered there.
    """
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return pts
    diffs = np.diff(pts, axis=0)
    seg_len = np.sqrt((diffs ** 2).sum(axis=1))
    cumlen = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cumlen[-1])
    if total == 0.0:
        return pts[:1]
    targets = np.linspace(0.0, total, n)
    xs = np.interp(targets, cumlen, pts[:, 0])
    ys = np.interp(targets, cumlen, pts[:, 1])
    return np.column_stack([xs, ys])


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _draw_raw_panel(ax, trials, key, color, label):
    drew_any = False
    for t in trials:
        p = t[key]
        if p is not None:
            ax.plot(p[:, 0], p[:, 1], color=color, linewidth=0.7, alpha=0.55, zorder=3)
            drew_any = True

    radii = [math.hypot(t["target_x"], t["target_y"]) for t in trials]
    if radii:
        r = float(np.mean(radii))
        theta = np.linspace(0, 2 * math.pi, 300)
        ax.plot(r * np.cos(theta), r * np.sin(theta),
                color="gray", linewidth=0.5, linestyle="--", alpha=0.4, zorder=1)

    tx = [t["target_x"] for t in trials]
    ty = [t["target_y"] for t in trials]
    ax.scatter(tx, ty, color=TARGET_COLOR, s=28, zorder=5, alpha=0.65, label="Target")

    ax.scatter([0], [0], color="gray", s=24, zorder=6)
    ax.axhline(0, color="lightgray", linewidth=0.4)
    ax.axvline(0, color="lightgray", linewidth=0.4)

    lim = max((max(radii) if radii else 0.5) + 0.12, 0.6)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (PsychoPy norm)")
    ax.set_ylabel("y (PsychoPy norm)")

    handles = [
        Line2D([0], [0], color=color, lw=1.4, label=f"{label} path"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=TARGET_COLOR,
               markersize=6, label="Target"),
    ]
    note = "" if drew_any else " (no trajectory samples found)"
    ax.set_title(f"{label} vs target - raw{note}", fontsize=10)
    ax.legend(handles=handles, fontsize=8, loc="upper left")


def _draw_rotated_panel(ax, trials, key, label, n_resample=200, bins=80, smooth_sigma=1.5):
    """Arc-length-resampled density heatmap after rotation of each trial.

    Matches fig_density_pair in analyse_trajectories.py:
      - Each trajectory resampled to n_resample equidistant points by arc length
        (removes dwell-time bias).
      - Each trial rotated so its target lies on +x (origin stays at (0,0)).
      - Points pooled across trials and binned into a 2D histogram.
      - Optional gaussian smoothing, cmap="hot".
    """
    radii = [math.hypot(t["target_x"], t["target_y"]) for t in trials]
    target_r = float(np.mean(radii)) if radii else 0.5
    lim = max(target_r + 0.15, 0.65)

    pooled = []
    for t in trials:
        p = t[key]
        if p is None or len(p) < 2:
            continue
        angle = math.atan2(t["target_y"], t["target_x"])
        resampled = resample_by_arclength(p, n=n_resample)
        rotated = rotate_to_right(resampled, angle)
        if rotated is not None and len(rotated) > 0:
            pooled.append(rotated)

    if not pooled:
        ax.text(0.5, 0.5, "No trajectory samples found",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color="gray")
        ax.set_title(f"{label} - rotated density (no data)", fontsize=10)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        return

    pts = np.vstack(pooled)
    h, _xe, _ye = np.histogram2d(
        pts[:, 0], pts[:, 1],
        bins=bins,
        range=[[-lim, lim], [-lim, lim]],
    )
    if _SCIPY_OK:
        h = gaussian_filter(h, sigma=smooth_sigma)

    im = ax.imshow(
        h.T, origin="lower", aspect="equal",
        extent=(-lim, lim, -lim, lim),
        cmap="hot", interpolation="bilinear",
    )
    plt.colorbar(im, ax=ax, label="arc-length density", shrink=0.8)

    ax.axhline(0, color="white", linewidth=0.4, alpha=0.5)
    ax.axvline(0, color="white", linewidth=0.4, alpha=0.5)
    ax.scatter([0], [0], color="cyan", s=120, marker="+", linewidths=2.5,
               zorder=5, label="Start (screen centre)")
    ax.scatter([target_r], [0], color="cyan", s=80, marker="o", zorder=5,
               edgecolors="black", linewidths=0.8, label="Target (end)")

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("along target vector (PsychoPy norm)")
    ax.set_ylabel("perpendicular deviation (PsychoPy norm)")
    ax.set_title(f"{label} - rotated density (mean r={target_r:.2f})", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")


def make_figure(trials, participant, source_meta, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))

    _draw_raw_panel(axes[0, 0], trials, "gaze_path", GAZE_COLOR, "Gaze")
    _draw_raw_panel(axes[0, 1], trials, "mouse_path", MOUSE_COLOR, "Mouse")
    _draw_rotated_panel(axes[1, 0], trials, "gaze_path", "Gaze")
    _draw_rotated_panel(axes[1, 1], trials, "mouse_path", "Mouse")

    n_trials = len(trials)
    n_gaze = sum(1 for t in trials if t["gaze_path"] is not None)
    n_mouse = sum(1 for t in trials if t["mouse_path"] is not None)

    suptitle = (
        f"{participant} - quick QC ({n_trials} trials; gaze={n_gaze}, mouse={n_mouse})\n"
        f"coords: PsychoPy norm [-1,1]; aspect={DEFAULT_ASPECT:.2f}  "
        f"(future: convert to cm / deg of visual angle via monitor size + viewing distance)"
    )
    fig.suptitle(suptitle, fontsize=11, fontweight="bold", y=0.995)

    meta_text = "  |  ".join(f"{k}: {v}" for k, v in source_meta.items() if v)
    if meta_text:
        fig.text(0.5, 0.015, meta_text, ha="center", fontsize=7, color="#555555")

    plt.tight_layout(rect=(0, 0.03, 1, 0.96))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--participant", required=True, help="Participant id / name (matches CSV prefix).")
    ap.add_argument("--csv", type=Path, default=None, help="Explicit CSV path (default: auto-discover in experiment/data/).")
    ap.add_argument("--xdf", type=Path, default=None, help="Explicit XDF path (default: latest in LabRecorder dir).")
    ap.add_argument("--labrec-dir", type=Path, default=DEFAULT_LABREC_DIR,
                    help=f"LabRecorder output dir (default: {DEFAULT_LABREC_DIR}).")
    ap.add_argument("--wait-for-xdf", action="store_true",
                    help="Poll LabRecorder dir until a new, size-stable XDF appears.")
    ap.add_argument("--since-mtime", type=float, default=0.0,
                    help="When polling, only accept XDF files newer than this epoch time.")
    ap.add_argument("--out", type=Path, default=None, help="Explicit output PNG path.")
    args = ap.parse_args(argv)

    csv_path = args.csv or find_latest_csv(args.participant)
    if csv_path is None or not csv_path.exists():
        print(f"[ERROR] No PsychoPy CSV found for '{args.participant}' in {CSV_DIR}", file=sys.stderr)
        return 1
    print(f"[quick_qc] CSV: {csv_path.name}")

    xdf_path = args.xdf
    if xdf_path is None:
        if args.wait_for_xdf:
            print(f"[quick_qc] Waiting for XDF in {args.labrec_dir} ...")
            xdf_path = wait_for_new_xdf(args.labrec_dir, args.since_mtime)
        else:
            xdf_path = find_latest_xdf(args.labrec_dir)

    if xdf_path is None or not xdf_path.exists():
        print(f"[ERROR] No XDF found in {args.labrec_dir}. "
              f"Start LabRecorder and try again, or pass --xdf.", file=sys.stderr)
        return 1
    print(f"[quick_qc] XDF: {xdf_path.name}")

    csv_df = load_csv(csv_path)
    if csv_df is None or csv_df.empty:
        print(f"[ERROR] CSV has no usable trials.", file=sys.stderr)
        return 1

    marker_df, m_t, m_xy, g_t, g_arr = load_xdf(xdf_path)
    if marker_df is None:
        print(f"[ERROR] XDF missing PsychoPyMarkers stream.", file=sys.stderr)
        return 1

    trials = build_trials(csv_df, marker_df, m_t, m_xy, g_t, g_arr)
    if not trials:
        print(f"[ERROR] No trials matched between CSV and XDF markers.", file=sys.stderr)
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or (RESULTS_DIR / args.participant / f"quick_qc_{timestamp}.png")

    source_meta = {
        "participant": args.participant,
        "csv": csv_path.name,
        "xdf": xdf_path.name,
        "generated": timestamp,
    }

    saved = make_figure(trials, args.participant, source_meta, Path(out_path))
    print(f"[quick_qc] Saved: {saved}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
