#!/usr/bin/env python
"""Final QC-aligned eye-tracking analysis for thesis figures.

Applies the pre-specified rules documented in the thesis:
  - gaze clip: at least 70% valid FPOG samples;
  - pupil clip: at least 70% valid millimetre pupil samples and a valid baseline;
  - modality-level participant inclusion: no more than three failed observed
    clips. Clips that were never recorded are reported separately as missing
    and are not silently reclassified as failed quality-control checks.

The primary recording only is loaded from sub-*/ses-*/*.xdf. Nested technical
or repeat-session XDF files are therefore not mixed into the primary analysis.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analyse_gaze_pupil_isc import (
    baseline_correct,
    epoch,
    isc_leave_one_out,
    load_session,
    video_windows,
)


MIN_VALID_PCT = 70.0
MAX_FAILED_CLIPS = 3
EXPECTED_CLIPS = 26


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sessions = {}
    for path in sorted(args.data_dir.glob("sub-*/ses-*/*.xdf")):
        participant = path.name.split("_")[0]
        session = load_session(path)
        if session is not None:
            sessions[participant] = session

    epoch_rows = []
    traces = {}
    for participant, session in sessions.items():
        for video_id, condition, start, end, base_start, base_end in video_windows(session["events"]):
            if condition not in {"viral", "flop"}:
                continue
            item = epoch(session, start, end)
            if item is None:
                continue
            pupil_bc, pupil_baseline = baseline_correct(
                item["pupil"], session, base_start, base_end
            )
            gaze_pass = item["valid_gaze_pct"] >= MIN_VALID_PCT
            pupil_pass = (
                item["valid_pupil_pct"] >= MIN_VALID_PCT
                and np.isfinite(pupil_baseline)
                and np.isfinite(pupil_bc).sum() >= 20
            )
            epoch_rows.append({
                "participant": participant,
                "video_id": video_id,
                "condition": condition,
                "duration_s": item["duration_s"],
                "valid_gaze_pct": item["valid_gaze_pct"],
                "valid_pupil_pct": item["valid_pupil_pct"],
                "gaze_clip_pass": gaze_pass,
                "pupil_clip_pass": pupil_pass,
                "n_fixations": item.get("n_fixations", np.nan),
                "fixation_rate_hz": item.get("n_fixations", np.nan) / item["duration_s"],
                "mean_fix_dur": item.get("mean_fix_dur", np.nan),
                "gaze_dispersion": item.get("gaze_dispersion", np.nan),
                "blink_rate_per_min": item.get("blink_rate_per_min", np.nan),
                "pupil_baseline_mm": pupil_baseline,
                "pupil_mean_bc": float(np.nanmean(pupil_bc)),
            })
            traces[(participant, video_id)] = {
                "condition": condition,
                "gaze_x": item["gaze_x"],
                "gaze_y": item["gaze_y"],
                "pupil": pupil_bc,
                "gaze_pass": gaze_pass,
                "pupil_pass": pupil_pass,
            }

    metrics = pd.DataFrame(epoch_rows)
    if metrics.empty:
        raise RuntimeError("No participant-video epochs were found")

    participant_qc = []
    for participant, group in metrics.groupby("participant"):
        observed = group["video_id"].nunique()
        gaze_passed = int(group.loc[group.gaze_clip_pass, "video_id"].nunique())
        pupil_passed = int(group.loc[group.pupil_clip_pass, "video_id"].nunique())
        gaze_failed = observed - gaze_passed
        pupil_failed = observed - pupil_passed
        missing = max(0, EXPECTED_CLIPS - observed)
        participant_qc.append({
            "participant": participant,
            "observed_clips": observed,
            "missing_clips": missing,
            "complete_session": observed == EXPECTED_CLIPS,
            "gaze_passed_clips": gaze_passed,
            "gaze_failed_clips": gaze_failed,
            "gaze_eligible": gaze_failed <= MAX_FAILED_CLIPS,
            "pupil_passed_clips": pupil_passed,
            "pupil_failed_clips": pupil_failed,
            "pupil_eligible": pupil_failed <= MAX_FAILED_CLIPS,
        })
    qc = pd.DataFrame(participant_qc)
    metrics = metrics.merge(qc[["participant", "gaze_eligible", "pupil_eligible"]],
                            on="participant", how="left")

    gaze_eligible = set(qc.loc[qc.gaze_eligible, "participant"])
    pupil_eligible = set(qc.loc[qc.pupil_eligible, "participant"])

    isc_rows = []
    for video_id in sorted(metrics.video_id.unique()):
        condition = metrics.loc[metrics.video_id == video_id, "condition"].iloc[0]
        for channel in ("gaze_x", "gaze_y", "pupil"):
            modality_set = pupil_eligible if channel == "pupil" else gaze_eligible
            pass_key = "pupil_pass" if channel == "pupil" else "gaze_pass"
            selected = {
                participant: value[channel]
                for (participant, vid), value in traces.items()
                if vid == video_id and participant in modality_set and value[pass_key]
            }
            if len(selected) < 3:
                continue
            common_length = min(len(array) for array in selected.values())
            selected = {key: array[:common_length] for key, array in selected.items()}
            for participant, correlation in isc_leave_one_out(selected).items():
                isc_rows.append({
                    "participant": participant,
                    "video_id": video_id,
                    "condition": condition,
                    "channel": channel,
                    "isc_loo": correlation,
                    "contributing_participants": len(selected),
                })

    isc = pd.DataFrame(isc_rows)
    clip_isc = (isc.groupby(["video_id", "condition", "channel"], as_index=False)
                   .agg(mean_isc=("isc_loo", "mean"),
                        median_isc=("isc_loo", "median"),
                        n_participant_values=("isc_loo", "size")))

    metrics.to_csv(args.out / "participant_video_metrics_final.csv", index=False)
    qc.to_csv(args.out / "participant_modality_qc_final.csv", index=False)
    isc.to_csv(args.out / "gaze_pupil_isc_final.csv", index=False)
    clip_isc.to_csv(args.out / "clip_level_isc_final.csv", index=False)

    print(f"sessions loaded: {len(sessions)}")
    print(f"gaze eligible: {len(gaze_eligible)} -> {', '.join(sorted(gaze_eligible))}")
    print(f"pupil eligible: {len(pupil_eligible)} -> {', '.join(sorted(pupil_eligible))}")
    print(qc.to_string(index=False))
    print(f"outputs: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
