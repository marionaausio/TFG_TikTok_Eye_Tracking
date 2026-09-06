#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone PsychoPy + LSL task that replicates the prior Builder workflow.

Flow:
1) Welcome / instructions
2) Mouse targeting task (10 trials): click inside circle as fast and accurately as possible
3) Optional video block: configured test videos (sound on)
4) Goodbye

LSL streams:
- PsychoPyStream: continuous mouse data (10 channels)
- PsychoPyMarkers: event markers

Notes:
- External device streams (e.g., OpenBCI EEG, GazePoint) are expected to be
  published by their own LSL outlets and aligned offline via LSL timestamps.
"""

import os
import hashlib
import random
import argparse
import subprocess
import sys
import math
import time
import re
from pathlib import Path

from psychopy import prefs

_audio_device_override = os.environ.get("PSYCHOPY_AUDIO_DEVICE", "").strip()
if _audio_device_override:
    prefs.hardware["audioDevice"] = [_audio_device_override]
else:
    # Prefer the local Realtek speakers over monitor HDMI audio on this machine.
    prefs.hardware["audioDevice"] = ["Speakers (Realtek(R) Audio)", "default"]

from psychopy import core, data, event, gui, logging, sound, visual
from psychopy.constants import FINISHED

# Participant-facing UI translations (EN/ES/CA). Importable from this script's dir.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import translations as I18N

# Language the participant picks at the start; drives all participant-facing text.
# Operator dialogs stay English. Set by run_language_select().
EXPERIMENT_LANG = "en"

ALIGNMENT_TARGET_HZ = 250
# Common ERP-style pre-stimulus fixation baseline with conservative jitter.
MOUSE_FIXATION_BASE_S = 1.0
MOUSE_FIXATION_JITTER_FRACTION = 0.10
MOUSE_TRIAL_COUNT = 30
N_PRACTICE_MOUSE_TRIALS = 5  # warm-up trials before the real block (not analysed)
MOUSE_TRIAL_CLICK_TIMEOUT_S = 15.0
MOUSE_POST_CLICK_FIXATION_S = 0.5
# Pupil baseline shown before EVERY video: a fixed-duration grey screen (the window
# colour) with a fixation cross. Each clip needs its OWN baseline - previously the
# fixation ran once before the whole video block, so only clip 1 had a clean baseline
# and every later clip inherited pupil carry-over from the preceding rating page,
# which makes baseline-corrected dilation uninterpretable.
# This duration is deliberately NOT jittered: baseline windows must be the same
# length across clips for baseline correction to be comparable between them.
# The window is delimited in the marker stream by pre_video_baseline_start/_end so
# analysis can extract exactly this interval per clip.
PRE_VIDEO_BASELINE_S = 1.0
# Fraction of the clear centre-to-edge distance at which target centres are placed.
# Reduced from 0.90 to 0.50 on 2026-08-17: the screen-mounted Gazepoint eye tracker
# sits below the display and physically obscured the lowest circles, so participants
# could not see targets they were being asked to click. This scales the placement
# radius, not the circle size.
MOUSE_TARGET_RADIUS_FRACTION = 0.50
# Go/No-Go (redirect) block, run right after the targeting block: a circle appears
# in one of two luminances - click a GREY circle (go), but on a WHITE circle redirect
# and click the centre fixation cross instead (no-go). White is the response just
# trained in the targeting block, so overriding it indexes response inhibition.
N_GONOGO_TRIALS = 30
N_GONOGO_NOGO = 10               # of N_GONOGO_TRIALS, this many are WHITE (no-go); rest GREY (go)
N_GONOGO_PRACTICE = 6
GONOGO_CROSS_HIT_RADIUS = 0.06   # height units: a click within this of centre = "clicked the cross"
OPERATOR_NOTE_MAX_CHARS = 200
OPERATOR_NOTE_WARNING = (
    "Do not enter participant names or identifying details. "
    "Use the participant CID (e.g. VC-07) if you need to reference a person."
)
PARTICIPANT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
PARTICIPANT_ID_ERROR = (
    "Participant ID must be alphanumeric (e.g. VC-07). "
    "Real names are not permitted; use the CID from participants.csv."
)
PREFER_EXTERNAL_AUDIO_SYNC = True
# Positive values delay video start relative to audio when sidecar audio is used.
# Keep default neutral on this laptop; override via CLI/env/UI if needed.
DEFAULT_EXTERNAL_AV_VIDEO_DELAY_S = 0.000
# Playback safety guards to prevent indefinite hangs on backend/file issues.
DEFAULT_MOVIE_GUARD_S = 180.0
DEFAULT_OPENCV_GUARD_S = 180.0
# Keep OpenCV as init-time rescue only. Mid-run backend switching is disabled by default.
ALLOW_OPENCV_FALLBACK_ON_INIT_ERROR = True
ALLOW_OPENCV_FALLBACK_AFTER_DRAW_ERROR = False
# Wider defaults for machines where video remains ahead at <=180 ms delay.
DEFAULT_AV_SYNC_SWEEP_DELAYS_S = [0.200, 0.250, 0.300, 0.350, 0.400]
# Allow wider temporary offsets for diagnostic sync validation runs.
MIN_AV_VIDEO_DELAY_S = -2.000
MAX_AV_VIDEO_DELAY_S = 2.000

QUICK_QC_SCRIPT = Path(__file__).resolve().parent / "quick_qc.py"
QUICK_QC_PYTHON = os.environ.get(
    "QUICK_QC_PYTHON",
    sys.executable,
)
EXTERNAL_AUDIO_SUFFIXES = (
    "_audio_pcm_48k.wav",
    "_audio_pcm.wav",
    "_audio.wav",
)
LAST_VIDEO_ORDER_FILE = "_last_video_order.txt"
# Keep configured stimuli in a PsychoPy-safe profile (MP4/H.264/yuv420p, 60fps, MP3 audio).
TEST_VIDEO_CANDIDATES = [
    {"id": "catrice_flop",            "filename": "Catrice_Flop.mp4",            "file_key": "Catrice_Flop",            "condition": "flop"},
    {"id": "diorbeauty_flop",         "filename": "DiorBeauty_Flop.mp4",         "file_key": "DiorBeauty_Flop",         "condition": "flop"},
    {"id": "elfcosmetics_viral",      "filename": "elfCosmetics_Viral.mp4",      "file_key": "elfCosmetics_Viral",      "condition": "viral"},
    {"id": "essence_flop",            "filename": "Essence_Flop.mp4",            "file_key": "Essence_Flop",            "condition": "flop"},
    {"id": "hudabeauty_paris_flop",   "filename": "HudaBeauty_Paris_Flop.mp4",   "file_key": "HudaBeauty_Paris_Flop",   "condition": "flop"},
    {"id": "hudabeauty_viral",        "filename": "HudaBeauty_Viral.mp4",        "file_key": "HudaBeauty_Viral",        "condition": "viral"},
    {"id": "kiehld_flop",             "filename": "Kiehld_Flop.mp4",             "file_key": "Kiehld_Flop",             "condition": "flop"},
    {"id": "kikomilano_lipcomb_viral","filename": "KikoMilanoLipComb_Viral.mp4", "file_key": "KikoMilanoLipComb_Viral", "condition": "viral"},
    {"id": "kikomilano_peach_flop",   "filename": "KikoMilanoPeach_Flop.mp4",    "file_key": "KikoMilanoPeach_Flop",    "condition": "flop"},
    {"id": "kyliecosmetics_flop",     "filename": "KylieCosmetics_Flop.mp4",     "file_key": "KylieCosmetics_Flop",     "condition": "flop"},
    {"id": "kyliecosmetics_red_flop", "filename": "KylieCosmeticsRed_Flop.mp4",  "file_key": "KylieCosmeticsRed_Flop",  "condition": "flop"},
    {"id": "lancome_viral",           "filename": "Lancome_Viral.mp4",           "file_key": "Lancome_Viral",           "condition": "viral"},
    {"id": "laniege_viral",           "filename": "Laniege_Viral.mp4",           "file_key": "Laniege_Viral",           "condition": "viral"},
    {"id": "makeupbymario_viral",     "filename": "MakeUpByMario_Viral.mp4",     "file_key": "MakeUpByMario_Viral",     "condition": "viral"},
    {"id": "makeupforever_flop",      "filename": "MakeUpForever_Flop.mp4",      "file_key": "MakeUpForever_Flop",      "condition": "flop"},
    {"id": "maybellineny_flop",       "filename": "MaybellineNY_Flop.mp4",       "file_key": "MaybellineNY_Flop",       "condition": "flop"},
    {"id": "gisou_flop",              "filename": "Gisou_Flop.mp4",              "file_key": "Gisou_Flop",              "condition": "flop"},
    {"id": "olaplex_flop",            "filename": "Olaplex_Flop.mp4",            "file_key": "Olaplex_Flop",            "condition": "flop"},
    {"id": "orealparis_flop",         "filename": "OrealParis_Flop.mp4",         "file_key": "OrealParis_Flop",         "condition": "flop"},
    {"id": "pradabeauty_viral",       "filename": "PradaBeauty_Viral.mp4",       "file_key": "PradaBeauty_Viral",       "condition": "viral"},
    {"id": "rarebeauty_lip_viral",    "filename": "RareBeautyLip_Viral.mp4",     "file_key": "RareBeautyLip_Viral",     "condition": "viral"},
    {"id": "rarebeauty_perfum_viral", "filename": "RareBeautyPerfum_Viral.mp4",  "file_key": "RareBeautyPerfum_Viral",  "condition": "viral"},
    {"id": "rhode_birthday_viral",    "filename": "RhodeBirthday_Viral.mp4",     "file_key": "RhodeBirthday_Viral",     "condition": "viral"},
    {"id": "sheglam_viral",           "filename": "Sheglam_Viral.mp4",           "file_key": "Sheglam_Viral",           "condition": "viral"},
    {"id": "yslbeauty_makeup_viral",  "filename": "YSLBeauty_Makeup_Viral.mp4",  "file_key": "YSLBeauty_Makeup_Viral",  "condition": "viral"},
    {"id": "yslbeauty_perfum_viral",  "filename": "YSLBeauty_Viral_Perfum.mp4",  "file_key": "YSLBeauty_Viral_Perfum",  "condition": "viral"},
]


def prefer_external_audio_sync():
    """Resolve whether sidecar audio should be preferred for video playback."""
    raw = os.environ.get("PSYCHOPY_PREFER_EXTERNAL_AUDIO_SYNC")
    if raw is None:
        return bool(PREFER_EXTERNAL_AUDIO_SYNC)
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return bool(PREFER_EXTERNAL_AUDIO_SYNC)
VIDEO_FILE_EXTENSIONS = (".avi", ".mp4", ".mov", ".mkv", ".wmv", ".m4v")

try:
    from pylsl import StreamInfo, StreamOutlet, StreamInlet, local_clock, resolve_byprop
except Exception as exc:  # pragma: no cover - runtime dependency check
    StreamInfo = None
    StreamOutlet = None
    StreamInlet = None
    local_clock = None
    resolve_byprop = None
    PYLSL_IMPORT_ERROR = exc
else:
    PYLSL_IMPORT_ERROR = None

# Expected external LSL streams to monitor during experiment.
# Expected external LSL streams to monitor during experiment.
#
# min_samples is the per-stream floor for the 1-second pull check. It must be set
# from THAT stream's real rate, not a shared constant: the EEG runs at 125 Hz but
# the GP3 runs at 60 Hz unless high-speed mode is enabled, so a single 50-sample
# threshold would false-alarm on a perfectly healthy tracker (~60 samples/s).
# These are "something is badly wrong" floors, NOT quality gates - the check only
# ever warns, and sample drops are expected and handled by interpolation during
# analysis. expected_hz is used to report measured-vs-expected rate.
EXPECTED_STREAMS = [
    {"name": "OpenBCI_CytonDaisy_EEG", "type": "EEG", "critical": True, "pull_check": True,
     "min_samples": 50, "expected_hz": 125.0},
    # critical stays False: gaze drops are expected and must never abort a session.
    {"name": "GazepointEyeTracker", "type": "Gaze", "critical": False, "pull_check": True,
     "min_samples": 30, "expected_hz": 60.0},
    {"name": "GazepointEvents", "type": "Markers", "critical": False, "pull_check": False},
]
STREAM_CHECK_TIMEOUT = 2.0  # seconds to wait when resolving a stream
EEG_PULL_CHECK_DURATION = 1.0  # seconds to pull samples for the health check
EEG_MIN_SAMPLES_PER_CHECK = 50  # fallback floor when a stream has no min_samples


def check_lsl_streams(marker_outlet=None, context="unknown"):
    """Check that expected external LSL streams are alive and flowing data.

    Returns a dict with stream names as keys and status dicts as values.
    Pushes a marker with the health check result.
    """
    if resolve_byprop is None:
        return {}

    results = {}
    for stream_spec in EXPECTED_STREAMS:
        sname = stream_spec["name"]
        stype = stream_spec["type"]
        status = {"found": False, "data_flowing": None, "samples": 0,
                  "measured_hz": None, "expected_hz": stream_spec.get("expected_hz")}

        try:
            found = resolve_byprop("name", sname, timeout=STREAM_CHECK_TIMEOUT)
        except Exception:
            found = []

        if not found:
            results[sname] = status
            continue

        status["found"] = True

        # For EEG, pull samples to verify data is actually flowing
        if stream_spec.get("pull_check"):
            try:
                inlet = StreamInlet(found[0], max_buflen=2)
                inlet.open_stream(timeout=STREAM_CHECK_TIMEOUT)
                samples_pulled = 0
                import time as _time
                t_end = _time.time() + EEG_PULL_CHECK_DURATION
                while _time.time() < t_end:
                    sample, ts = inlet.pull_sample(timeout=0.05)
                    if ts is not None:
                        samples_pulled += 1
                inlet.close_stream()
                del inlet
                status["samples"] = samples_pulled
                if EEG_PULL_CHECK_DURATION > 0:
                    status["measured_hz"] = samples_pulled / EEG_PULL_CHECK_DURATION
                min_samples = stream_spec.get("min_samples", EEG_MIN_SAMPLES_PER_CHECK)
                status["data_flowing"] = samples_pulled >= min_samples
            except Exception:
                status["data_flowing"] = False

        results[sname] = status

    # Build summary marker
    parts = []
    for sname, st in results.items():
        short = sname.replace("OpenBCI_CytonDaisy_", "").replace("Gazepoint", "GP_")
        hz = st.get("measured_hz")
        hz_txt = f"@{hz:.0f}Hz" if hz is not None else ""
        if not st["found"]:
            parts.append(f"{short}:MISSING")
        elif st["data_flowing"] is False:
            parts.append(f"{short}:LOW_RATE({st['samples']}{hz_txt})")
        elif st["data_flowing"] is True:
            parts.append(f"{short}:OK({st['samples']}{hz_txt})")
        else:
            parts.append(f"{short}:FOUND")

    marker_str = f"stream_health_check,context:{context}," + ",".join(parts)
    if marker_outlet is not None:
        try:
            from pylsl import local_clock as _lc
            marker_outlet.push_sample([marker_str], _lc())
        except Exception:
            pass

    print(f"[STREAM CHECK] ({context}) {' | '.join(parts)}")
    return results


def format_stream_health_warning(results):
    """Return a user-facing warning string if any streams have problems, else None."""
    warnings = []
    for sname, st in results.items():
        if not st["found"]:
            warnings.append(f"  MISSING: {sname}")
        elif st["data_flowing"] is False:
            hz = st.get("measured_hz")
            exp = st.get("expected_hz")
            rate_txt = ""
            if hz is not None:
                rate_txt = f" = {hz:.0f} Hz"
                if exp:
                    rate_txt += f", expected ~{exp:.0f} Hz"
            warnings.append(
                f"  LOW RATE: {sname} ({st['samples']} samples in "
                f"{EEG_PULL_CHECK_DURATION}s{rate_txt})"
            )
    if not warnings:
        return None
    return (
        "Stream health WARNING:\n" + "\n".join(warnings)
        + "\n\nThis is advisory only - the session is not blocked. Occasional sample\n"
        + "loss is expected and is handled by interpolation during analysis. Check\n"
        + "the tracker/cap only if a stream is MISSING or the rate is far below target."
    )


def parse_display_mode(argv=None):
    """Resolve fullscreen/windowed mode from CLI flags or environment."""
    parser = argparse.ArgumentParser(add_help=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--fullscreen",
        action="store_true",
        help="Run PsychoPy in fullscreen mode.",
    )
    group.add_argument(
        "--windowed",
        action="store_true",
        help="Run PsychoPy in windowed mode.",
    )
    args, _unknown = parser.parse_known_args(argv)

    if args.fullscreen:
        return True, "cli(--fullscreen)"
    if args.windowed:
        return False, "cli(--windowed)"

    env_value = os.environ.get("PSYCHOPY_FULLSCREEN")
    if env_value is not None:
        normalized = env_value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True, "env(PSYCHOPY_FULLSCREEN)"
        if normalized in {"0", "false", "no", "n", "off"}:
            return False, "env(PSYCHOPY_FULLSCREEN)"
        print(
            f"[WARN] Invalid PSYCHOPY_FULLSCREEN='{env_value}'. "
            "Expected one of: 1/0, true/false, yes/no, on/off."
        )

    # Safe default for remote sessions: keep experiment windowed.
    return False, "default(windowed_remote_safe)"


def parse_runtime_options(argv=None):
    """Parse optional runtime controls (sync sweep etc.) without affecting normal runs."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--av-sync-sweep",
        type=str,
        default="",
        help="Comma-separated video delay values in seconds (e.g. 0.06,0.09,0.12).",
    )
    parser.add_argument(
        "--av-sync-video",
        type=str,
        default="nyx_viral_short",
        help="Video id (or path) to use for A/V sync sweep mode.",
    )
    parser.add_argument(
        "--av-sync-only",
        action="store_true",
        help="Run only A/V sync sweep mode and skip the main task flow.",
    )
    parser.add_argument(
        "--av-video-delay",
        type=float,
        default=None,
        help="Override video delay in seconds for normal playback and sync runs.",
    )
    parser.add_argument(
        "--video-ids",
        type=str,
        default="",
        help=(
            "Comma-separated video ids/stems/filenames to include in the main video block "
            "(e.g. nyx_viral_short,covergirl_flop_long)."
        ),
    )
    parser.add_argument(
        "--mouse-trials",
        type=int,
        default=None,
        help=(
            "Override the number of mouse-targeting trials for this run (default: "
            f"{MOUSE_TRIAL_COUNT}). Useful for quick spot checks."
        ),
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help=(
            "Skip the automated pre-session checks (preflight.py) run at task launch. "
            "Not recommended for real sessions; useful for debugging."
        ),
    )
    parser.add_argument(
        "--preflight-quick",
        action="store_true",
        help="Use shorter LSL discovery timeouts (1.5 s vs 3.0 s) during the inline preflight.",
    )
    parser.add_argument(
        "--screen",
        type=int,
        default=None,
        help=(
            "Stimulus monitor index (PsychoPy/pyglet screen). Default: auto-detect the "
            "PRIMARY monitor (the one at virtual origin 0,0), which is the frame the "
            "mouse bridge normalises against. Override only to force a specific display."
        ),
    )
    parser.add_argument(
        "--practice-trials",
        type=int,
        default=None,
        help=(
            f"Warm-up mouse-targeting practice trials before the real block "
            f"(default: {N_PRACTICE_MOUSE_TRIALS}; 0 = skip). Practice trials emit "
            f"'mouse_practice_trial_*' markers and are excluded from analysis."
        ),
    )
    parser.add_argument(
        "--no-hardware",
        action="store_true",
        help=(
            "No-EEG / no-Gazepoint trial run: skips the inline LSL preflight, the EEG "
            "quality dialog, and the gaze-calibration dialogs, so the FULL experiment "
            "runs and records responses + mouse with only the mouse bridge + "
            "LabRecorder. For verifying the PsychoPy/LSL/recording pipeline, not for "
            "real participants."
        ),
    )
    args, _unknown = parser.parse_known_args(argv)
    return args


def parse_delay_values(raw_value):
    """Parse comma-separated delay values in seconds."""
    if not raw_value:
        return []
    delays = []
    for part in str(raw_value).split(","):
        token = part.strip()
        if not token:
            continue
        try:
            val = float(token)
        except Exception:
            continue
        delays.append(max(MIN_AV_VIDEO_DELAY_S, min(MAX_AV_VIDEO_DELAY_S, val)))
    return delays


def parse_csv_tokens(raw_value):
    """Parse comma- or semicolon-separated string tokens."""
    if not raw_value:
        return []
    values = []
    for part in re.split(r"[;,]", str(raw_value)):
        token = part.strip()
        if token:
            values.append(token)
    return values


def sanitize_operator_note(value):
    """Sanitize free-text operator notes before persistence or marker emission."""
    return str(value or "").strip().replace(",", ";")[:OPERATOR_NOTE_MAX_CHARS]


def get_last_video_order_path(this_dir):
    """Return path used to persist the previous run's shuffled video order."""
    data_dir = Path(this_dir).resolve() / "data"
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return data_dir / LAST_VIDEO_ORDER_FILE


def read_last_video_order(this_dir, marker_outlet=None):
    """Read previously used video order ids from disk."""
    path = get_last_video_order_path(this_dir)
    if not path.exists():
        return []
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        safe_push(marker_outlet, f"last_video_order_unreadable,error:{type(exc).__name__}")
        return []
    if not raw:
        return []
    return [token for token in raw.split("|") if token]


def write_last_video_order(this_dir, order_ids, marker_outlet=None):
    """Persist current run video order ids for next-run de-duplication."""
    path = get_last_video_order_path(this_dir)
    payload = "|".join(str(x) for x in order_ids)
    try:
        path.write_text(payload, encoding="utf-8")
    except OSError as exc:
        safe_push(marker_outlet, f"last_video_order_unwritable,error:{type(exc).__name__}")
        pass


def shuffle_video_jobs_avoid_repeat(video_jobs, this_dir, marker_outlet=None):
    """Shuffle videos, avoiding exact repeat of prior run order when possible."""
    if len(video_jobs) <= 1:
        return list(video_jobs)

    last_ids = read_last_video_order(this_dir, marker_outlet=marker_outlet)
    if not last_ids:
        return random.sample(video_jobs, k=len(video_jobs))

    current_ids = [str(v.get("id", "video")) for v in video_jobs]
    if set(last_ids) != set(current_ids):
        return random.sample(video_jobs, k=len(video_jobs))

    for _ in range(12):
        shuffled = random.sample(video_jobs, k=len(video_jobs))
        shuffled_ids = [str(v.get("id", "video")) for v in shuffled]
        if shuffled_ids != last_ids:
            return shuffled

    # Extremely unlikely fallback: force a different order deterministically.
    forced = random.sample(video_jobs, k=len(video_jobs))
    return forced[1:] + forced[:1]


def cli_flag_present(argv, flag_name):
    """Return True if a flag was explicitly provided in argv."""
    if not argv:
        return False
    normalized = str(flag_name).strip()
    if not normalized:
        return False
    prefix = normalized + "="
    for token in argv:
        token = str(token).strip()
        if token == normalized or token.startswith(prefix):
            return True
    return False


def lsl_now():
    """Return current local LSL time, or None if unavailable."""
    if local_clock is None:
        return None
    try:
        return float(local_clock())
    except Exception:
        return None


def safe_push(marker_outlet, message, timestamp=None):
    """Push marker text; if possible include explicit LSL timestamp."""
    if marker_outlet is None:
        return
    if timestamp is None:
        timestamp = lsl_now()
    try:
        if timestamp is None:
            marker_outlet.push_sample([message])
        else:
            marker_outlet.push_sample([message], float(timestamp))
    except TypeError:
        try:
            marker_outlet.push_sample([message])
        except Exception:
            pass
    except Exception:
        pass


def safe_push_data(data_outlet, sample, timestamp=None):
    """Push numeric sample; if possible include explicit LSL timestamp."""
    if data_outlet is None:
        return
    if timestamp is None:
        timestamp = lsl_now()
    try:
        if timestamp is None:
            data_outlet.push_sample(sample)
        else:
            data_outlet.push_sample(sample, float(timestamp))
    except TypeError:
        try:
            data_outlet.push_sample(sample)
        except Exception:
            pass
    except Exception:
        pass


def build_lsl_outlets():
    # Mouse position is published continuously by the standalone stream_mouse.py
    # bridge (the 'PsychoPyStream' stream, started at the LSL-streamers step), which
    # captures the cursor for the WHOLE session - including the targeting trials.
    # So the task no longer creates its own PsychoPyStream; doing so would record
    # the mouse twice. It emits only the PsychoPyMarkers event stream. The
    # targeting target position is in the mouse_trial_start markers, so trials
    # stay fully reconstructable from PsychoPyStream + PsychoPyMarkers.
    marker_info = StreamInfo(
        name="PsychoPyMarkers",
        type="Markers",
        channel_count=1,
        nominal_srate=0,
        channel_format="string",
        source_id="my_unique_marker_stream_id",
    )
    marker_outlet = StreamOutlet(marker_info)

    # data_outlet is None: every safe_push_data()/push_mouse_sample() call becomes
    # a no-op (safe_push_data returns early on None), so the targeting loop is
    # unchanged except that it no longer publishes a duplicate mouse stream.
    return None, marker_outlet


LABRECORDER_RCS_ADDR = ("127.0.0.1", 22345)


def labrecorder_rcs(*commands, timeout=1.5):
    """Best-effort control of LabRecorder via its Remote Control Server (RCS).

    Sends each command (e.g. 'update', 'select all', 'start', 'stop') over one TCP
    connection to 127.0.0.1:22345. Returns True if LabRecorder accepted the
    connection, False if it is not reachable (not open, or 'Enable RCS' unchecked)
    - in which case the caller falls back to the on-screen manual instructions, so
    the session is never blocked by RCS being unavailable."""
    import socket
    try:
        with socket.create_connection(LABRECORDER_RCS_ADDR, timeout=timeout) as sock:
            sock.settimeout(0.4)
            for cmd in commands:
                sock.sendall((str(cmd).strip() + "\n").encode("utf-8"))
                time.sleep(0.15)
                try:
                    reply = sock.recv(4096).decode("utf-8", "replace").strip().lower()
                except Exception:
                    reply = ""  # most RCS commands are silent on success
                if any(bad in reply for bad in ("error", "fail", "already")):
                    logging.warning(f"LabRecorder RCS rejected '{cmd}': {reply}")
                    return False
        return True
    except Exception as exc:
        try:
            logging.warning(f"LabRecorder RCS not reachable ({exc}); use manual Start/Stop.")
        except Exception:
            pass
        return False


def wait_for_space(win, text, height=0.08):
    stim = visual.TextStim(
        win=win,
        text=text,
        font="Arial",
        units="norm",
        pos=(0, 0),
        height=height,
        wrapWidth=1.8,
        color="white",
        colorSpace="rgb",
    )

    event.clearEvents(eventType="keyboard")
    while True:
        stim.draw()
        win.flip()
        keys = event.getKeys(keyList=["space", "escape"])
        if "escape" in keys:
            raise KeyboardInterrupt
        if "space" in keys:
            # Flash the prompt green so the press is visibly registered, even when
            # the next phase takes a moment to appear.
            stim.color = "lime"
            stim.draw()
            win.flip()
            core.wait(0.30)
            break


def show_fixation(win, duration_s=0.5):
    cross = visual.TextStim(
        win=win,
        text="+",
        font="Arial",
        units="norm",
        pos=(0, 0),
        height=0.12,
        color="white",
        colorSpace="rgb",
    )
    timer = core.Clock()
    while timer.getTime() < duration_s:
        cross.draw()
        win.flip()
        if "escape" in event.getKeys(keyList=["escape"]):
            raise KeyboardInterrupt


def jittered_duration(base_s, jitter_fraction):
    if base_s <= 0:
        return 0.0
    jitter_fraction = max(0.0, float(jitter_fraction))
    low = base_s * (1.0 - jitter_fraction)
    high = base_s * (1.0 + jitter_fraction)
    return random.uniform(low, high)


def get_fixed_target_radius(win, circle_size):
    """Return fixed target radius in height units (with visibility clamp)."""
    aspect_ratio = float(win.size[0]) / float(win.size[1])
    # Distance from center to top/bottom edge keeping full circle visible.
    max_y = max(0.0, 0.5 - (circle_size[1] / 2.0))
    requested_radius = float(MOUSE_TARGET_RADIUS_FRACTION) * max_y
    # Clamp by horizontal visibility as a safety guard across display aspect ratios.
    max_x = max(0.0, (0.5 * aspect_ratio) - (circle_size[0] / 2.0))
    return max(0.0, min(requested_radius, max_x))


def random_circle_pos(win, circle_size, fixed_radius=None):
    radius = get_fixed_target_radius(win, circle_size) if fixed_radius is None else float(fixed_radius)
    if radius <= 0:
        return (0.0, 0.0)
    theta = random.uniform(0.0, 2.0 * math.pi)
    return (radius * math.cos(theta), radius * math.sin(theta))


def push_mouse_sample(data_outlet, mouse):
    curr_x, curr_y = mouse.getPos()
    curr_rel = mouse.getRel()
    curr_wheel = mouse.getWheelRel()
    buttons = mouse.getPressed()

    sample = [
        0.0,
        float(curr_x),
        float(curr_y),
        float(curr_rel[0]),
        float(curr_rel[1]),
        float(curr_wheel[0]),
        float(curr_wheel[1]),
        float(buttons[0]),
        float(buttons[1]),
        float(buttons[2]),
    ]
    safe_push_data(data_outlet, sample)


def run_mouse_trial(win, mouse, circle, target_pos, data_outlet, marker_outlet, trial_number, practice=False):
    # Practice trials emit a distinct 'mouse_practice_trial_*' marker so the real
    # targeting analysis (which keys on 'mouse_trial_start') ignores them automatically.
    prefix = "mouse_practice_trial" if practice else "mouse_trial"
    circle.setPos(target_pos)
    # Persistent fixation cross, kept on screen through the whole reach (so the cross
    # is continuous across both mouse blocks; in the Go/No-Go block it is also the
    # redirect target). Same look as the pre-trial show_fixation cross.
    cross = visual.TextStim(win, text="+", units="norm", pos=(0, 0), height=0.12,
                            color="white", font="Arial")
    # Recenter cursor at trial start so every target begins from the same origin.
    mouse.setPos((0.0, 0.0))
    mouse.getRel()  # clear relative-motion accumulator after recentering
    mouse.clickReset()
    mouse.mouseClock.reset()

    onset_marker_scheduled = False
    prev_buttons = [bool(b) for b in mouse.getPressed()]
    trial_clock = core.Clock()
    trial_start_time = core.getTime()

    while True:
        if not onset_marker_scheduled:
            win.callOnFlip(
                safe_push,
                marker_outlet,
                (
                    f"{prefix}_start,trial:{trial_number},"
                    f"target_x:{target_pos[0]:.4f},target_y:{target_pos[1]:.4f}"
                ),
            )
            onset_marker_scheduled = True

        cross.draw()
        circle.draw()

        push_mouse_sample(data_outlet, mouse)
        curr_x, curr_y = mouse.getPos()
        buttons = mouse.getPressed()
        left_rising = bool(buttons[0]) and not prev_buttons[0]
        prev_buttons = [bool(b) for b in buttons]

        win.flip()

        if left_rising and circle.contains(mouse):
            rt = trial_clock.getTime()
            safe_push(
                marker_outlet,
                (
                    f"{prefix}_end,trial:{trial_number},success:1,rt:{rt:.4f},"
                    f"click_x:{curr_x:.4f},click_y:{curr_y:.4f}"
                ),
            )
            return {
                "success": 1,
                "rt": rt,
                "click_x": float(curr_x),
                "click_y": float(curr_y),
                "timeout": False,
            }

        if (core.getTime() - trial_start_time) > MOUSE_TRIAL_CLICK_TIMEOUT_S:
            safe_push(
                marker_outlet,
                (
                    f"{prefix}_end,trial:{trial_number},success:0,timeout:1,"
                    f"rt:{MOUSE_TRIAL_CLICK_TIMEOUT_S:.3f}"
                ),
            )
            return {
                "success": False,
                "rt": float("nan"),
                "click_x": None,
                "click_y": None,
                "timeout": True,
            }

        if "escape" in event.getKeys(keyList=["escape"]):
            safe_push(marker_outlet, f"{prefix}_end,trial:{trial_number},success:0,aborted:1")
            raise KeyboardInterrupt


def gonogo_luminance_sequence(n_total, n_white):
    """Shuffled luminances: n_white 'white' (no-go) + the rest 'grey' (go)."""
    n_white = max(0, min(int(n_white), int(n_total)))
    seq = ["white"] * n_white + ["grey"] * (int(n_total) - n_white)
    random.shuffle(seq)
    return seq


def run_gonogo_trial(win, mouse, circle, cross_hit_radius, luminance, target_pos,
                     data_outlet, marker_outlet, trial_number, practice=False):
    """Redirect Go/No-Go trial. A circle appears at target_pos in one of two
    luminances while the central fixation cross stays visible:
      - GREY  circle -> click the circle  (go)
      - WHITE circle -> click the centre cross instead  (no-go / redirect)
    Self-paced: ends on the first left-click landing on the circle OR within
    cross_hit_radius of centre. correct = clicked the rule-appropriate target.
    """
    prefix = "gonogo_practice_trial" if practice else "gonogo_trial"
    if luminance == "white":
        circle.fillColor = "white"
        circle.lineColor = "white"
    else:
        circle.fillColor = "grey"
        circle.lineColor = "grey"
    circle.setPos(target_pos)
    cross = visual.TextStim(win, text="+", units="norm", pos=(0, 0), height=0.12,
                            color="white", font="Arial")
    mouse.setPos((0.0, 0.0))
    mouse.getRel()
    mouse.clickReset()

    onset_scheduled = False
    prev_buttons = [bool(b) for b in mouse.getPressed()]
    trial_clock = core.Clock()
    trial_start_time = core.getTime()

    while True:
        if not onset_scheduled:
            win.callOnFlip(
                safe_push,
                marker_outlet,
                (
                    f"{prefix}_start,trial:{trial_number},luminance:{luminance},"
                    f"target_x:{target_pos[0]:.4f},target_y:{target_pos[1]:.4f}"
                ),
            )
            onset_scheduled = True

        cross.draw()
        circle.draw()

        push_mouse_sample(data_outlet, mouse)
        curr_x, curr_y = mouse.getPos()
        buttons = mouse.getPressed()
        left_rising = bool(buttons[0]) and not prev_buttons[0]
        prev_buttons = [bool(b) for b in buttons]

        win.flip()

        if left_rising:
            on_circle = bool(circle.contains(mouse))
            on_cross = (math.hypot(curr_x, curr_y) <= cross_hit_radius)
            if on_circle or on_cross:
                clicked = "circle" if on_circle else "cross"
                rt = trial_clock.getTime()
                correct = 1 if (
                    (luminance == "grey" and clicked == "circle")
                    or (luminance == "white" and clicked == "cross")
                ) else 0
                safe_push(
                    marker_outlet,
                    (
                        f"{prefix}_end,trial:{trial_number},luminance:{luminance},"
                        f"clicked:{clicked},correct:{correct},rt:{rt:.4f},"
                        f"click_x:{curr_x:.4f},click_y:{curr_y:.4f}"
                    ),
                )
                return {
                    "luminance": luminance, "clicked": clicked, "correct": correct,
                    "rt": rt, "click_x": float(curr_x), "click_y": float(curr_y),
                    "timeout": False,
                }

        if (core.getTime() - trial_start_time) > MOUSE_TRIAL_CLICK_TIMEOUT_S:
            safe_push(
                marker_outlet,
                (
                    f"{prefix}_end,trial:{trial_number},luminance:{luminance},"
                    f"clicked:none,correct:0,timeout:1,rt:{MOUSE_TRIAL_CLICK_TIMEOUT_S:.3f}"
                ),
            )
            return {
                "luminance": luminance, "clicked": "none", "correct": 0,
                "rt": float("nan"), "click_x": None, "click_y": None, "timeout": True,
            }

        if "escape" in event.getKeys(keyList=["escape"]):
            safe_push(marker_outlet, f"{prefix}_end,trial:{trial_number},aborted:1")
            raise KeyboardInterrupt


def build_movie_stim(win, video_file, audio_on):
    movie_cls = getattr(visual, "MovieStim3", None)
    if movie_cls is None:
        movie_cls = getattr(visual, "MovieStim", None)
    if movie_cls is None:
        raise RuntimeError("No MovieStim class found in this PsychoPy version.")

    # PsychoPy 2026.1.x can error if size is left as None in MovieStim init.
    # Passing size=(None, None) preserves native movie size without triggering that bug.
    common_kwargs = {
        "loop": False,
        "units": "pix",
        "size": (None, None),
        "noAudio": (not audio_on),
        "autoStart": False,
    }
    compat_kwargs = {"loop": False, "units": "pix", "size": (None, None), "noAudio": (not audio_on)}
    minimal_kwargs = {"loop": False, "size": (None, None), "noAudio": (not audio_on)}
    attempts = [
        lambda: movie_cls(win, filename=video_file, **common_kwargs),
        lambda: movie_cls(win, video_file, **common_kwargs),
        lambda: movie_cls(win, filename=video_file, **compat_kwargs),
        lambda: movie_cls(win, video_file, **compat_kwargs),
        lambda: movie_cls(win, filename=video_file, **minimal_kwargs),
        lambda: movie_cls(win, video_file, **minimal_kwargs),
    ]

    last_error = None
    for make in attempts:
        try:
            return make()
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Unable to initialize movie stimulus: {last_error}")


def movie_finished(movie):
    try:
        if hasattr(movie, "isFinished"):
            return bool(movie.isFinished)
    except Exception:
        pass

    try:
        return getattr(movie, "status", None) == FINISHED
    except Exception:
        return False


def fit_movie_to_window(movie, win):
    """Letterbox movie to fit inside the current window while preserving aspect."""
    try:
        frame_size = getattr(movie, "frameSize", None)
        if not frame_size:
            frame_size = getattr(movie, "size", None)
        if not frame_size:
            return

        frame_w, frame_h = float(frame_size[0]), float(frame_size[1])
        win_w, win_h = float(win.size[0]), float(win.size[1])
        if frame_w <= 0 or frame_h <= 0 or win_w <= 0 or win_h <= 0:
            return

        scale = min(win_w / frame_w, win_h / frame_h)
        draw_size = (frame_w * scale, frame_h * scale)

        movie.units = "pix"
        movie.size = draw_size
        movie.pos = (0.0, 0.0)
    except Exception as exc:
        logging.warning(f"Movie layout fit failed: {exc}")


def hold_movie_at_start(movie):
    """Force movie into a paused frame-0 state until explicit play() call."""
    for control in ("stop", "pause"):
        if hasattr(movie, control):
            try:
                getattr(movie, control)()
            except Exception:
                pass
    if hasattr(movie, "seek"):
        try:
            movie.seek(0.0)
        except Exception:
            pass
    if hasattr(movie, "pause"):
        try:
            movie.pause()
        except Exception:
            pass


def force_movie_audio_mute(movie):
    """Best-effort hard mute across MovieStim variants and backends."""
    if movie is None:
        return

    for attr_name, attr_value in (("_noAudio", True), ("noAudio", True), ("muted", True)):
        if hasattr(movie, attr_name):
            try:
                setattr(movie, attr_name, attr_value)
            except Exception:
                pass

    if hasattr(movie, "setMuted"):
        try:
            movie.setMuted(True)
        except Exception:
            pass

    if hasattr(movie, "setVolume"):
        try:
            movie.setVolume(0.0)
        except Exception:
            pass

    if hasattr(movie, "volume"):
        try:
            movie.volume = 0.0
        except Exception:
            pass

    player = getattr(movie, "_player", None)
    if player is None:
        return

    if hasattr(player, "mute"):
        try:
            mute_attr = getattr(player, "mute")
            if callable(mute_attr):
                try:
                    mute_attr(True)
                except TypeError:
                    mute_attr()
            else:
                setattr(player, "mute", True)
        except Exception:
            pass

    if hasattr(player, "setVolume"):
        try:
            player.setVolume(0.0)
        except Exception:
            pass

    if hasattr(player, "volume"):
        try:
            player.volume = 0.0
        except Exception:
            pass


def play_video_with_opencv(
    win,
    video_file,
    marker_outlet,
    video_id,
    tick_interval_s=1.0,
    max_duration_s=None,
):
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError(f"OpenCV import failed: {exc}")

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise RuntimeError("OpenCV could not open video file.")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 60.0
    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    if max_duration_s is None:
        if frame_count > 0 and fps > 0:
            max_duration_s = frame_count / fps
        else:
            max_duration_s = DEFAULT_OPENCV_GUARD_S
    max_duration_s = max(1.0, float(max_duration_s))
    frame_interval_s = 1.0 / fps

    start_clock = core.Clock()
    next_frame_t = 0.0
    last_tick = 0.0
    elapsed = 0.0
    frame_count = 0
    image_stim = None

    try:
        while True:
            if "escape" in event.getKeys(keyList=["escape"]):
                raise KeyboardInterrupt

            elapsed = start_clock.getTime()
            if elapsed > (max_duration_s + 1.0):
                raise RuntimeError(
                    f"OpenCV playback guard timeout at {elapsed:.3f}s (guard={max_duration_s:.3f}s)."
                )
            if elapsed < next_frame_t:
                core.wait(min(0.001, next_frame_t - elapsed))
                continue

            ok, frame = cap.read()
            if not ok:
                break

            # Keep native 8-bit RGB values to avoid accidental clipping/posterization.
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if image_stim is None:
                frame_h, frame_w = frame_rgb.shape[:2]
                win_w, win_h = float(win.size[0]), float(win.size[1])
                scale = min(win_w / max(1.0, float(frame_w)), win_h / max(1.0, float(frame_h)))
                draw_size = (frame_w * scale, frame_h * scale)
                image_stim = visual.ImageStim(
                    win=win,
                    image=frame_rgb,
                    units="pix",
                    size=draw_size,
                    interpolate=True,
                    flipVert=True,
                )
            else:
                image_stim.image = frame_rgb

            image_stim.draw()
            win.flip()

            elapsed = start_clock.getTime()
            while elapsed >= (last_tick + tick_interval_s):
                last_tick += tick_interval_s
                safe_push(
                    marker_outlet,
                    f"video_tick,id:{video_id},audio:off,t:{last_tick:.3f}",
                )

            frame_count += 1
            next_frame_t = frame_count * frame_interval_s
    finally:
        cap.release()

    return float(start_clock.getTime())


def normalize_video_key(value):
    if not value:
        return ""
    cleaned = str(value).lower()
    for token in ("1080p", "720p", "480p"):
        cleaned = cleaned.replace(token, "")
    return "".join(ch for ch in cleaned if ch.isalnum())


def discover_video_files(search_roots):
    discovered = []
    seen = set()
    for root in search_roots:
        if not root.exists() or not root.is_dir():
            continue
        for ext in VIDEO_FILE_EXTENSIONS:
            for candidate in sorted(root.glob(f"*{ext}")):
                resolved = candidate.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                discovered.append(resolved)
    return discovered


def resolve_video_path(item, search_roots, discovered_files):
    explicit_path = str(item.get("path", "")).strip()
    if explicit_path:
        candidate = Path(explicit_path).expanduser()
        if candidate.exists():
            return candidate.resolve()

    raw_keys = []
    for key_name in ("filename", "file_key", "id"):
        value = str(item.get(key_name, "")).strip()
        if value:
            raw_keys.append(value)

    for key in raw_keys:
        key_path = Path(key)
        has_extension = bool(key_path.suffix)

        for root in search_roots:
            if has_extension:
                candidate = (root / key_path).resolve()
                if candidate.exists():
                    return candidate
            else:
                for ext in VIDEO_FILE_EXTENSIONS:
                    candidate = (root / f"{key}{ext}").resolve()
                    if candidate.exists():
                        return candidate

    normalized_keys = [normalize_video_key(Path(key).stem) for key in raw_keys]
    normalized_keys = [key for key in normalized_keys if key]
    if not normalized_keys:
        return None

    for candidate in discovered_files:
        candidate_key = normalize_video_key(candidate.stem)
        if any(key and key in candidate_key for key in normalized_keys):
            return candidate

    return None


def resolve_video_playlist(this_dir):
    project_dir = Path(this_dir).resolve()
    search_roots = []

    env_video_dir = os.environ.get("PSYCHOPY_VIDEO_DIR", "").strip()
    if env_video_dir:
        search_roots.append(Path(env_video_dir).expanduser().resolve())

    search_roots.append(project_dir / "videos")
    search_roots.append(project_dir.parent / "data" / "Videos")
    search_roots.append(project_dir)

    discovered_files = discover_video_files(search_roots)
    playlist = []
    for item in TEST_VIDEO_CANDIDATES:
        resolved_path = resolve_video_path(item, search_roots, discovered_files)
        if resolved_path is not None:
            playlist.append(
                {
                    "id": item.get("id", resolved_path.stem),
                    "path": str(resolved_path),
                    "condition": item.get("condition", "unspecified"),
                }
            )
    return playlist


def select_video_jobs(video_jobs, selectors):
    """Select video jobs by id/stem/filename tokens while preserving selector order."""
    if not selectors:
        return list(video_jobs)

    picked = []
    used_indices = set()
    lowered = [str(token).strip().lower() for token in selectors if str(token).strip()]
    for token in lowered:
        for idx, job in enumerate(video_jobs):
            if idx in used_indices:
                continue
            job_id = str(job.get("id", "")).lower()
            stem = Path(str(job.get("path", ""))).stem.lower()
            filename = Path(str(job.get("path", ""))).name.lower()
            if token in {job_id, stem, filename} or token in job_id or token in stem or token in filename:
                picked.append(job)
                used_indices.add(idx)
                break
    return picked


def resolve_external_audio_track(video_file):
    """Resolve sidecar audio file for a video if present."""
    video_path = Path(video_file).resolve()
    candidates = []
    for suffix in EXTERNAL_AUDIO_SUFFIXES:
        candidates.append(video_path.with_name(f"{video_path.stem}{suffix}"))
    candidates.append(video_path.with_suffix(".wav"))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def get_external_av_video_delay_s():
    raw = os.environ.get("PSYCHOPY_AV_SYNC_VIDEO_DELAY_S")
    if raw is None:
        return DEFAULT_EXTERNAL_AV_VIDEO_DELAY_S
    try:
        val = float(raw)
    except Exception:
        return DEFAULT_EXTERNAL_AV_VIDEO_DELAY_S
    # Keep within a bounded range while allowing large diagnostic offsets.
    return max(MIN_AV_VIDEO_DELAY_S, min(MAX_AV_VIDEO_DELAY_S, val))


# Set by ensure_audio_device() when the DEFAULT speaker fails to open but a working
# device is found by index; passed to every Sound so the video sidecar audio uses
# the same recovered device. None = use PsychoPy's default (the normal case).
SELECTED_SPEAKER = None


def build_external_audio_stim(audio_file):
    """Load a sidecar audio track using PsychoPy sound backend (PTB if available)."""
    kwargs = {"stereo": True, "hamming": False}
    if SELECTED_SPEAKER is not None:
        kwargs["speaker"] = SELECTED_SPEAKER
    return sound.Sound(str(audio_file), **kwargs)


def ensure_audio_device():
    """Verify a usable audio output exists. Returns (ok: bool, note: str, exc).

    PsychoPy 2026's PTB backend resolves the DEFAULT speaker by NAME, which can
    fail to OPEN a device that plainly exists (e.g. 'Speakers (Realtek(R) Audio)').
    So: try the default; if it fails, try every enumerated device explicitly BY
    INDEX and, on the first that opens, remember it in SELECTED_SPEAKER so the
    video sidecar audio uses that same working device.
    """
    global SELECTED_SPEAKER
    try:
        _s = sound.Sound(value=440, secs=0.05, stereo=True)
        try:
            _s.stop()
        except Exception:
            pass
        return True, "default device", None
    except BaseException as first_err:  # DeviceNotConnectedError is a BaseException
        if isinstance(first_err, (KeyboardInterrupt, SystemExit)):
            raise
    try:
        from psychopy.hardware.speaker import SpeakerDevice
        devices = SpeakerDevice.getAvailableDevices()
    except BaseException:
        devices = []
    for d in devices:
        try:
            idx = int(d.get("index"))
        except (TypeError, ValueError, AttributeError):
            continue
        try:
            sp = SpeakerDevice(index=idx)
            _s = sound.Sound(value=440, secs=0.05, stereo=True, speaker=sp)
            try:
                _s.stop()
            except Exception:
                pass
            SELECTED_SPEAKER = sp
            return True, f"recovered by index #{idx} ({d.get('name', '?')})", None
        except BaseException as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            continue
    return False, "no openable audio output device", first_err


def _build_error_report(stage, exc, extra=None):
    """Assemble a self-contained, copy-pasteable error report for the operator."""
    from datetime import datetime as _dt
    import traceback as _tb
    lines = [
        "=== IQS EXPERIMENT ERROR REPORT (copy this whole block for technical support) ===",
        f"When:   {_dt.now():%Y-%m-%d %H:%M:%S}",
        f"Stage:  {stage}",
    ]
    for key, value in (extra or {}).items():
        lines.append(f"{key}: {value}")
    lines.append(f"Error:  {type(exc).__name__}: {exc}")
    try:
        import psychopy as _pp
        lines.append(f"PsychoPy {_pp.__version__} | Python {sys.version.split()[0]} | {sys.platform}")
    except Exception:
        pass
    lines.append("--- Traceback ---")
    lines.append(
        "".join(_tb.format_exception(type(exc), exc, getattr(exc, "__traceback__", None))).rstrip()
    )
    lines.append("=== END REPORT ===")
    return "\n".join(lines)


def present_fatal_error(stage, exc, marker_outlet=None, plain_explanation="", extra=None):
    """Surface an unrecoverable error to the operator clearly, then return.

    The caller is responsible for stopping the session afterwards. This writes a
    full report to experiment/data/ERROR_<timestamp>.txt, copies it to the
    clipboard (best effort), prints it, and shows a dialog the operator cannot
    miss. Used instead of letting the task die with a silently-closing window,
    e.g. when no audio output device is available (a BaseException that ordinary
    'except Exception' handlers cannot catch)."""
    from datetime import datetime as _dt
    # Assemble the report defensively: this is the last-resort handler, so it must
    # never raise (e.g. on an exception object with a broken __str__/__repr__) and
    # thereby mask the original error.
    try:
        report = _build_error_report(stage, exc, extra)
    except BaseException:
        try:
            _name = type(exc).__name__
        except Exception:
            _name = "UnknownError"
        report = (
            "=== IQS EXPERIMENT ERROR REPORT ===\n"
            f"Stage:  {stage}\n"
            f"Error:  {_name} (report assembly failed; exception could not be stringified)\n"
            "=== END REPORT ==="
        )
    if marker_outlet is not None:
        try:
            safe_push(marker_outlet, f"experiment_end,aborted:1,stage:{stage},fatal:{type(exc).__name__}")
        except Exception:
            pass
    saved_path = None
    try:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        os.makedirs(data_dir, exist_ok=True)
        saved_path = os.path.join(data_dir, f"ERROR_{_dt.now():%Y%m%d_%H%M%S}.txt")
        with open(saved_path, "w", encoding="utf-8") as handle:
            handle.write(report + "\n")
    except Exception:
        saved_path = None
    clipboard_ok = False
    try:
        completed = subprocess.run(
            ["clip"], input=report, text=True, timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        clipboard_ok = completed.returncode == 0
    except Exception:
        clipboard_ok = False
    try:
        print("\n" + report + "\n")
    except Exception:
        pass
    try:
        dlg = gui.Dlg(title="Session stopped - please read")
        if plain_explanation:
            for para in plain_explanation.strip().split("\n"):
                dlg.addText(para)
            dlg.addText("")
        dlg.addText(f"Error: {type(exc).__name__}: {exc}")
        dlg.addText("")
        if saved_path:
            dlg.addText("A full report was saved to:")
            dlg.addText(saved_path)
        if clipboard_ok:
            dlg.addText("(The report was also copied to the clipboard.)")
        dlg.addText("")
        dlg.addText("Copy that report (or open the file) and provide it to technical support.")
        dlg.show()
    except Exception:
        pass
    return report


def play_video_job(win, marker_outlet, this_exp, video_job, av_video_delay_s):
    """Play one video job and persist trial data."""
    video_id = str(video_job.get("id", "video")).replace(",", ";")
    video_file = video_job["path"]
    video_file_label = os.path.basename(video_file).replace(",", ";")
    video_condition = str(video_job.get("condition", "unspecified")).replace(",", ";")
    audio_label = "on"
    audio_mode = "movie_embedded"
    audio_file_label = ""
    video_backend = "moviestim"
    movie = None
    external_audio = None
    external_audio_duration_s = None
    elapsed = 0.0
    had_error = False
    use_opencv_fallback = False
    use_opencv_fallback_after_draw_error = False
    movie_audio_on = True

    if prefer_external_audio_sync():
        external_audio_path = resolve_external_audio_track(video_file)
        if external_audio_path is not None:
            try:
                external_audio = build_external_audio_stim(external_audio_path)
                movie_audio_on = False
                audio_mode = "external_sidecar"
                audio_file_label = os.path.basename(str(external_audio_path)).replace(",", ";")
                if hasattr(external_audio, "getDuration"):
                    try:
                        external_audio_duration_s = float(external_audio.getDuration())
                        if (not math.isfinite(external_audio_duration_s)) or external_audio_duration_s <= 0:
                            external_audio_duration_s = None
                    except Exception:
                        external_audio_duration_s = None
            except Exception as exc:
                err = str(exc).replace(",", ";")
                logging.error(f"Failed to init external audio ({video_id}): {exc}")
                safe_push(
                    marker_outlet,
                    (
                        f"video_error,stage:audio_init,id:{video_id},audio:on,"
                        f"condition:{video_condition},error:{err}"
                    ),
                )
                external_audio = None
                movie_audio_on = True
                audio_mode = "movie_embedded"

    try:
        movie = build_movie_stim(win, video_file, audio_on=movie_audio_on)
    except Exception as exc:
        # If external sidecar mode is enabled and movie init fails, retry with embedded audio.
        if not movie_audio_on:
            try:
                if external_audio is not None:
                    external_audio.stop()
                external_audio = None
            except Exception:
                external_audio = None
            movie_audio_on = True
            audio_mode = "movie_embedded"
            audio_file_label = ""
            try:
                movie = build_movie_stim(win, video_file, audio_on=True)
            except Exception as exc_retry:
                err = str(exc_retry).replace(",", ";")
                logging.error(f"Failed to initialize movie ({video_id}, on): {exc_retry}")
                safe_push(
                    marker_outlet,
                    (
                        f"video_error,stage:init,id:{video_id},audio:on,"
                        f"condition:{video_condition},error:{err}"
                    ),
                )
                if ALLOW_OPENCV_FALLBACK_ON_INIT_ERROR:
                    safe_push(marker_outlet, f"video_fallback,id:{video_id},backend:opencv,audio:off")
                    use_opencv_fallback = True
                    audio_label = "off"
                    audio_mode = "none"
                    video_backend = "opencv"
                else:
                    had_error = True
        else:
            err = str(exc).replace(",", ";")
            logging.error(f"Failed to initialize movie ({video_id}, on): {exc}")
            safe_push(
                marker_outlet,
                (
                    f"video_error,stage:init,id:{video_id},audio:on,"
                    f"condition:{video_condition},error:{err}"
                ),
            )
            if ALLOW_OPENCV_FALLBACK_ON_INIT_ERROR:
                safe_push(marker_outlet, f"video_fallback,id:{video_id},backend:opencv,audio:off")
                use_opencv_fallback = True
                audio_label = "off"
                audio_mode = "none"
                video_backend = "opencv"
            else:
                had_error = True

    video_start_marker = (
        f"video_start,id:{video_id},audio:{audio_label},backend:{video_backend},"
        f"condition:{video_condition},delay_s:{float(av_video_delay_s):.3f},file:{video_file_label}"
    )
    # In OpenCV fallback mode there may be no immediate flip on init errors.
    # Push directly so marker chronology remains complete.
    if use_opencv_fallback:
        safe_push(marker_outlet, video_start_marker)
    else:
        win.callOnFlip(safe_push, marker_outlet, video_start_marker)
        safe_push(
            marker_outlet,
            (
                f"config,video_playback_mode,id:{video_id},audio_mode:{audio_mode},"
                f"delay_s:{float(av_video_delay_s):.3f}"
            ),
        )
        safe_push(
            marker_outlet,
            "config,video_draw_error_opencv_fallback_enabled:"
            f"{1 if ALLOW_OPENCV_FALLBACK_AFTER_DRAW_ERROR else 0}",
        )

    if use_opencv_fallback:
        try:
            elapsed = play_video_with_opencv(
                win=win,
                video_file=video_file,
                marker_outlet=marker_outlet,
                video_id=video_id,
                tick_interval_s=1.0,
                max_duration_s=DEFAULT_OPENCV_GUARD_S,
            )
        except Exception as exc:
            had_error = True
            err = str(exc).replace(",", ";")
            logging.error(f"OpenCV video fallback failed ({video_id}): {exc}")
            safe_push(
                marker_outlet,
                (
                    f"video_error,stage:opencv,id:{video_id},audio:{audio_label},"
                    f"condition:{video_condition},error:{err}"
                ),
            )
    else:
        fit_movie_to_window(movie, win)
        hold_movie_at_start(movie)
        if external_audio is not None:
            force_movie_audio_mute(movie)
        elif abs(float(av_video_delay_s)) > 0.0005:
            logging.warning(
                f"Requested delay {float(av_video_delay_s):.3f}s ignored for {video_id}: "
                "no external sidecar audio was loaded."
            )
            safe_push(
                marker_outlet,
                (
                    f"video_warning,id:{video_id},reason:delay_ignored_no_external_audio,"
                    f"requested_delay_s:{float(av_video_delay_s):.3f}"
                ),
            )
        start_clock = core.Clock()
        last_tick = 0.0
        movie_start_scheduled = False
        movie_start_elapsed = None
        first_frame_marked = False
        audio_start_scheduled = False
        external_audio_started = False
        audio_start_elapsed = None

        # Fallback guard if this backend does not expose a reliable end signal.
        raw_duration = None
        try:
            raw_duration = float(getattr(movie, "duration"))
        except Exception:
            raw_duration = None
        if raw_duration is not None and (not math.isfinite(raw_duration) or raw_duration <= 0):
            raw_duration = None
        max_duration = (
            min(float(raw_duration), DEFAULT_MOVIE_GUARD_S)
            if raw_duration is not None
            else DEFAULT_MOVIE_GUARD_S
        )

        # Solid black backdrop behind the video so the surround is black for EVERY
        # video (some sources don't fill the frame / have non-black padding - e.g.
        # Olaplex showed the grey window colour). Covers the whole window.
        video_bg = visual.Rect(win, width=2.0, height=2.0, units="norm",
                               fillColor="black", lineColor=None)

        while True:
            video_bg.draw()
            # Some PsychoPy movie backends can throw duration-related exceptions
            # when draw() is called before the movie has actually started.
            if movie_start_scheduled:
                try:
                    movie.draw()
                    if not first_frame_marked:
                        # Mark the TRUE first-video-frame onset: callOnFlip fires on THIS flip -
                        # the flip that presents the first drawn movie frame (captures any
                        # first-draw texture-upload delay). video_start above marks the earlier
                        # black-backdrop flip and is intentionally left untouched.
                        win.callOnFlip(safe_push, marker_outlet,
                                       f"movie_frame_on,id:{video_id}")
                        first_frame_marked = True
                except Exception as exc:
                    elapsed_now = start_clock.getTime()
                    err_raw = str(exc)
                    err = err_raw.replace(",", ";")
                    err_lc = err_raw.lower()
                    is_duration_none_error = "nonetype" in err_lc and "duration" in err_lc
                    is_eof_duration_glitch = (
                        raw_duration is not None
                        and movie_start_elapsed is not None
                        and is_duration_none_error
                        and (elapsed_now - movie_start_elapsed) >= max(0.0, raw_duration - 0.750)
                    )
                    # If duration metadata is unavailable, but playback has already
                    # advanced at least ~1s, treat this specific error as EOF-like.
                    is_probable_eof_without_duration = (
                        raw_duration is None
                        and movie_start_elapsed is not None
                        and is_duration_none_error
                        and last_tick >= 1.0
                        and (elapsed_now - movie_start_elapsed) >= 1.0
                    )
                    if is_eof_duration_glitch or is_probable_eof_without_duration:
                        dur_txt = f"{raw_duration:.3f}" if raw_duration is not None else "unknown"
                        logging.warning(
                            "MovieStim draw duration glitch treated as finished "
                            f"({video_id}, elapsed={elapsed_now:.3f}, duration={dur_txt}): {exc}"
                        )
                        safe_push(
                            marker_outlet,
                            (
                                f"video_warning,id:{video_id},reason:eof_draw_exception_treated_finished,"
                                f"elapsed:{elapsed_now:.3f},duration_s:{dur_txt},error:{err}"
                            ),
                        )
                        elapsed = elapsed_now
                        break

                    had_error = True
                    logging.error(f"Video draw failed ({video_id}, {audio_label}): {exc}")
                    safe_push(
                        marker_outlet,
                        (
                            f"video_error,stage:draw,id:{video_id},audio:{audio_label},"
                            f"condition:{video_condition},error:{err}"
                        ),
                    )
                    if ALLOW_OPENCV_FALLBACK_AFTER_DRAW_ERROR:
                        use_opencv_fallback_after_draw_error = True
                    else:
                        safe_push(
                            marker_outlet,
                            (
                                f"video_warning,id:{video_id},reason:draw_error_no_opencv_fallback,"
                                f"elapsed:{elapsed_now:.3f}"
                            ),
                        )
                    break

            if external_audio is not None and not audio_start_scheduled:
                try:
                    # PTB-timed start aligned to the next screen refresh.
                    when_ptb = win.getFutureFlipTime(clock="ptb")
                    external_audio.play(when=when_ptb)
                    audio_start_scheduled = True
                except Exception:
                    win.callOnFlip(external_audio.play)
                    audio_start_scheduled = True
                    external_audio_started = True
                    audio_start_elapsed = 0.0
            elif external_audio is None and not movie_start_scheduled:
                if hasattr(movie, "play"):
                    win.callOnFlip(movie.play)
                movie_start_scheduled = True
                movie_start_elapsed = start_clock.getTime()

            win.flip()

            elapsed = start_clock.getTime()
            if external_audio is not None and audio_start_scheduled and not external_audio_started:
                external_audio_started = True
                audio_start_elapsed = elapsed

            if external_audio is not None and not movie_start_scheduled and external_audio_started:
                video_delay_elapsed = elapsed - (audio_start_elapsed or 0.0)
                if video_delay_elapsed >= av_video_delay_s:
                    force_movie_audio_mute(movie)
                    if hasattr(movie, "play"):
                        win.callOnFlip(force_movie_audio_mute, movie)
                        win.callOnFlip(movie.play)
                    movie_start_scheduled = True
                    movie_start_elapsed = start_clock.getTime()

            while elapsed >= (last_tick + 1.0):
                last_tick += 1.0
                safe_push(
                    marker_outlet,
                    f"video_tick,id:{video_id},audio:{audio_label},t:{last_tick:.3f}",
                )

            if movie_finished(movie):
                break

            # PsychoPy MovieStim backends can occasionally miss the FINISHED flag at EOF.
            # End cleanly based on known duration to avoid a frozen last frame / apparent A/V drift.
            if raw_duration is not None and movie_start_elapsed is not None:
                if (elapsed - movie_start_elapsed) >= (raw_duration + 0.250):
                    safe_push(
                        marker_outlet,
                        (
                            f"video_warning,id:{video_id},reason:forced_end_video_duration,"
                            f"elapsed:{elapsed:.3f},duration_s:{raw_duration:.3f}"
                        ),
                    )
                    break

            if external_audio_duration_s is not None and audio_start_elapsed is not None:
                if elapsed >= (audio_start_elapsed + external_audio_duration_s + 0.250):
                    safe_push(
                        marker_outlet,
                        (
                            f"video_warning,id:{video_id},reason:forced_end_audio_duration,"
                            f"elapsed:{elapsed:.3f},audio_duration_s:{external_audio_duration_s:.3f}"
                        ),
                    )
                    break

            if elapsed > (max_duration + 1.0):
                had_error = True
                safe_push(
                    marker_outlet,
                    (
                        f"video_warning,id:{video_id},reason:playback_guard_timeout,"
                        f"elapsed:{elapsed:.3f},guard_s:{max_duration:.3f}"
                    ),
                )
                break

            if "escape" in event.getKeys(keyList=["escape"]):
                raise KeyboardInterrupt

        for stopper in ("stop", "pause"):
            if hasattr(movie, stopper):
                try:
                    getattr(movie, stopper)()
                except Exception:
                    pass
        if external_audio is not None:
            # Stop HARD: a draw error can break the loop before a scheduled play()
            # actually starts (PTB when= / callOnFlip), so the sidecar would otherwise
            # play its full length in the background over the rating page. Stop, then
            # fire any pending callOnFlip with a blank flip, then stop again.
            try:
                external_audio.stop()
            except Exception:
                pass
            try:
                video_bg.draw()
                win.flip()
            except Exception:
                pass
            try:
                external_audio.stop()
            except Exception:
                pass

        if use_opencv_fallback_after_draw_error:
            audio_label = "off"
            audio_mode = "none"
            safe_push(marker_outlet, f"video_fallback,id:{video_id},backend:opencv,audio:off")
            video_backend = "opencv"
            try:
                elapsed = play_video_with_opencv(
                    win=win,
                    video_file=video_file,
                    marker_outlet=marker_outlet,
                    video_id=video_id,
                    tick_interval_s=1.0,
                    max_duration_s=DEFAULT_OPENCV_GUARD_S,
                )
            except Exception as exc:
                err = str(exc).replace(",", ";")
                logging.error(f"OpenCV fallback after draw error failed ({video_id}): {exc}")
                safe_push(
                    marker_outlet,
                    (
                        f"video_error,stage:opencv_after_draw,id:{video_id},audio:{audio_label},"
                        f"condition:{video_condition},error:{err}"
                    ),
                )

    safe_push(
        marker_outlet,
        (
            f"video_end,id:{video_id},audio:{audio_label},condition:{video_condition},"
            f"elapsed:{elapsed:.3f}"
        ),
    )

    this_exp.addData("video_id", video_id)
    this_exp.addData("video_condition", video_condition)
    this_exp.addData("video_file", video_file)
    this_exp.addData("video_audio", audio_label)
    this_exp.addData("video_audio_mode", audio_mode)
    this_exp.addData("video_audio_file", audio_file_label)
    this_exp.addData("video_backend", video_backend)
    this_exp.addData("video_elapsed", elapsed)
    this_exp.addData("video_error", int(had_error))
    # NOTE: caller (run_video_block) is responsible for nextEntry(), so that
    # the post-video rating responses are written to the same CSV row as
    # the video metadata above.

    return {
        "video_id": video_id,
        "video_file": video_file,
        "audio_mode": audio_mode,
        "elapsed": elapsed,
        "error": had_error,
    }


# Post-video rating items (v2). Each item carries its own fully-labelled scale.
# Designed against the literature in RATING_SCALE_DESIGN.md at the workspace
# root, which cites the primary sources for every choice below. Keep the
# instruments here, RATING_SCALE_DESIGN.md, and ethics Appendix E in sync.
RATING_SCALE_3PT_RECOG = [
    ("No", 0),
    ("Not sure", 1),
    ("Yes", 2),
]
# Bipolar evaluative liking scale, fully labelled (Hohne et al. 2022;
# Krosnick & Presser 2010). Codes negative for dislike so the mean per
# condition is a directly interpretable signed value.
RATING_SCALE_LIKING_7PT = [
    ("Strongly\ndisliked", -3),
    ("Disliked",            -2),
    ("Slightly\ndisliked",  -1),
    ("Neutral",              0),
    ("Slightly\nliked",      1),
    ("Liked",                2),
    ("Strongly\nliked",      3),
]
# Juster (1966) 11-point purchase probability scale; verbal anchor on top,
# implied probability label below. Day et al. (1991) meta-analysis confirms
# improved predictive validity over 5-point verbal scales.
RATING_SCALE_JUSTER_11PT = [
    ("0\n(1%)",  0),
    ("1\n(10%)", 1),
    ("2\n(20%)", 2),
    ("3\n(30%)", 3),
    ("4\n(40%)", 4),
    ("5\n(50%)", 5),
    ("6\n(60%)", 6),
    ("7\n(70%)", 7),
    ("8\n(80%)", 8),
    ("9\n(90%)", 9),
    ("10\n(99%)", 10),
]
# Single bipolar virality judgement (replaces the old binary + confidence pair):
# direction AND confidence on one signed 7-point scale (-3 = definitely not viral,
# +3 = definitely viral). One signed item = the old signed-confidence calibration.
RATING_SCALE_VIRALITY_7PT = [
    ("Definitely\nNOT viral", -3),
    ("Probably\nnot",         -2),
    ("Maybe\nnot",            -1),
    ("Unsure",                 0),
    ("Maybe\nviral",           1),
    ("Probably\nviral",        2),
    ("Definitely\nviral",      3),
]
RATING_SCALE_VIRALITY_BINARY = [
    ("No", 0),
    ("Yes", 1),
]
RATING_SCALE_CONFIDENCE_7PT = [
    ("Not at all", 1),
    ("Slightly",   2),
    ("Somewhat",   3),
    ("Quite",      4),
    ("Moderately", 5),
    ("Very",       6),
    ("Completely", 7),
]
# Kent & Allen (1994) brand familiarity scale.
RATING_SCALE_FAMILIARITY_7PT = [
    ("Not at all", 1),
    ("Slightly",   2),
    ("Somewhat",   3),
    ("Moderately", 4),
    ("Quite",      5),
    ("Very",       6),
    ("Extremely",  7),
]

# THE post-video question set - identical for EVERY video. One list, asked by the
# single run_post_video_rating() function each time, so the questionnaire never
# varies between videos. ('seen this brand before' was removed as a duplicate of
# 'how familiar are you with this brand'.)
POST_VIDEO_RATING_ITEMS = [
    {"key": "seen_video_before",   "prompt": "Have you seen this video before?",
     "scale": RATING_SCALE_3PT_RECOG},
    {"key": "liking",              "prompt": "How did you feel about this video?",
     "scale": RATING_SCALE_LIKING_7PT},
    {"key": "purchase_intent",     "prompt": "How likely are you to buy this product?   (0 = not at all, 10 = definitely would)",
     "scale": RATING_SCALE_JUSTER_11PT},
    {"key": "virality",            "prompt": "Do you think this video went viral on TikTok?",
     "scale": RATING_SCALE_VIRALITY_7PT},
    {"key": "brand_familiarity",   "prompt": "How familiar are you with this brand?",
     "scale": RATING_SCALE_FAMILIARITY_7PT},
]

RATING_BUTTON_COLOR_IDLE = [-0.55, -0.55, -0.50]
RATING_BUTTON_COLOR_SELECTED = [0.20, 0.30, 0.55]
RATING_BUTTON_LINE_IDLE = [0.30, 0.30, 0.40]
RATING_BUTTON_LINE_SELECTED = [0.95, 0.95, 0.95]


def _rounded_rect_vertices(width, height, radius=None, segments=5):
    """Vertices (centred on 0,0, height units) for a rounded rectangle, for use as
    a visual.ShapeStim so rating buttons have rounded rather than square corners.
    ShapeStim.contains() still works, so click detection is unchanged."""
    hw, hh = width / 2.0, height / 2.0
    if radius is None:
        radius = min(width, height) * 0.28
    radius = min(radius, hw, hh)
    centres = [
        (hw - radius, hh - radius, 0.0),       # top-right
        (-hw + radius, hh - radius, 90.0),     # top-left
        (-hw + radius, -hh + radius, 180.0),   # bottom-left
        (hw - radius, -hh + radius, 270.0),    # bottom-right
    ]
    verts = []
    for cx, cy, a0 in centres:
        for k in range(segments + 1):
            a = math.radians(a0 + 90.0 * k / segments)
            verts.append((cx + radius * math.cos(a), cy + radius * math.sin(a)))
    return verts


def _rating_make_button(win, label, pos, width, height, *, font_height=None):
    """Build a rounded button (ShapeStim) + TextStim pair for a clickable rating button."""
    label = str(label)
    if font_height is None:
        # Auto-fit: shrink the font so the longest line fits inside the button
        # width and all lines fit its height, so labels never spill outside the
        # box (the previous fixed size overflowed for words like "Moderately").
        lines = label.split("\n")
        n_lines = max(1, len(lines))
        longest = max((len(ln) for ln in lines), default=1)
        longest = max(1, longest)
        width_limit = (width * 0.88) / (0.55 * longest)   # ~0.55 em average advance
        height_limit = (height * 0.78) / n_lines
        font_height = max(0.011, min(0.024, width_limit, height_limit))
    rect = visual.ShapeStim(
        win=win,
        vertices=_rounded_rect_vertices(width, height),
        pos=pos,
        fillColor=RATING_BUTTON_COLOR_IDLE,
        lineColor=RATING_BUTTON_LINE_IDLE,
        lineWidth=1.5,
        units="height",
        closeShape=True,
        interpolate=True,
    )
    text = visual.TextStim(
        win=win,
        text=label,
        pos=pos,
        height=font_height,
        color="white",
        units="height",
        font="Arial",
        alignText="center",
        anchorHoriz="center",
        anchorVert="center",
    )
    return rect, text


def _rating_layout_row(item, y_center, x_left_edge=-0.85, x_right_skip=0.78):
    """Compute geometry for a single item row.

    Returns (prompt_pos, prompt_wrap, button_specs, skip_pos, skip_w, btn_h)
    where button_specs is a list of (label, value, (cx, cy), width, height).
    """
    scale = item["scale"]
    n = len(scale)

    # The longer the scale, the narrower the buttons must be.
    if n <= 3:
        btn_w, gap = 0.118, 0.018
    elif n <= 7:
        btn_w, gap = 0.086, 0.010
    else:  # 8-11
        btn_w, gap = 0.058, 0.006

    # Taller rows when ANY label wraps to two lines, so 2-line options aren't cramped.
    btn_h = 0.082 if any("\n" in lbl for lbl, _ in scale) else 0.064

    # Buttons live in the middle-right. Compute start x so the row right-edge
    # ends just left of the Skip button.
    total_w = n * btn_w + (n - 1) * gap
    skip_w = 0.10
    skip_cx = x_right_skip - skip_w / 2
    right_edge = skip_cx - skip_w / 2 - 0.020  # 0.020 gap before Skip
    start_cx = right_edge - total_w + btn_w / 2

    button_specs = []
    for k, (label, value) in enumerate(scale):
        cx = start_cx + k * (btn_w + gap)
        button_specs.append((label, value, (cx, y_center), btn_w, btn_h))

    # Prompt sits in the left margin.
    prompt_pos = (x_left_edge + 0.01, y_center)
    prompt_wrap = (start_cx - btn_w / 2) - (x_left_edge + 0.01) - 0.015
    skip_pos = (skip_cx, y_center)
    return prompt_pos, prompt_wrap, button_specs, skip_pos, skip_w, btn_h


DEMOGRAPHIC_KEYS = (
    "gender", "handedness", "vision_correction",
    "tiktok_use", "cosmetics_buy", "cosmetics_interest",
)

# Full question wording + concise button labels (the prompt carries the meaning).
DEMOGRAPHIC_QUESTIONS = [
    {"key": "gender", "prompt": "What is your gender?",
     "scale": [("Female", "female"), ("Male", "male"), ("Other", "other"), ("Prefer\nnot to say", "na")]},
    {"key": "handedness", "prompt": "Which hand do you use for the mouse?",
     "scale": [("Right", "right"), ("Left", "left"), ("Either", "ambi")]},
    {"key": "vision_correction", "prompt": "Are you wearing glasses or contact lenses right now?",
     "scale": [("Neither", "none"), ("Glasses", "glasses"), ("Contacts", "contacts")]},
    {"key": "tiktok_use", "prompt": "How often do you use TikTok?",
     "scale": [("Daily", "daily"), ("A few/\nweek", "weekly_plus"), ("Weekly", "weekly"), ("Rarely", "rarely"), ("Never", "never")]},
    {"key": "cosmetics_buy", "prompt": "How often do you buy cosmetics or make-up?",
     "scale": [("Weekly", "weekly"), ("Monthly", "monthly"), ("Every\nfew mo.", "few_months"), ("Rarely", "rarely"), ("Never", "never")]},
    {"key": "cosmetics_interest", "prompt": "How interested are you in cosmetics / make-up?   (1 = not at all, 7 = very)",
     "scale": [(str(i), i) for i in range(1, 8)]},
]


def _localize_items(items, lang):
    """Return items (demographic or rating) with prompt + button labels swapped to
    `lang`. The English structure (keys, values, order) is the canonical source; only
    the display text changes. Labels absent from the table pass through unchanged."""
    return [
        {
            "key": it["key"],
            "prompt": I18N.prompt(it["key"], lang),
            "scale": [(I18N.label(lbl, lang), val) for (lbl, val) in it["scale"]],
        }
        for it in items
    ]


def run_language_select(win, marker_outlet):
    """Participant picks the run language (trilingual prompt, shown before any language
    is set). Sets the module global EXPERIMENT_LANG, pushes a config marker, returns
    the code."""
    global EXPERIMENT_LANG
    mouse = event.Mouse(win=win)
    mouse.setVisible(True)
    event.clearEvents(eventType="keyboard")
    prompt = visual.TextStim(
        win=win, text="Choose your language\nElige tu idioma\nTria la teva llengua",
        pos=(0, 0.22), height=0.045, color="white", units="height", font="Arial")
    langs = I18N.LANGUAGES
    n = len(langs)
    btn_w, gap = 0.30, 0.06
    start_cx = -(n * btn_w + (n - 1) * gap) / 2 + btn_w / 2
    buttons = []
    for k, (code, native) in enumerate(langs):
        cx = start_cx + k * (btn_w + gap)
        rect, text = _rating_make_button(win, native, (cx, -0.06), btn_w, 0.13)
        buttons.append({"code": code, "rect": rect, "text": text})
    prev_down = False
    while True:
        for b in buttons:
            b["rect"].draw()
            b["text"].draw()
        prompt.draw()
        win.flip()
        pressed = bool(mouse.getPressed()[0])
        rising = pressed and not prev_down
        prev_down = pressed
        if rising:
            for b in buttons:
                if b["rect"].contains(mouse):
                    EXPERIMENT_LANG = b["code"]
                    safe_push(marker_outlet, f"config,language:{b['code']}")
                    return b["code"]
        if "escape" in event.getKeys(keyList=["escape"]):
            raise KeyboardInterrupt


def run_demographics_page(win, marker_outlet):
    """Full-screen participant-facing demographics page. Age is typed; the rest are
    clickable rounded buttons (reuses the rating-page layout). Returns {key: value};
    an untouched row -> 'no_response'. ESC aborts."""
    mouse = event.Mouse(win=win)
    mouse.setVisible(True)
    event.clearEvents(eventType="keyboard")
    safe_push(marker_outlet, "demographics_start")
    questions = _localize_items(DEMOGRAPHIC_QUESTIONS, EXPERIMENT_LANG)

    half_w = (float(win.size[0]) / float(win.size[1])) / 2.0
    x_left = -half_w + 0.03
    x_right = half_w - 0.03

    title = visual.TextStim(
        win=win, text=I18N.screen("demographics_title", EXPERIMENT_LANG),
        pos=(0, 0.45), height=0.038, color="white", units="height", font="Arial")
    subtitle = visual.TextStim(
        win=win,
        text=I18N.screen("demographics_subtitle", EXPERIMENT_LANG),
        pos=(0, 0.40), height=0.019, color=[0.80, 0.80, 0.85], units="height",
        font="Arial", wrapWidth=1.7)

    n_rows = 1 + len(questions)
    top_y, bottom_y = 0.31, -0.30
    row_step = (top_y - bottom_y) / (n_rows - 1)

    age_chars = []
    age_y = top_y
    age_prompt = visual.TextStim(
        win=win, text=I18N.screen("age_prompt", EXPERIMENT_LANG), pos=(x_left + 0.01, age_y), height=0.021,
        color="white", units="height", font="Arial",
        alignText="left", anchorHoriz="left", anchorVert="center")
    age_box = visual.ShapeStim(
        win=win, vertices=_rounded_rect_vertices(0.12, 0.060), pos=(0.0, age_y),
        fillColor=RATING_BUTTON_COLOR_IDLE, lineColor=RATING_BUTTON_LINE_IDLE,
        lineWidth=1.5, units="height")
    age_text = visual.TextStim(
        win=win, text="", pos=(0.0, age_y), height=0.026, color="white",
        units="height", font="Arial", anchorHoriz="center", anchorVert="center")

    prompts = []
    buttons = []
    selected = {q["key"]: None for q in questions}
    for i, q in enumerate(questions):
        y = top_y - (i + 1) * row_step
        ppos, pwrap, bspecs, _sp, _sw, _bh = _rating_layout_row(
            q, y_center=y, x_left_edge=x_left, x_right_skip=x_right)
        prompts.append(visual.TextStim(
            win=win, text=q["prompt"], pos=ppos, height=0.0195, color="white",
            units="height", font="Arial", alignText="left", anchorHoriz="left",
            anchorVert="center", wrapWidth=pwrap))
        for (label, value, pos, w, h) in bspecs:
            rect, text = _rating_make_button(win, label, pos, w, h)
            buttons.append({"key": q["key"], "value": value, "rect": rect, "text": text})

    cont_rect, cont_text = _rating_make_button(win, I18N.screen("continue_button", EXPERIMENT_LANG), (x_right - 0.11, -0.42), 0.20, 0.07)

    prev_down = False
    while True:
        for b in buttons:
            chosen = selected[b["key"]] is not None and selected[b["key"]] == b["value"]
            b["rect"].fillColor = RATING_BUTTON_COLOR_SELECTED if chosen else RATING_BUTTON_COLOR_IDLE
            b["rect"].lineColor = RATING_BUTTON_LINE_SELECTED if chosen else RATING_BUTTON_LINE_IDLE
        age_box.fillColor = RATING_BUTTON_COLOR_SELECTED if age_chars else RATING_BUTTON_COLOR_IDLE
        age_text.text = "".join(age_chars) if age_chars else "type..."

        title.draw()
        subtitle.draw()
        age_prompt.draw()
        age_box.draw()
        age_text.draw()
        for p in prompts:
            p.draw()
        for b in buttons:
            b["rect"].draw()
            b["text"].draw()
        cont_rect.draw()
        cont_text.draw()
        win.flip()

        for k in event.getKeys():
            if k == "escape":
                safe_push(marker_outlet, "experiment_end,aborted:1,stage:demographics")
                raise KeyboardInterrupt
            if k in ("backspace", "delete"):
                if age_chars:
                    age_chars.pop()
            elif len(age_chars) < 3:
                if k.isdigit():
                    age_chars.append(k)
                elif k.startswith("num_") and k[4:].isdigit():
                    age_chars.append(k[4:])

        down = mouse.getPressed()[0]
        if down and not prev_down:
            pos = mouse.getPos()
            if cont_rect.contains(pos):
                cont_rect.fillColor = RATING_BUTTON_COLOR_SELECTED
                cont_rect.draw()
                cont_text.draw()
                win.flip()
                core.wait(0.15)
                break
            for b in buttons:
                if b["rect"].contains(pos):
                    selected[b["key"]] = b["value"]
                    safe_push(
                        marker_outlet,
                        f"demographic_response,item:{b['key']},value:{b['value']},"
                        f"x:{pos[0]:.4f},y:{pos[1]:.4f}",
                    )
                    break
        prev_down = down

    result = {"age": "".join(age_chars)}
    for q in DEMOGRAPHIC_QUESTIONS:
        v = selected[q["key"]]
        result[q["key"]] = v if v is not None else "no_response"
    _summary = "|".join(f"{k}={v}" for k, v in result.items())
    safe_push(marker_outlet, f"demographics_complete,values:{_summary}")
    return result


def run_post_video_rating(
    win,
    marker_outlet,
    this_exp,
    video_job,
):
    """Show the post-video rating page and return the responses.

    The SAME question set (POST_VIDEO_RATING_ITEMS) is shown for EVERY video -
    one list, one function, identical each time. Participants click a value for
    each row or leave it blank, then click CONTINUE. ESC aborts.

    Returns: dict mapping each item key to a numeric/string value or
    'no_response' (left blank). Responses are pushed to LSL and written via
    this_exp.addData under keys 'rating_<item>'.
    """
    video_id = video_job.get("id", "video")
    video_condition = video_job.get("condition", "unspecified")

    items = _localize_items(POST_VIDEO_RATING_ITEMS, EXPERIMENT_LANG)

    safe_push(
        marker_outlet,
        f"rating_start,video:{video_id},condition:{video_condition}",
    )

    mouse = event.Mouse(win=win)
    mouse.setVisible(True)
    event.clearEvents(eventType="keyboard")

    title = visual.TextStim(
        win=win,
        text=I18N.screen("rating_title", EXPERIMENT_LANG),
        pos=(0, 0.44),
        height=0.040,
        color="white",
        units="height",
        font="Arial",
    )
    subtitle = visual.TextStim(
        win=win,
        text=I18N.screen("rating_subtitle", EXPERIMENT_LANG),
        pos=(0, 0.39),
        height=0.020,
        color=[0.80, 0.80, 0.85],
        units="height",
        font="Arial",
        wrapWidth=1.40,
    )

    n_items = len(items)
    # Usable half-width in height units = aspect/2 (x spans +/- aspect/2). Derive it
    # from the ACTUAL window aspect so prompts/buttons never spill off a narrower
    # screen: the old fixed -0.85 left edge clipped on a 16:10 display (x only spans
    # +/-0.8), which is why the questions started outside the frame fullscreen.
    half_w = (float(win.size[0]) / float(win.size[1])) / 2.0
    x_left = -half_w + 0.03
    x_right = half_w - 0.03
    # Vertical layout: fit n_items between y=top_y and y=bottom_y, with a
    # row step that adapts. The continue button sits below bottom_y.
    top_y = 0.30
    bottom_y = -0.30
    if n_items > 1:
        row_step = (top_y - bottom_y) / (n_items - 1)
    else:
        row_step = 0.0

    prompts = []
    buttons = []          # {"i": item_idx, "value": v, "rect": Rect, "text": TextStim}
    selected_value = [None] * n_items

    for i, item in enumerate(items):
        y = top_y - i * row_step
        prompt_pos, prompt_wrap, button_specs, skip_pos, skip_w, btn_h = _rating_layout_row(
            item, y_center=y, x_left_edge=x_left, x_right_skip=x_right,
        )

        prompts.append(visual.TextStim(
            win=win,
            text=item["prompt"],
            pos=prompt_pos,
            height=0.022,
            color="white",
            units="height",
            font="Arial",
            alignText="left",
            anchorHoriz="left",
            anchorVert="center",
            wrapWidth=prompt_wrap,
        ))

        for (label, value, pos, w, h) in button_specs:
            rect, text = _rating_make_button(win, label, pos, w, h)
            buttons.append({"i": i, "value": value, "rect": rect, "text": text})
        # No 'Skip' button: an untouched row is recorded as 'no_response' (= left
        # blank), which the participant is told they may do.

    cont_rect, cont_text = _rating_make_button(
        win, I18N.screen("continue_button", EXPERIMENT_LANG), (0.65, -0.42), 0.20, 0.07,
    )

    prev_mouse_down = False

    while True:
        # Update visual state of all rating buttons based on selection.
        for b in buttons:
            chosen = selected_value[b["i"]] is not None and selected_value[b["i"]] == b["value"]
            if chosen:
                b["rect"].fillColor = RATING_BUTTON_COLOR_SELECTED
                b["rect"].lineColor = RATING_BUTTON_LINE_SELECTED
            else:
                b["rect"].fillColor = RATING_BUTTON_COLOR_IDLE
                b["rect"].lineColor = RATING_BUTTON_LINE_IDLE

        title.draw()
        subtitle.draw()
        for p in prompts:
            p.draw()
        for b in buttons:
            b["rect"].draw()
            b["text"].draw()
        cont_rect.draw()
        cont_text.draw()
        win.flip()

        keys = event.getKeys(keyList=["escape"])
        if "escape" in keys:
            safe_push(marker_outlet, f"rating_aborted,video:{video_id}")
            raise KeyboardInterrupt

        mouse_pressed = mouse.getPressed()[0]
        if mouse_pressed and not prev_mouse_down:
            pos = mouse.getPos()
            # Continue first so a click that lands on Continue exits cleanly.
            if cont_rect.contains(pos):
                # Flash CONTINUE in the selected colour so the participant sees
                # that the click registered before the page advances.
                cont_rect.fillColor = RATING_BUTTON_COLOR_SELECTED
                cont_rect.lineColor = RATING_BUTTON_LINE_SELECTED
                title.draw()
                subtitle.draw()
                for p in prompts:
                    p.draw()
                for b in buttons:
                    b["rect"].draw()
                    b["text"].draw()
                cont_rect.draw()
                cont_text.draw()
                win.flip()
                core.wait(0.18)
                prev_mouse_down = True
                break
            for b in buttons:
                if b["rect"].contains(pos):
                    selected_value[b["i"]] = b["value"]
                    item_key = items[b["i"]]["key"]
                    safe_push(
                        marker_outlet,
                        f"rating_response,video:{video_id},item:{item_key},value:{b['value']},"
                        f"x:{pos[0]:.4f},y:{pos[1]:.4f}",
                    )
                    break
        prev_mouse_down = mouse_pressed

    # Build result and persist.
    result = {}
    for i, item in enumerate(items):
        val = selected_value[i]
        result[item["key"]] = val if val is not None else "no_response"

    # The virality item is already a single signed scale (-3 = definitely not viral,
    # +3 = definitely viral), so it directly serves as the signed calibration variable.
    result["virality_signed_confidence"] = result.get("virality", "no_response")

    for key, value in result.items():
        this_exp.addData(f"rating_{key}", value)

    summary = "|".join(f"{k}={v}" for k, v in result.items())
    safe_push(
        marker_outlet,
        f"rating_complete,video:{video_id},condition:{video_condition},values:{summary}",
    )

    return result


def run_video_block(
    win,
    marker_outlet,
    this_exp,
    this_dir,
    av_video_delay_override_s=None,
    video_selector_csv="",
):
    wait_for_space(
        win,
        I18N.screen("video_block_intro", EXPERIMENT_LANG),
        height=0.08,
    )

    video_jobs = resolve_video_playlist(this_dir)
    if not video_jobs:
        logging.warning("No valid embedded videos were found; skipping video block.")
        safe_push(marker_outlet, "video_block_skipped,no_valid_video:1")
        return

    requested_selectors = parse_csv_tokens(video_selector_csv)
    if requested_selectors:
        selected_jobs = select_video_jobs(video_jobs, requested_selectors)
        if selected_jobs:
            video_jobs = selected_jobs
            safe_push(
                marker_outlet,
                "config,video_filter_selectors:" + "|".join(s.replace(",", ";") for s in requested_selectors),
            )
        else:
            logging.warning("No videos matched --video-ids selectors; using full configured playlist.")
            safe_push(
                marker_outlet,
                "video_warning,reason:no_video_selector_match,selectors:"
                + "|".join(s.replace(",", ";") for s in requested_selectors),
            )

    # Uniform shuffle gives each video an equal probability of appearing first (1/N),
    # while avoiding repeating the exact previous run order when possible.
    video_jobs = shuffle_video_jobs_avoid_repeat(
        video_jobs,
        this_dir=this_dir,
        marker_outlet=marker_outlet,
    )
    safe_push(marker_outlet, f"config,video_playlist_count:{len(video_jobs)}")
    safe_push(
        marker_outlet,
        "config,video_order:" + "|".join(str(v.get("id", "video")) for v in video_jobs),
    )
    write_last_video_order(
        this_dir,
        [str(v.get("id", "video")) for v in video_jobs],
        marker_outlet=marker_outlet,
    )
    av_video_delay_s = (
        float(av_video_delay_override_s)
        if av_video_delay_override_s is not None
        else get_external_av_video_delay_s()
    )
    safe_push(marker_outlet, f"config,external_av_video_delay_s:{av_video_delay_s:.3f}")

    # Announce the rating schema once so the XDF carries the item->scale map.
    rating_item_summary = "|".join(
        f"{item['key']}={len(item['scale'])}pt" for item in POST_VIDEO_RATING_ITEMS
    )
    safe_push(marker_outlet, f"config,rating_items:{rating_item_summary}")
    for item in POST_VIDEO_RATING_ITEMS:
        labels = ";".join(f"{lbl.replace(chr(10), '_')}={val}" for lbl, val in item["scale"])
        safe_push(marker_outlet, f"config,rating_scale,{item['key']}:{labels}")

    # Warm up the ffpyplayer / sdl2 decoder before the first real video so the
    # first-to-second video transition does not stall while the library state
    # initializes (cold-start freeze visible at the end of the first clip).
    # The warm-up is hidden behind the pre-video fixation below.
    try:
        warmup = build_movie_stim(win, video_jobs[0]["path"], audio_on=False)
        try:
            warmup.draw()
            win.clearBuffer()
            win.flip()
        except Exception:
            pass
        for cleanup in ("stop", "unload"):
            try:
                getattr(warmup, cleanup, lambda: None)()
            except Exception:
                pass
        warmup = None
        safe_push(marker_outlet, "video_decoder_warmup,ok:1")
    except Exception as exc:
        err = str(exc).replace(",", ";")
        logging.warning(f"Video decoder warm-up skipped: {exc}")
        safe_push(marker_outlet, f"video_decoder_warmup,ok:0,error:{err}")

    show_fixation(win, duration_s=1.0)

    for vid_idx, video_job in enumerate(video_jobs):
        # Videos play, then the participant rates the video on the 5-question
        # post-video rating page (identical for every video). The pre_video_block
        # stream health check at the start of the block is enough; mid-block
        # stream drops are captured in the marker log without interrupting
        # the participant.
        #
        # Per-clip pupil baseline: constant-luminance grey screen + fixation cross,
        # bracketed by markers so the exact interval is recoverable from the XDF.
        _baseline_id = str(video_job.get("id", "video"))
        safe_push(
            marker_outlet,
            f"pre_video_baseline_start,id:{_baseline_id},"
            f"duration_s:{PRE_VIDEO_BASELINE_S:.3f}",
        )
        show_fixation(win, duration_s=PRE_VIDEO_BASELINE_S)
        safe_push(marker_outlet, f"pre_video_baseline_end,id:{_baseline_id}")

        play_video_job(
            win=win,
            marker_outlet=marker_outlet,
            this_exp=this_exp,
            video_job=video_job,
            av_video_delay_s=av_video_delay_s,
        )
        run_post_video_rating(
            win=win,
            marker_outlet=marker_outlet,
            this_exp=this_exp,
            video_job=video_job,
        )
        # Commit one CSV row per video, with video metadata AND rating
        # responses on the same row.
        this_exp.nextEntry()


def resolve_video_job_for_sync_sweep(this_dir, selector):
    """Resolve one video job for sync calibration by id, filename stem, or explicit path."""
    selector = str(selector or "").strip()
    if selector:
        maybe_path = Path(selector).expanduser()
        if maybe_path.exists():
            resolved = maybe_path.resolve()
            return {"id": resolved.stem, "path": str(resolved), "condition": "sync_test"}

    playlist = resolve_video_playlist(this_dir)
    if not playlist:
        return None

    if not selector:
        selector = "nyx_viral_short"
    selector_norm = selector.lower()

    for job in playlist:
        job_id = str(job.get("id", "")).lower()
        stem = Path(str(job.get("path", ""))).stem.lower()
        filename = Path(str(job.get("path", ""))).name.lower()
        if selector_norm in {job_id, stem, filename}:
            return job
        if selector_norm in job_id or selector_norm in stem or selector_norm in filename:
            return job

    return playlist[0]


def collect_av_sync_feedback(win, delay_s):
    """Collect user judgment for one sync trial."""
    delay_ms = int(round(float(delay_s) * 1000.0))
    prompt = visual.TextStim(
        win=win,
        text=(
            f"Delay tested: {delay_ms} ms\n\n"
            "How did it look?\n"
            "1 = Video ahead of audio (need MORE delay)\n"
            "2 = Closest / in sync\n"
            "3 = Audio ahead of video (need LESS delay)\n\n"
            "Press 1, 2, or 3."
        ),
        font="Arial",
        units="norm",
        pos=(0, 0),
        height=0.065,
        wrapWidth=1.8,
        color="white",
        colorSpace="rgb",
    )

    event.clearEvents(eventType="keyboard")
    while True:
        prompt.draw()
        win.flip()
        keys = event.getKeys(keyList=["1", "2", "3", "escape"])
        if "escape" in keys:
            raise KeyboardInterrupt
        for key in ("1", "2", "3"):
            if key in keys:
                return key


def run_av_sync_sweep(win, marker_outlet, this_exp, this_dir, delays_s, video_selector):
    """Run one-video repeated playback across multiple A/V delay values."""
    video_job = resolve_video_job_for_sync_sweep(this_dir, video_selector)
    if video_job is None:
        logging.warning("No valid video found for A/V sync sweep.")
        safe_push(marker_outlet, "av_sync_sweep_skipped,no_valid_video:1")
        return

    safe_push(marker_outlet, f"av_sync_sweep_start,video_id:{video_job.get('id', 'video')}")
    safe_push(
        marker_outlet,
        "config,av_sync_sweep_delays_s:" + "|".join(f"{d:.3f}" for d in delays_s),
    )
    wait_for_space(
        win,
        (
            "A/V Sync Calibration\n\n"
            "The same short video will repeat with different video delays.\n"
            "After each playback, rate sync quality:\n"
            "1 = video ahead, 2 = closest/in sync, 3 = audio ahead.\n\n"
            "Press SPACE to begin."
        ),
        height=0.07,
    )

    for trial_idx, delay_s in enumerate(delays_s, start=1):
        safe_push(
            marker_outlet,
            f"config,av_sync_trial,trial:{trial_idx},delay_s:{delay_s:.3f}",
        )
        show_fixation(win, duration_s=0.5)
        play_video_job(
            win=win,
            marker_outlet=marker_outlet,
            this_exp=this_exp,
            video_job=video_job,
            av_video_delay_s=float(delay_s),
        )
        safe_push(marker_outlet, f"rating_start,type:av_sync,trial:{trial_idx}")
        feedback_key = collect_av_sync_feedback(win, delay_s)
        feedback_map = {
            "1": ("video_ahead", +0.020),
            "2": ("in_sync", 0.000),
            "3": ("audio_ahead", -0.020),
        }
        feedback_label, suggestion_delta = feedback_map[feedback_key]
        safe_push(
            marker_outlet,
            (
                f"rating_response,type:av_sync,trial:{trial_idx},delay_s:{delay_s:.3f},"
                f"response:{feedback_label}"
            ),
        )

        this_exp.addData("av_sync_trial", trial_idx)
        this_exp.addData("av_sync_video_id", video_job.get("id", "video"))
        this_exp.addData("av_sync_delay_s", float(delay_s))
        this_exp.addData("av_sync_feedback_key", feedback_key)
        this_exp.addData("av_sync_feedback", feedback_label)
        this_exp.addData("av_sync_suggest_delta_s", float(suggestion_delta))
        this_exp.nextEntry()

    safe_push(marker_outlet, "av_sync_sweep_end")


def spawn_quick_qc(participant, session_start_mtime):
    """Spawn the per-participant QC figure generator in the background.

    Polls the LabRecorder output folder for a new, size-stable XDF (so the
    operator can still be saving the file when this runs) and then writes a
    4-panel PNG to data/results_by_participant/<pid>/. Detached, non-blocking,
    silently swallows all errors so experiment shutdown is never delayed.
    """
    try:
        if not QUICK_QC_SCRIPT.exists():
            logging.warning(f"quick_qc.py not found at {QUICK_QC_SCRIPT}; skipping QC figure")
            return
        python = QUICK_QC_PYTHON if Path(QUICK_QC_PYTHON).exists() else sys.executable
        cmd = [
            python, str(QUICK_QC_SCRIPT),
            "--participant", str(participant),
            "--wait-for-xdf",
            "--since-mtime", f"{float(session_start_mtime):.3f}",
        ]
        flags = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "DETACHED_PROCESS", 0)
        )
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
            close_fds=True,
        )
        logging.info(f"quick_qc spawned for participant={participant}")
    except Exception as exc:  # pragma: no cover - best-effort background task
        logging.warning(f"quick_qc spawn failed: {exc}")


def build_experiment_handler(exp_info, exp_name, this_dir):
    data_dir = os.path.join(this_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    data_file = os.path.join(
        data_dir,
        f"{exp_info['participant']}_{exp_name}_{exp_info['date']}",
    )

    this_exp = data.ExperimentHandler(
        name=exp_name,
        version="",
        extraInfo=exp_info,
        runtimeInfo=None,
        originPath=__file__,
        savePickle=True,
        saveWideText=True,
        dataFileName=data_file,
        sortColumns="time",
    )

    logging.LogFile(data_file + ".log")
    logging.console.setLevel("warning")

    return this_exp, data_file


def auto_preflight(runtime_opts):
    """Run preflight.py's check suite inline at task launch.

    Returns: (verdict, summary)
        verdict is 'proceed' or 'abort'.
        summary is a dict with ok / warn / fail counts plus a status field.

    If preflight.py is missing or fails to import, returns ('proceed', {...})
    with a 'module' field describing the issue. This is a safety net so a
    broken preflight cannot itself block a session; the operator can always
    relaunch with --skip-preflight to bypass entirely.
    """
    import importlib.util as _ilu
    preflight_path = Path(__file__).resolve().parent / "preflight.py"
    summary = {"ok": 0, "warn": 0, "fail": 0, "module": "ok"}

    if not preflight_path.is_file():
        logging.warning("preflight.py not found alongside task script; skipping inline preflight.")
        summary["module"] = "missing"
        return ("proceed", summary)

    try:
        spec = _ilu.spec_from_file_location("preflight", preflight_path)
        preflight = _ilu.module_from_spec(spec)
        spec.loader.exec_module(preflight)
    except Exception as e:
        logging.warning(f"Could not import preflight module: {e}")
        summary["module"] = "load_failed"
        return ("proceed", summary)

    # In av_sync_only calibration mode the EEG / Gazepoint streams are not
    # required, so skip the live LSL portion. Static and storage checks still run.
    skip_live = bool(getattr(runtime_opts, "av_sync_only", False))
    print("[Inline preflight] running automated pre-session checks ...")
    try:
        checks = preflight.run_all(skip_live=skip_live, quick=runtime_opts.preflight_quick)
    except Exception as e:
        logging.warning(f"Preflight run_all() raised: {e}")
        summary["module"] = f"run_failed: {type(e).__name__}"
        return ("proceed", summary)

    summary["ok"] = sum(1 for c in checks if c.status == "ok")
    crit_fails = [c for c in checks if c.status == "fail" and c.severity == "critical"]
    advisories = [c for c in checks if c.status in ("fail", "warn") and c.severity == "advisory"]
    summary["fail"] = len(crit_fails)
    summary["warn"] = len(advisories)

    if crit_fails:
        msg_lines = ["Preflight BLOCKED with critical errors:", ""]
        for c in crit_fails:
            msg_lines.append(f"  - {c.name}")
            msg_lines.append(f"      {c.detail}")
            if c.fix:
                msg_lines.append(f"      FIX: {c.fix}")
        msg_lines.append("")
        msg_lines.append("Fix the listed items and relaunch.")
        msg_lines.append("To bypass for debugging, relaunch with --skip-preflight.")
        full = "\n".join(msg_lines)
        try:
            gui.popupError(full)
        except Exception:
            pass
        print(full, file=sys.stderr)
        return ("abort", summary)

    if advisories:
        warn_lines = ["Preflight passed critical checks but flagged advisories:", ""]
        for c in advisories:
            warn_lines.append(f"  - {c.name}: {c.detail}")
        warn_lines.append("")
        warn_lines.append("Click OK to continue, Cancel to abort.")
        warn_full = "\n".join(warn_lines)
        try:
            confirm = gui.DlgFromDict(
                dictionary={"status": warn_full, "proceed": ["yes", "no"]},
                sortKeys=False,
                title="Preflight advisory warnings",
                fixed=["status"],
                alwaysOnTop=True,
            )
            if (not confirm.OK) or str(confirm.dictionary.get("proceed", "no")).strip().lower() != "yes":
                return ("abort", summary)
        except Exception:
            # If the dialog itself fails, default to proceed (advisories are
            # by definition non-blocking).
            pass

    return ("proceed", summary)


def main(argv=None):
    if StreamInfo is None:
        gui.popupError(
            "pylsl is not available in this PsychoPy environment.\n"
            f"Import error: {PYLSL_IMPORT_ERROR}"
        )
        return 1

    session_start_mtime = time.time()
    exp_name = "test_LSL_mouseclick"
    runtime_opts = parse_runtime_options(argv)

    # No-EEG/no-Gazepoint trial mode: skips the inline preflight + the hardware
    # dialogs, so the FULL experiment runs and records responses + mouse with only
    # the mouse bridge + LabRecorder. For verifying the pipeline, not real sessions.
    no_hardware = bool(getattr(runtime_opts, "no_hardware", False))

    # Automated pre-session preflight. Critical failures abort the launch before any
    # LSL outlet is created. --skip-preflight or --no-hardware bypasses this.
    preflight_summary = {"ok": 0, "warn": 0, "fail": 0, "module": "skipped"}
    if not getattr(runtime_opts, "skip_preflight", False) and not no_hardware:
        verdict, preflight_summary = auto_preflight(runtime_opts)
        if verdict == "abort":
            print("Inline preflight aborted launch.", file=sys.stderr)
            return 1

    cli_has_av_sync_only = cli_flag_present(argv, "--av-sync-only")
    cli_has_av_sync_video = cli_flag_present(argv, "--av-sync-video")
    cli_has_av_sync_sweep = cli_flag_present(argv, "--av-sync-sweep")
    cli_has_av_video_delay = cli_flag_present(argv, "--av-video-delay")
    cli_has_video_ids = cli_flag_present(argv, "--video-ids")
    mode_choices = ["task", "av_sync_only"]
    if runtime_opts.av_sync_only:
        mode_choices = ["av_sync_only", "task"]

    default_sweep_csv = runtime_opts.av_sync_sweep.strip()

    default_av_video_delay_s = runtime_opts.av_video_delay
    if default_av_video_delay_s is None:
        env_delay_raw = os.environ.get("PSYCHOPY_AV_SYNC_VIDEO_DELAY_S")
        if env_delay_raw is None:
            default_av_video_delay_s = DEFAULT_EXTERNAL_AV_VIDEO_DELAY_S
        else:
            try:
                default_av_video_delay_s = float(env_delay_raw)
            except Exception:
                default_av_video_delay_s = DEFAULT_EXTERNAL_AV_VIDEO_DELAY_S
    default_av_video_delay_s = max(
        MIN_AV_VIDEO_DELAY_S,
        min(MAX_AV_VIDEO_DELAY_S, float(default_av_video_delay_s)),
    )

    # Operator-facing dialog: only the participant CID + session. The A/V-sync /
    # video-subset / run-mode fields are developer overrides driven by CLI flags
    # (--av-sync-*, --video-ids, --av-sync-only); the downstream reads all fall back
    # to those CLI/env defaults via exp_info.get(..., default) when absent here.
    participant_info = {
        "instructions": (
            "Participant ID must be the CID from participants.csv (e.g. VC-07).\n"
            "Do not enter participant names or identifying details."
        ),
        "participant": f"{random.randint(0, 999999):06d}",
        "session": "001",
    }

    data_outlet = None
    marker_outlet = None
    win = None
    experiment_end_sent = False
    rcs_started = False  # True once we auto-start a LabRecorder recording (RCS)

    try:
        # Bring up PsychoPy LSL outlets before participant info so LabRecorder can discover them.
        data_outlet, marker_outlet = build_lsl_outlets()
        safe_push_data(data_outlet, [0.0] * 10)
        safe_push(marker_outlet, "config,mouse_stream_ready:1")
        core.wait(0.25)
    except Exception as exc:
        logging.error(f"Failed to initialize LSL outlets before participant dialog: {exc}")
        try:
            gui.popupError(f"LSL outlet setup failed:\n{exc}")
        except Exception:
            pass
        return 1

    # Audio output check. The ad videos must play their soundtrack, so in a REAL
    # session a missing device makes the data invalid - stop now (before participant
    # setup) with a clear, copyable error rather than crashing mid-video. But:
    #  - PsychoPy 2026's PTB backend can FAIL TO OPEN the default device by name
    #    even when it exists, so ensure_audio_device() recovers a working one by
    #    index (and routes the video audio to it via SELECTED_SPEAKER);
    #  - in a NO-HARDWARE / mouse-only capture test, audio is not the point, so a
    #    missing device only WARNS (videos play silent) and the run continues.
    audio_ok, audio_note, audio_exc = ensure_audio_device()
    safe_push(marker_outlet, f"config,audio_device,status:{'ok' if audio_ok else 'unavailable'}")
    if audio_ok and SELECTED_SPEAKER is not None:
        safe_push(marker_outlet, "config,audio_device,recovered:1")
        logging.warning(f"Audio: default device failed to open; {audio_note}.")
    if not audio_ok:
        if no_hardware:
            safe_push(marker_outlet, "warning,no_hardware,audio_device_unavailable:1")
            print(
                "[WARN] No usable audio output device - videos will be SILENT. "
                "Continuing because this is a no-hardware / mouse-only run.",
                file=sys.stderr,
            )
        else:
            present_fatal_error(
                "audio_device_check",
                audio_exc if audio_exc is not None else RuntimeError("no audio output device"),
                marker_outlet=marker_outlet,
                plain_explanation=(
                    "No usable audio output device was found, so the videos cannot "
                    "play their sound.\n"
                    "The session has been stopped because the ad videos need audio "
                    "to be valid.\n\n"
                    "What to do:\n"
                    "- Plug in / switch on a speaker or headset, set it as the Windows "
                    "default output device, then start again.\n"
                    "- If you are connected by Remote Desktop, audio is usually not "
                    "available remotely - run this on the lab computer.\n"
                    "- For a MOUSE-ONLY test with no audio, launch with --no-hardware "
                    "(or tick 'Mouse-only' on the launcher's preflight step) - videos "
                    "will be silent but mouse + responses still record."
                ),
            )
            return 1

    preflight_info = {
        "status": (
            "LSL Preflight\n\n"
            "1) In Gazepoint Control, ensure camera/tracker is connected\n"
            "2) EEG quality will be confirmed after participant entry\n"
            "3) Open LabRecorder (do NOT click Start yet)\n"
            "4) Click Update and verify these streams are visible:\n"
            "   - PsychoPyStream\n"
            "   - PsychoPyMarkers\n"
            "   - GazepointEyeTracker\n"
            "   - GazepointEvents\n"
            "5) Click OK to continue to participant entry"
        )
    }
    preflight_dlg = gui.DlgFromDict(
        dictionary=preflight_info,
        sortKeys=False,
        title="LSL Preflight",
        fixed=["status"],
        alwaysOnTop=True,
    )
    if not preflight_dlg.OK:
        safe_push(marker_outlet, "experiment_end,aborted:1,stage:lsl_preflight")
        return 0

    dlg = gui.DlgFromDict(
        dictionary=participant_info,
        sortKeys=False,
        title=exp_name,
        fixed=["instructions"],
        alwaysOnTop=True,
    )
    if not dlg.OK:
        return 0

    exp_info = {key: value for key, value in participant_info.items() if key != "instructions"}
    for key in ("participant", "session", "video_ids", "av_sync_video"):
        exp_info[key] = sanitize_operator_note(exp_info.get(key, ""))
    if not PARTICIPANT_ID_RE.fullmatch(exp_info.get("participant", "")):
        try:
            gui.popupError(PARTICIPANT_ID_ERROR)
        except Exception:
            pass
        safe_push(marker_outlet, "experiment_end,aborted:1,stage:participant_id_invalid")
        return 0
    # Will be updated after explicit calibration checkpoint.
    exp_info["gaze_calibrated"] = "no"
    exp_info["gaze_calibration_quality"] = "unknown"
    exp_info["gaze_calibration_notes"] = ""
    exp_info["date"] = data.getDateStr()
    exp_info["expName"] = exp_name

    # Demographics are collected in-window as a full-screen participant-facing page
    # (run_demographics_page), AFTER the Welcome screen and before the mouse task -
    # see below. CONFIRM the fields match the approved CER-IQS protocol before real
    # use; they are written to the CSV (extraInfo) and PsychoPyMarkers (config,demographic,*).

    run_mode_value = str(exp_info.get("run_mode", "task")).strip().lower()
    if not cli_has_av_sync_only:
        runtime_opts.av_sync_only = run_mode_value == "av_sync_only"

    av_sync_video_value = str(exp_info.get("av_sync_video", "")).strip()
    if av_sync_video_value and not cli_has_av_sync_video:
        runtime_opts.av_sync_video = av_sync_video_value

    av_sync_sweep_value = str(exp_info.get("av_sync_sweep_s", "")).strip()
    if not cli_has_av_sync_sweep:
        runtime_opts.av_sync_sweep = av_sync_sweep_value

    video_ids_value = str(exp_info.get("video_ids", "")).strip()
    if not cli_has_video_ids:
        runtime_opts.video_ids = video_ids_value
    gaze_calibrated_value = str(exp_info.get("gaze_calibrated", "no")).strip().lower()
    if gaze_calibrated_value not in {"yes", "no"}:
        gaze_calibrated_value = "no"
    gaze_calibration_quality_value = str(
        exp_info.get("gaze_calibration_quality", "unknown")
    ).strip().lower()
    if gaze_calibration_quality_value not in {"good", "ok", "poor", "unknown"}:
        gaze_calibration_quality_value = "unknown"
    gaze_calibration_notes_value = str(exp_info.get("gaze_calibration_notes", "")).strip()

    av_video_delay_override_s = None
    if runtime_opts.av_video_delay is not None:
        av_video_delay_override_s = max(
            MIN_AV_VIDEO_DELAY_S,
            min(MAX_AV_VIDEO_DELAY_S, float(runtime_opts.av_video_delay)),
        )

    av_video_delay_value = str(exp_info.get("av_video_delay_s", "")).strip()
    if av_video_delay_value and not cli_has_av_video_delay:
        try:
            av_video_delay_override_s = max(
                MIN_AV_VIDEO_DELAY_S,
                min(MAX_AV_VIDEO_DELAY_S, float(av_video_delay_value)),
            )
        except Exception:
            logging.warning(
                f"Invalid av_video_delay_s '{av_video_delay_value}', falling back to env/default."
            )

    av_sync_sweep_delays = parse_delay_values(runtime_opts.av_sync_sweep)
    run_av_sync_sweep_mode = runtime_opts.av_sync_only or bool(av_sync_sweep_delays)
    if run_av_sync_sweep_mode and not av_sync_sweep_delays:
        av_sync_sweep_delays = list(DEFAULT_AV_SYNC_SWEEP_DELAYS_S)

    fullscreen_mode, mode_source = parse_display_mode(argv)
    print(
        f"[INFO] Display mode: {'fullscreen' if fullscreen_mode else 'windowed'} "
        f"({mode_source})"
    )
    if run_av_sync_sweep_mode:
        print(
            "[INFO] A/V sync sweep enabled: "
            f"video='{runtime_opts.av_sync_video}', "
            "delays_s=[" + ",".join(f"{d:.3f}" for d in av_sync_sweep_delays) + "]"
        )
        print(f"[INFO] A/V sync only mode: {bool(runtime_opts.av_sync_only)}")
    if av_video_delay_override_s is not None:
        print(f"[INFO] Video delay override resolved: {av_video_delay_override_s:.3f}s")
    if str(runtime_opts.video_ids or "").strip():
        print(f"[INFO] Main video selection filter: {runtime_opts.video_ids}")

    this_dir = os.path.dirname(os.path.abspath(__file__))
    this_exp, data_file = build_experiment_handler(exp_info, exp_name, this_dir)

    try:
        # Outlets are already active from the LSL preflight phase above.
        # Early best-effort refresh + select-all in LabRecorder via its Remote
        # Control Server. The AUTHORITATIVE re-select happens right before we start
        # recording (below): PsychoPyMarkers only appears when the task launches -
        # after LabRecorder was opened - so a select-all issued now can miss it
        # before LabRecorder's resolve has listed it. No operator popup here; the
        # launcher already walked through LabRecorder setup, and the start
        # checkpoint re-selects automatically.
        rcs_selected = labrecorder_rcs("update", "select all")
        safe_push(marker_outlet, f"config,labrecorder_rcs_select:{1 if rcs_selected else 0}")
        # Provenance: the task does NOT set the LabRecorder filename, so record the
        # intended CID (the launcher's LabRecorder step reminds the operator to set
        # the Participant field; we no longer pop a dialog for it here).
        _xdf_pid = exp_info.get("participant", "?")
        safe_push(marker_outlet, f"config,xdf_participant_should_be:{_xdf_pid}")
        print(
            f"[INFO] LabRecorder RCS early select-all sent (reachable={rcs_selected}). "
            f"Reminder: LabRecorder Participant field should be '{_xdf_pid}'."
        )

        if no_hardware:
            safe_push(marker_outlet, "config,no_hardware_mode:1")
            eeg_qc_method = "none_no_hardware"
            eeg_quality_verified = "no"
            eeg_qc_notes = ""
        else:
            eeg_qc_info = {
                "status": OPERATOR_NOTE_WARNING,
                "eeg_qc_method": ["openbci_gui_impedance", "stream_eeg_quality_markers"],
                "eeg_quality_verified": ["yes", "no"],
                "eeg_qc_notes": "",
            }
            eeg_qc_dlg = gui.DlgFromDict(
                dictionary=eeg_qc_info,
                sortKeys=False,
                title="EEG Quality Check",
                fixed=["status"],
                alwaysOnTop=True,
            )
            if not eeg_qc_dlg.OK:
                safe_push(marker_outlet, "experiment_end,aborted:1,stage:eeg_qc_confirmation")
                return 0
            eeg_qc_method = str(eeg_qc_info.get("eeg_qc_method", "openbci_gui_impedance")).strip().lower()
            eeg_quality_verified = str(eeg_qc_info.get("eeg_quality_verified", "no")).strip().lower()
            eeg_qc_notes = sanitize_operator_note(eeg_qc_info.get("eeg_qc_notes", ""))
            if eeg_qc_method not in {"openbci_gui_impedance", "stream_eeg_quality_markers"}:
                eeg_qc_method = "openbci_gui_impedance"
            if eeg_quality_verified not in {"yes", "no"}:
                eeg_quality_verified = "no"

        safe_push(marker_outlet, f"config,eeg_qc_method:{eeg_qc_method}")
        safe_push(marker_outlet, f"config,eeg_quality_verified:{1 if eeg_quality_verified == 'yes' else 0}")
        if eeg_qc_notes:
            safe_push(marker_outlet, "config,eeg_qc_notes:" + eeg_qc_notes.replace(",", ";"))

        exp_info["eeg_qc_method"] = eeg_qc_method
        exp_info["eeg_quality_verified"] = eeg_quality_verified
        exp_info["eeg_qc_notes"] = eeg_qc_notes
        if isinstance(getattr(this_exp, "extraInfo", None), dict):
            this_exp.extraInfo["eeg_qc_method"] = eeg_qc_method
            this_exp.extraInfo["eeg_quality_verified"] = eeg_quality_verified
            this_exp.extraInfo["eeg_qc_notes"] = eeg_qc_notes

        if eeg_quality_verified != "yes" and not no_hardware:
            safe_push(marker_outlet, "experiment_end,aborted:1,stage:eeg_qc_failed")
            return 0

        if not runtime_opts.av_sync_only:
            safe_push(marker_outlet, "config,gazepoint_calibration_capture_prompted:1")
            # Name the XDF after this participant (CID) over RCS, BEFORE recording
            # starts, so the operator no longer has to set LabRecorder's Participant
            # field by hand (which otherwise keeps the PREVIOUS participant's name).
            # Syntax probed live against LabRecorder v1.16:
            #   filename {participant:P}{session:S}{task:T}{run:R}
            #   -> sub-P/ses-S/eeg/sub-P_ses-S_task-T_run-00R_eeg.xdf
            # BIDS labels must be alphanumeric, so the CID is stripped to [A-Za-z0-9].
            _pid = re.sub(r"[^A-Za-z0-9]", "", str(exp_info.get("participant", "") or "")) or "UNKNOWN"
            _sess = re.sub(r"[^A-Za-z0-9]", "", str(exp_info.get("session", "") or "")) or "S001"
            _taskname = re.sub(r"[^A-Za-z0-9]", "", str(exp_name or "task")) or "task"
            _fname_ok = labrecorder_rcs(
                f"filename {{participant:{_pid}}} {{session:{_sess}}} {{task:{_taskname}}} {{run:1}}"
            )
            safe_push(marker_outlet, f"config,labrecorder_rcs_filename_set:{1 if _fname_ok else 0}")
            safe_push(marker_outlet, f"config,xdf_filename_participant:{_pid}")
            # AUTHORITATIVE, operator-proof recording control, run AFTER the markers
            # stream exists. PsychoPyMarkers is late-born (created at task startup,
            # after LabRecorder was opened), and is only recorded if it is SELECTED
            # before recording starts. If a recording is already running (e.g. the
            # operator pressed Start during setup), its stream set is LOCKED and the
            # late markers are silently excluded - which is exactly what dropped them
            # in testing. So we STOP any in-progress recording first, then re-select
            # ALL streams (now including PsychoPyMarkers) and start fresh. Verified
            # against LabRecorder v1.16: a premature start drops the late stream; this
            # stop -> update -> select all -> start sequence recovers it. The operator
            # therefore cannot break it by pressing (or forgetting) Start themselves.
            labrecorder_rcs("stop")            # clear any premature / wrongly-selected recording
            core.wait(0.8)
            labrecorder_rcs("update")
            core.wait(2.0)                      # let LabRecorder resolve the late marker stream
            rcs_reselected = labrecorder_rcs("select all")
            safe_push(marker_outlet, f"config,labrecorder_rcs_reselect:{1 if rcs_reselected else 0}")
            core.wait(0.3)
            # Start a FRESH recording (RCS). Falls back to a manual-Start instruction +
            # an explicit operator confirmation if LabRecorder/RCS is not reachable.
            rcs_started = labrecorder_rcs("start")
            safe_push(marker_outlet, f"config,labrecorder_rcs_start:{1 if rcs_started else 0}")
            if no_hardware:
                # No eye-tracker present: the RCS recording start above still ran, but
                # there is nothing to calibrate, so SKIP the gaze-calibration operator
                # dialogs entirely. Warn (do NOT abort) if recording could not be
                # confirmed, so a no-hardware capture test still runs end to end.
                if not rcs_started:
                    safe_push(marker_outlet, "config,operator_confirmed_recording_active:0")
                    safe_push(marker_outlet, "warning,no_hardware,labrecorder_rcs_unreachable:1")
                    print(
                        "[WARN] LabRecorder RCS not reachable - no XDF will be recorded. "
                        "Open LabRecorder with 'Enable RCS' ticked to capture the mouse stream.",
                        file=sys.stderr,
                    )
                    # On-screen warning so this is IMPOSSIBLE to miss: with no
                    # LabRecorder the PsychoPy CSV still saves, but the mouse + marker
                    # LSL streams are NOT recorded to an XDF (so there is nothing to
                    # analyse offline). Let the operator stop and fix LabRecorder, or
                    # knowingly continue CSV-only.
                    try:
                        _warn_dlg = gui.Dlg(title="NO XDF WILL BE RECORDED")
                        _warn_dlg.addText("LabRecorder is NOT recording (RCS not reachable).")
                        _warn_dlg.addText("")
                        _warn_dlg.addText("Responses will still save to a CSV, but the mouse +")
                        _warn_dlg.addText("marker LSL streams will NOT be saved to an XDF - so there")
                        _warn_dlg.addText("will be no analysable recording from this run.")
                        _warn_dlg.addText("")
                        _warn_dlg.addText("To capture the XDF: click CANCEL, open LabRecorder, tick")
                        _warn_dlg.addText("'Enable RCS', click Select All (do NOT press Start), rerun.")
                        _warn_dlg.addText("Click OK to continue WITHOUT recording (CSV only).")
                        _warn_dlg.show()
                        _warn_ok = bool(_warn_dlg.OK)
                    except Exception:
                        _warn_ok = True  # a GUI hiccup must never crash the run
                    if not _warn_ok:
                        safe_push(
                            marker_outlet,
                            "experiment_end,aborted:1,stage:no_hardware_rcs_unreachable",
                        )
                        return 0
                safe_push(marker_outlet, "config,gaze_calibration_skipped_no_hardware:1")
                gaze_calibrated_value = "skipped_no_hardware"
                gaze_calibration_quality_value = "unknown"
                gaze_calibration_notes_value = ""
            else:
                if rcs_started:
                    cal_start_line = (
                        "1) Recording was STARTED automatically in LabRecorder over RCS\n"
                        "   (its button now reads Stop, and PsychoPyMarkers is included).\n"
                        "   You do NOT need to press Start - the task controls it."
                    )
                else:
                    cal_start_line = (
                        "1) !! Could NOT control LabRecorder over RCS. Open LabRecorder,\n"
                        "   tick 'Enable RCS', then click Update -> Select All -> Start NOW.\n"
                        "   Confirm its button reads STOP, or NOTHING is being recorded."
                    )
                calibration_capture_info = {
                    "status": (
                        OPERATOR_NOTE_WARNING + "\n\n"
                        "Gaze Calibration Capture Checkpoint\n\n"
                        + cal_start_line + "\n"
                        "2) In Gazepoint Control, run participant gaze calibration now\n"
                        "3) Wait for calibration to complete\n"
                        "4) Return here and click OK"
                    )
                }
                if not rcs_started:
                    # RCS could not confirm recording - force the operator to verify it is
                    # actually recording before we proceed (the start-check you asked for).
                    calibration_capture_info["recording_is_active_button_reads_Stop"] = ["no", "yes"]
                calibration_capture_dlg = gui.DlgFromDict(
                    dictionary=calibration_capture_info,
                    sortKeys=False,
                    title="Gaze Calibration Capture",
                    fixed=["status"],
                    alwaysOnTop=True,
                )
                if not calibration_capture_dlg.OK:
                    safe_push(
                        marker_outlet,
                        "experiment_end,aborted:1,stage:calibration_capture_checkpoint",
                    )
                    return 0
                if not rcs_started:
                    _rec_active = str(
                        calibration_capture_info.get("recording_is_active_button_reads_Stop", "no")
                    ).strip().lower()
                    safe_push(
                        marker_outlet,
                        f"config,operator_confirmed_recording_active:{1 if _rec_active == 'yes' else 0}",
                    )
                    if _rec_active != "yes":
                        safe_push(marker_outlet, "experiment_end,aborted:1,stage:recording_not_active")
                        try:
                            gui.popupError(
                                "Recording was not confirmed active in LabRecorder.\n"
                                "Aborting so we do not run a whole session with no data.\n\n"
                                "Open LabRecorder with 'Enable RCS' ticked and run again."
                            )
                        except Exception:
                            pass
                        return 0

                calibration_meta = {
                    "status": OPERATOR_NOTE_WARNING,
                    "gaze_calibrated": ["yes", "no"],
                    "gaze_calibration_quality": ["good", "ok", "poor", "unknown"],
                    "gaze_calibration_notes": "",
                }
                calibration_meta_dlg = gui.DlgFromDict(
                    dictionary=calibration_meta,
                    sortKeys=False,
                    title="Calibration Result",
                    fixed=["status"],
                    alwaysOnTop=True,
                )
                if not calibration_meta_dlg.OK:
                    safe_push(
                        marker_outlet,
                        "experiment_end,aborted:1,stage:calibration_metadata_entry",
                    )
                    return 0

                gaze_calibrated_value = str(calibration_meta.get("gaze_calibrated", "no")).strip().lower()
                if gaze_calibrated_value not in {"yes", "no"}:
                    gaze_calibrated_value = "no"
                gaze_calibration_quality_value = str(
                    calibration_meta.get("gaze_calibration_quality", "unknown")
                ).strip().lower()
                if gaze_calibration_quality_value not in {"good", "ok", "poor", "unknown"}:
                    gaze_calibration_quality_value = "unknown"
                gaze_calibration_notes_value = sanitize_operator_note(
                    calibration_meta.get("gaze_calibration_notes", "")
                )

            exp_info["gaze_calibrated"] = gaze_calibrated_value
            exp_info["gaze_calibration_quality"] = gaze_calibration_quality_value
            exp_info["gaze_calibration_notes"] = gaze_calibration_notes_value
            if isinstance(getattr(this_exp, "extraInfo", None), dict):
                this_exp.extraInfo["gaze_calibrated"] = gaze_calibrated_value
                this_exp.extraInfo["gaze_calibration_quality"] = gaze_calibration_quality_value
                this_exp.extraInfo["gaze_calibration_notes"] = gaze_calibration_notes_value

            safe_push(marker_outlet, "config,gazepoint_calibration_capture_confirmed:1")

        # Choose the stimulus monitor. The mouse bridge normalises the cursor against
        # the PRIMARY monitor (the one at virtual origin 0,0), so the stimulus must be
        # fullscreen THERE for the cursor to map 1:1 to targets. Hardcoding screen=0
        # could land on a larger SECONDARY monitor (it does on Work-PC-2) and silently
        # skew the mouse<->target mapping. Honour --screen if given, else auto-detect
        # the primary's pyglet index.
        _stim_screen = getattr(runtime_opts, "screen", None)
        if _stim_screen is None:
            try:
                import pyglet as _pyglet
                _scrs = _pyglet.canvas.get_display().get_screens()
                _stim_screen = next((i for i, s in enumerate(_scrs) if s.x == 0 and s.y == 0), 0)
            except Exception:
                _stim_screen = 0
        safe_push(marker_outlet, f"config,stimulus_screen_index:{_stim_screen}")

        win = visual.Window(
            size=(1280, 720),
            fullscr=fullscreen_mode,
            screen=_stim_screen,
            winType="pyglet",
            allowGUI=not fullscreen_mode,
            allowStencil=False,
            monitor="testMonitor",
            color=[0.3255, 0.3255, 0.3255],
            colorSpace="rgb",
            units="height",
            useFBO=True,
        )

        mouse = event.Mouse(win=win)
        mouse.setVisible(True)

        circle = visual.ShapeStim(
            win=win,
            name="circlestim",
            size=(0.05, 0.05),
            vertices="circle",
            pos=(0, 0),
            lineColor="white",
            fillColor="white",
            colorSpace="rgb",
            lineWidth=1.0,
            interpolate=True,
        )

        safe_push(
            marker_outlet,
            f"config,av_sync_sweep_mode:{1 if run_av_sync_sweep_mode else 0}",
        )
        safe_push(
            marker_outlet,
            f"config,av_sync_only_mode:{1 if runtime_opts.av_sync_only else 0}",
        )
        if run_av_sync_sweep_mode:
            safe_push(
                marker_outlet,
                f"config,av_sync_video_selector:{str(runtime_opts.av_sync_video).replace(',', ';')}",
            )
            safe_push(
                marker_outlet,
                "config,av_sync_sweep_delays_s:" + "|".join(f"{d:.3f}" for d in av_sync_sweep_delays),
            )
            run_av_sync_sweep(
                win=win,
                marker_outlet=marker_outlet,
                this_exp=this_exp,
                this_dir=this_dir,
                delays_s=av_sync_sweep_delays,
                video_selector=runtime_opts.av_sync_video,
            )
            safe_push(marker_outlet, "experiment_end")
            experiment_end_sent = True
            wait_for_space(
                win,
                "A/V sync calibration complete.\n\nPress SPACE to end.",
                height=0.07,
            )
            return 0

        run_language_select(win, marker_outlet)
        wait_for_space(win, I18N.screen("welcome", EXPERIMENT_LANG), height=0.10)

        # Full-screen demographics page (participant-facing), before the mouse task.
        _demo = run_demographics_page(win, marker_outlet)
        exp_info["demo_age"] = re.sub(r"[^0-9]", "", str(_demo.get("age", "")))[:3]
        for _dk in DEMOGRAPHIC_KEYS:
            exp_info[f"demo_{_dk}"] = sanitize_operator_note(str(_demo.get(_dk, "")))
        if isinstance(getattr(this_exp, "extraInfo", None), dict):
            for _dk in ("age",) + DEMOGRAPHIC_KEYS:
                this_exp.extraInfo[f"demo_{_dk}"] = exp_info.get(f"demo_{_dk}", "")

        wait_for_space(
            win,
            I18N.screen("targeting_practice", EXPERIMENT_LANG),
            height=0.08,
        )

        safe_push(marker_outlet, f"config,monitor_size_px:{int(win.size[0])},{int(win.size[1])}")
        # Demographics into the XDF (now that the recording is live; collected at start).
        for _dk in ("age",) + DEMOGRAPHIC_KEYS:
            safe_push(marker_outlet, f"config,demographic,{_dk}:{exp_info.get('demo_' + _dk, '')}")
        safe_push(marker_outlet, f"config,mouse_units:{mouse.units}")
        safe_push(marker_outlet, f"config,mouse_visible:{mouse.getVisible()}")
        safe_push(marker_outlet, "config,data_sampling_mode:irregular_per_frame")
        safe_push(marker_outlet, "config,data_timestamp_mode:explicit_lsl_local_clock")
        safe_push(marker_outlet, f"config,display_mode:{'fullscreen' if fullscreen_mode else 'windowed'}")
        safe_push(marker_outlet, f"config,display_mode_source:{mode_source}")
        # Multi-monitor provenance: the stream_mouse.py bridge streams the cursor in
        # virtual-desktop pixels. Record this window's origin + the virtual-desktop
        # bounds in the SAME frame so cursor samples map to the stimulus (and to
        # PsychoPy units) offline, unambiguously, even on a multi-monitor rig.
        try:
            import ctypes as _ctypes
            _u32 = _ctypes.windll.user32
            _vd = (
                int(_u32.GetSystemMetrics(76)), int(_u32.GetSystemMetrics(77)),
                int(_u32.GetSystemMetrics(78)), int(_u32.GetSystemMetrics(79)),
            )  # SM_X/YVIRTUALSCREEN, SM_YVIRTUALSCREEN, SM_CX/CYVIRTUALSCREEN
            safe_push(marker_outlet, "config,virtual_desktop_px:%d,%d,%d,%d" % _vd)
        except Exception:
            pass
        try:
            _wx, _wy = win.winHandle.get_location()
            safe_push(marker_outlet, f"config,window_screen_pos_px:{int(_wx)},{int(_wy)}")
        except Exception:
            safe_push(marker_outlet, "config,window_screen_pos_px:unknown")
        safe_push(marker_outlet, f"config,window_screen_index:{getattr(win, 'screen', 0)}")
        # DPI scale of the window's monitor. The mouse bridge logs the cursor in
        # DPI-aware PHYSICAL pixels; if PsychoPy reports win.size in LOGICAL (scaled)
        # units the two differ by this factor (e.g. 1.5 at 150%). Recording it lets
        # the offline mapping reconcile them exactly without guessing - compare
        # monitor_size_px*scale against the bridge's physical screen size.
        try:
            import ctypes as _ctypes
            _u = _ctypes.windll.user32
            _hwnd = getattr(getattr(win, "winHandle", None), "_hwnd", None) or _u.GetForegroundWindow()
            _dpi = _u.GetDpiForWindow(_hwnd) if hasattr(_u, "GetDpiForWindow") else 96
            safe_push(marker_outlet, f"config,window_dpi:{int(_dpi or 96)}")
            safe_push(marker_outlet, f"config,window_dpi_scale:{(_dpi or 96) / 96.0:.4f}")
        except Exception:
            pass
        safe_push(marker_outlet, f"config,mouse_fixation_base_s:{MOUSE_FIXATION_BASE_S:.3f}")
        safe_push(marker_outlet, f"config,mouse_fixation_jitter_fraction:{MOUSE_FIXATION_JITTER_FRACTION:.3f}")
        safe_push(marker_outlet, f"config,mouse_trial_count:{MOUSE_TRIAL_COUNT}")
        safe_push(marker_outlet, "config,mouse_target_mode:fixed_radius_random_direction")
        safe_push(
            marker_outlet,
            "config,mouse_target_radius_fraction_of_vertical_half:"
            f"{MOUSE_TARGET_RADIUS_FRACTION:.3f}",
        )
        fixed_target_radius = get_fixed_target_radius(win, circle.size)
        safe_push(marker_outlet, f"config,mouse_target_radius_height_units:{fixed_target_radius:.4f}")
        safe_push(marker_outlet, f"config,resample_hint_hz:{ALIGNMENT_TARGET_HZ}")
        safe_push(marker_outlet, f"config,alignment_target_hz:{ALIGNMENT_TARGET_HZ}")
        safe_push(
            marker_outlet,
            f"config,gazepoint_calibrated:{1 if gaze_calibrated_value == 'yes' else 0}",
        )
        safe_push(
            marker_outlet,
            "config,gazepoint_calibration_quality:"
            + gaze_calibration_quality_value.replace(",", ";"),
        )
        if gaze_calibration_notes_value:
            safe_push(
                marker_outlet,
                "config,gazepoint_calibration_notes:"
                + gaze_calibration_notes_value.replace(",", ";"),
            )
        safe_push(
            marker_outlet,
            "config,intended_alignment_streams:"
            "OpenBCI_CytonDaisy_EEG|GazepointEyeTracker|GazepointEvents|PsychoPyMarkers",
        )

        # ── Stream health check: before mouse task ─────────────────
        health = check_lsl_streams(marker_outlet, context="pre_mouse_task")
        health_warn = format_stream_health_warning(health)
        if health_warn and not no_hardware:
            wait_for_space(
                win,
                f"{health_warn}\n\nYou may continue, but data may be incomplete.\n"
                "Fix streams and restart if needed, or press SPACE to continue anyway.",
                height=0.04,
            )

        # ── Practice block: warm-up trials to learn the task; NOT analysed.
        # They emit 'mouse_practice_trial_*' markers (so the real analysis ignores
        # them) and are not written to the CSV. A grey "real test begins" screen
        # separates practice from the real block.
        n_practice = N_PRACTICE_MOUSE_TRIALS
        if getattr(runtime_opts, "practice_trials", None) is not None and runtime_opts.practice_trials >= 0:
            n_practice = int(runtime_opts.practice_trials)
        if n_practice > 0:
            safe_push(marker_outlet, f"mouse_practice_block_start,n:{n_practice}")
            for p_index in range(n_practice):
                fixation_s = jittered_duration(MOUSE_FIXATION_BASE_S, MOUSE_FIXATION_JITTER_FRACTION)
                mouse.setPos((0.0, 0.0))
                show_fixation(win, duration_s=fixation_s)
                target_pos = random_circle_pos(win, circle.size, fixed_radius=fixed_target_radius)
                run_mouse_trial(
                    win=win, mouse=mouse, circle=circle, target_pos=target_pos,
                    data_outlet=data_outlet, marker_outlet=marker_outlet,
                    trial_number=p_index + 1, practice=True,
                )
                if MOUSE_POST_CLICK_FIXATION_S > 0:
                    show_fixation(win, duration_s=MOUSE_POST_CLICK_FIXATION_S)
            safe_push(marker_outlet, "mouse_practice_block_end")
            wait_for_space(
                win,
                I18N.screen("targeting_transition", EXPERIMENT_LANG),
                height=0.08,
            )
        safe_push(marker_outlet, "mouse_test_block_start")

        n_trials = MOUSE_TRIAL_COUNT
        if getattr(runtime_opts, "mouse_trials", None) is not None and runtime_opts.mouse_trials >= 0:
            n_trials = int(runtime_opts.mouse_trials)
            safe_push(marker_outlet, f"config,mouse_trials_override:{n_trials}")
        for trial_index in range(n_trials):
            fixation_s = jittered_duration(MOUSE_FIXATION_BASE_S, MOUSE_FIXATION_JITTER_FRACTION)
            safe_push(
                marker_outlet,
                f"config,trial_fixation_s,trial:{trial_index + 1},duration:{fixation_s:.3f}",
            )
            # Re-centre the cursor on the fixation cross the moment fixation appears,
            # so every trial starts from the centre. It is recentred again right before
            # the circle (in run_mouse_trial) in case the participant drifts it during
            # fixation. We MOVE it rather than freeze it, so it never looks broken.
            mouse.setPos((0.0, 0.0))
            show_fixation(win, duration_s=fixation_s)
            target_pos = random_circle_pos(win, circle.size, fixed_radius=fixed_target_radius)
            trial_result = run_mouse_trial(
                win=win,
                mouse=mouse,
                circle=circle,
                target_pos=target_pos,
                data_outlet=data_outlet,
                marker_outlet=marker_outlet,
                trial_number=trial_index + 1,
            )

            # Post-click fixation: clears the target circle and gives the
            # participant a visual confirmation that the click registered
            # before the next trial (or the video block) begins.
            if MOUSE_POST_CLICK_FIXATION_S > 0 and not trial_result.get("timeout"):
                show_fixation(win, duration_s=MOUSE_POST_CLICK_FIXATION_S)

            this_exp.addData("trial", trial_index + 1)
            this_exp.addData("target_x", target_pos[0])
            this_exp.addData("target_y", target_pos[1])
            this_exp.addData("success", trial_result["success"])
            this_exp.addData("rt", trial_result["rt"])
            this_exp.addData("click_x", trial_result["click_x"])
            this_exp.addData("click_y", trial_result["click_y"])
            this_exp.nextEntry()

        # ── Go/No-Go (redirect) block ──────────────────────────────
        # Click GREY circles; on a WHITE circle redirect and click the centre cross
        # instead. Runs straight after the targeting block, so "click the circle" is
        # the freshly-trained prepotent response to override on the white (no-go)
        # trials. Self-paced (no timed window); the cross stays on screen throughout.
        n_gonogo = N_GONOGO_TRIALS
        n_gonogo_white = N_GONOGO_NOGO
        if getattr(runtime_opts, "mouse_trials", None) is not None and runtime_opts.mouse_trials >= 0:
            n_gonogo = int(runtime_opts.mouse_trials)
            n_gonogo_white = max(1, n_gonogo // 3) if n_gonogo > 1 else 0
        n_gonogo_practice = N_GONOGO_PRACTICE
        if getattr(runtime_opts, "practice_trials", None) is not None and runtime_opts.practice_trials >= 0:
            n_gonogo_practice = int(runtime_opts.practice_trials)
        safe_push(marker_outlet, f"config,gonogo_count:{n_gonogo}")
        safe_push(marker_outlet, f"config,gonogo_nogo_count:{n_gonogo_white}")
        safe_push(marker_outlet, f"config,gonogo_cross_hit_radius_height:{GONOGO_CROSS_HIT_RADIUS:.4f}")
        wait_for_space(
            win,
            I18N.screen("gonogo_rule", EXPERIMENT_LANG),
            height=0.05,
        )
        if n_gonogo_practice > 0:
            safe_push(marker_outlet, f"gonogo_practice_block_start,n:{n_gonogo_practice}")
            for p_index, lum in enumerate(
                gonogo_luminance_sequence(
                    n_gonogo_practice,
                    max(1, n_gonogo_practice // 3) if n_gonogo_practice > 1 else 0,
                )
            ):
                fixation_s = jittered_duration(MOUSE_FIXATION_BASE_S, MOUSE_FIXATION_JITTER_FRACTION)
                mouse.setPos((0.0, 0.0))
                show_fixation(win, duration_s=fixation_s)
                target_pos = random_circle_pos(win, circle.size, fixed_radius=fixed_target_radius)
                run_gonogo_trial(
                    win=win, mouse=mouse, circle=circle,
                    cross_hit_radius=GONOGO_CROSS_HIT_RADIUS, luminance=lum,
                    target_pos=target_pos, data_outlet=data_outlet,
                    marker_outlet=marker_outlet, trial_number=p_index + 1, practice=True,
                )
                if MOUSE_POST_CLICK_FIXATION_S > 0:
                    show_fixation(win, duration_s=MOUSE_POST_CLICK_FIXATION_S)
            safe_push(marker_outlet, "gonogo_practice_block_end")
            wait_for_space(
                win,
                I18N.screen("gonogo_transition", EXPERIMENT_LANG),
                height=0.05,
            )
        safe_push(marker_outlet, "gonogo_test_block_start")
        for g_index, lum in enumerate(gonogo_luminance_sequence(n_gonogo, n_gonogo_white)):
            fixation_s = jittered_duration(MOUSE_FIXATION_BASE_S, MOUSE_FIXATION_JITTER_FRACTION)
            safe_push(
                marker_outlet,
                f"config,gonogo_fixation_s,trial:{g_index + 1},duration:{fixation_s:.3f}",
            )
            mouse.setPos((0.0, 0.0))
            show_fixation(win, duration_s=fixation_s)
            target_pos = random_circle_pos(win, circle.size, fixed_radius=fixed_target_radius)
            g_result = run_gonogo_trial(
                win=win, mouse=mouse, circle=circle,
                cross_hit_radius=GONOGO_CROSS_HIT_RADIUS, luminance=lum,
                target_pos=target_pos, data_outlet=data_outlet,
                marker_outlet=marker_outlet, trial_number=g_index + 1,
            )
            if MOUSE_POST_CLICK_FIXATION_S > 0 and not g_result.get("timeout"):
                show_fixation(win, duration_s=MOUSE_POST_CLICK_FIXATION_S)
            this_exp.addData("gonogo_trial", g_index + 1)
            this_exp.addData("gonogo_luminance", g_result["luminance"])
            this_exp.addData("gonogo_clicked", g_result["clicked"])
            this_exp.addData("gonogo_correct", g_result["correct"])
            this_exp.addData("gonogo_rt", g_result["rt"])
            this_exp.addData("gonogo_click_x", g_result["click_x"])
            this_exp.addData("gonogo_click_y", g_result["click_y"])
            this_exp.nextEntry()
        # Restore the circle to the targeting default (white) after the block.
        circle.fillColor = "white"
        circle.lineColor = "white"

        # ── Stream health check: before video block ────────────────
        health = check_lsl_streams(marker_outlet, context="pre_video_block")
        health_warn = format_stream_health_warning(health)
        if health_warn and not no_hardware:
            wait_for_space(
                win,
                f"{health_warn}\n\nPress SPACE to continue to videos anyway.",
                height=0.04,
            )

        run_video_block(
            win=win,
            marker_outlet=marker_outlet,
            this_exp=this_exp,
            this_dir=this_dir,
            av_video_delay_override_s=av_video_delay_override_s,
            video_selector_csv=runtime_opts.video_ids,
        )

        safe_push(marker_outlet, "experiment_end")
        experiment_end_sent = True

        # Stop the LabRecorder recording for the operator (RCS) so the XDF is
        # saved automatically. Falls back to asking them to stop it by hand.
        rcs_stopped = labrecorder_rcs("stop")
        safe_push(marker_outlet, f"config,labrecorder_rcs_stop:{1 if rcs_stopped else 0}")

        # Participant-facing end screen: just a thank-you. LabRecorder status is not
        # the participant's concern, so it is hidden when auto-stop succeeded. Only if
        # auto-stop FAILED do we surface an operator instruction to stop it by hand.
        end_text = I18N.screen("end_screen", EXPERIMENT_LANG)
        if not rcs_stopped:
            # Operator-facing (English): auto-stop failed, stop LabRecorder by hand.
            end_text = (
                "Operator: STOP LabRecorder now and save the XDF (auto-stop failed).\n\n"
                + end_text
            )
        wait_for_space(win, end_text, height=0.07)

    except KeyboardInterrupt:
        logging.info("Experiment interrupted by escape key.")
    except Exception as exc:
        logging.error(f"Unhandled error: {exc}")
        try:
            gui.popupError(f"Experiment failed:\n{exc}")
        except Exception:
            pass
    finally:
        if not experiment_end_sent:
            safe_push(marker_outlet, "experiment_end,aborted:1")
            # Abnormal end (escape/abort/exception). If we auto-started a
            # LabRecorder recording, stop it so it does not keep running into the
            # next participant's session and mislabel their data.
            if rcs_started:
                try:
                    labrecorder_rcs("stop")
                except Exception:
                    pass
        try:
            this_exp.saveAsWideText(data_file + ".csv", delim="auto")
            this_exp.saveAsPickle(data_file)
            this_exp.abort()
        except Exception:
            pass

        try:
            if exp_info.get("run_mode", "task") == "task":
                spawn_quick_qc(exp_info.get("participant", "unknown"), session_start_mtime)
        except Exception:
            pass

        if win is not None:
            try:
                win.close()
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except SystemExit:
        raise
    except BaseException as fatal_exc:
        # Any unexpected crash that reached here would otherwise just close the
        # window with no message. Show the same copyable diagnostic instead.
        try:
            present_fatal_error(
                "unexpected_error",
                fatal_exc,
                plain_explanation=(
                    "The experiment stopped because of an unexpected error.\n"
                    "No more data will be collected for this run."
                ),
            )
        except Exception:
            pass
        raise SystemExit(1)
