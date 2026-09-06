# TikTok Eye-Tracking Pilot Study

This repository contains the Python scripts used to run and analyse the pilot study reported in the Final Degree Project *Visual Attention and Virality in Cosmetics Advertising on TikTok*.

The study compared 13 viral and 13 flop cosmetics advertisements. Participants completed two mouse tasks, watched the advertisements while gaze and pupil data were recorded with a Gazepoint GP3 HD eye tracker, and rated each advertisement afterwards.

## Repository structure

- `experiment/`: PsychoPy task, Gazepoint LSL stream, translations and session quality-control code.
- `analysis/`: gaze synchrony, editing rhythm, post-cut sensitivity, ratings, mouse-task and exploratory prediction analyses.

## Main experiment

The experiment was run with `experiment/mouse_lsl_task_stable.py`. The participant-facing interface is available in English, Spanish and Catalan through `experiment/translations.py`. Gazepoint data were published to Lab Streaming Layer using `experiment/stream_gazepoint.py` and recorded together with task markers.

The task expects the 26 stimulus videos and their external audio sidecars in a local `data/Videos` directory. These media files are not included because the original advertisements may be protected by copyright.

## Analysis

The `analysis` directory contains the scripts supporting the analyses reported in the thesis. The main workflow includes:

1. preprocessing and quality control of gaze and pupil recordings;
2. leave-one-participant-out gaze synchrony;
3. clip-level editing-rhythm analyses;
4. removal of post-cut gaze intervals and matched random-removal controls;
5. subjective-rating and mouse-task analyses;
6. exploratory advertisement-classification and liking-prediction models; and
7. generation of confidence intervals and thesis figures.

Some scripts expect intermediate CSV files produced by earlier stages of the workflow. Their command-line help describes the required input directories and output locations.

## Requirements

The experiment requires Python, PsychoPy and Lab Streaming Layer. The analysis scripts use NumPy, pandas, SciPy, statsmodels, scikit-learn, XGBoost and SHAP. FFmpeg is required for video and audio checks.

Install the Python packages in a separate environment using:

```bash
python -m pip install -r requirements.txt
```

Hardware-specific drivers and Gazepoint Control must be installed separately when collecting new eye-tracking data.

## Data and privacy

Participant-level CSV and XDF recordings are not included. They are covered by the approved ethical and data-management procedures. The repository also excludes participant identifiers, consent records, videos, audio files and local machine configuration.

Because the participant-level data are not public, the complete numerical analysis cannot be rerun from this repository alone. The scripts are provided to document the executed computational workflow and support methodological transparency.

## Study status

This was an observational pilot study. The scripts and reported findings should not be treated as a validated system for predicting whether a new advertisement will become viral.

