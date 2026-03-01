# Exam Monitoring MVP

## Project Overview

This project uses YOLOv8 + OpenCV to:

- detect `person` and `cell phone`
- assign temporary IDs to people
- classify each student as `CHEATING`, `SUSPICIOUS`, `NOT_CHEATING`, or `CLEAR`
- log suspicious events and save evidence frames
- play an audible alarm on confirmed cheating
- auto-export an end-of-session incident report
- optionally run baseline calibration before monitoring

## Tech Stack

- Python
- OpenCV
- Ultralytics YOLOv8 (detection + pose)
- NumPy

## Repository Layout

```text
exam-monitor/
├── monitor.py
├── requirements.txt
├── README.md
├── .gitignore
├── docs/
├── models/
└── examples/
```

## Features Included

- `PHONE_DETECTED`: person box overlaps with detected phone.
- `HOLDING_PHONE`: phone is inside/near a person box.
- `PHONE_SIDEWAYS`: phone is held in landscape/sideways orientation.
- `LOOKING_AT_PHONE`: phone near upper body/head area.
- `LOOKING_AT_PAPER`: pose keypoints suggest head-down focus on paper.
- `NOT_LOOKING_AT_PAPER`: strict rule, head not focused on paper.
- `LOOKING_AT_SCREEN_OR_CAMERA`: off-paper + front-facing to camera/screen.
- `POSSIBLE_HIDDEN_PHONE`: strict heuristic (face turned away + hidden wrists + off-paper).
- `MOVING`: person moved more than a configured pixel threshold in the last window.
- `STATIONARY`: person moved less than a configured threshold in the last window.
- `POSSIBLE_TALKING`: two people are very close (proximity heuristic).
- Decision engine with confidence:
  - `CHEATING`: confirmed phone use (holding / sideways / looking at phone), confirmed not-looking-at-paper, looking at screen/camera, hidden-phone heuristic, or confirmed talking+moving.
  - `SUSPICIOUS`: confirmed talking or excessive movement.
  - `NOT_CHEATING`: confirmed paper-focus or stationary behavior.
  - `CLEAR`: no risk signals confirmed.

Note: talking detection here is a heuristic, not speech recognition. For production quality, add face landmarks + mouth activity + directional audio.

## Setup

Use a stable Python version: `3.10`, `3.11`, or `3.12` (do not use Python `3.13` alpha/beta builds).

```bash
cd /path/to/exam-monitor
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you already created a venv with Python 3.13, recreate it:

```bash
cd /path/to/exam-monitor
rm -rf .venv
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```bash
python monitor.py
```

Optional args:

```bash
python monitor.py \
  --camera -1 \
  --model yolov8n.pt \
  --pose-model yolov8n-pose.pt \
  --conf 0.35 \
  --pose-conf 0.35 \
  --person-conf-min 0.35 \
  --phone-conf-min 0.18 \
  --window-sec 3 \
  --stationary-px 25 \
  --moving-px 100 \
  --talk-distance-px 180 \
  --stationary-sec 3 \
  --moving-sec 1.5 \
  --talk-sec 2 \
  --phone-sec 0.2 \
  --paper-sec 1 \
  --off-paper-sec 0.8 \
  --face-away-sec 0.8 \
  --hidden-phone-sec 0.8 \
  --risk-alpha 0.3 \
  --risk-cheating-threshold 0.78 \
  --risk-suspicious-threshold 0.45 \
  --min-cheating-hold-sec 0.5 \
  --min-suspicious-hold-sec 0.6 \
  --bbox-smooth-alpha 0.35 \
  --phone-near-head-ratio 0.45 \
  --phone-sideways-ratio 1.2 \
  --calibrate-sec 10 \
  --alarm-cooldown-sec 1.0 \
  --alarm-on-suspicious
```

Press `q` to quit.

## Demo Media

Project demo video:
[demo.mov](examples/demo.mov)

<video src="examples/demo.mov" controls width="900"></video>

## macOS Camera Permission Fix

If you see `Could not open camera index` or `not authorized to capture video`:

1. Open `System Settings -> Privacy & Security -> Camera`.
2. Enable camera access for the app hosting Python (`Terminal` or `Visual Studio Code`).
3. Fully quit and reopen that app.
4. Retry with auto-probe: `python monitor.py --camera -1`.
5. If needed, try specific indices: `--camera 0`, then `--camera 1`, etc.

## Outputs

- Event log CSV: `logs/events.csv`
- Evidence frames: `logs/evidence/*.jpg`
- CSV now includes: `event`, `verdict`, and `confidence`.
- End-of-session report: `logs/incident_report.txt`

## Calibration Tips

- Increase `--moving-px` if normal posture shifts are causing too many alerts.
- Increase `--stationary-px` if stationary alerts are too strict.
- Tune `--talk-distance-px` by camera angle and classroom density.
- Increase `--paper-sec` if paper-focus gets detected too quickly.
- Decrease `--off-paper-sec` (e.g. `0.5`) to make off-paper cheating alerts trigger faster.
- Increase `--phone-sideways-ratio` if sideways-phone alerts trigger too often.
- Use `--calibrate-sec 10` to auto-tune movement/talking thresholds to your classroom.
- Use `--mute-alarm` if you only want visual alerts.
- Decrease `--bbox-smooth-alpha` (e.g. `0.25`) for smoother boxes/overlay motion.
- Use better lighting and camera placement for stronger phone detection.

## High Accuracy Profile

Use this preset when you want maximum sensitivity for exam proctoring:

```bash
python monitor.py \
  --camera -1 \
  --model yolov8m.pt \
  --pose-model yolov8m-pose.pt \
  --conf 0.28 \
  --pose-conf 0.28 \
  --phone-conf-min 0.14 \
  --off-paper-sec 0.5 \
  --hidden-phone-sec 0.5 \
  --face-away-sec 0.5 \
  --risk-alpha 0.35 \
  --risk-cheating-threshold 0.72 \
  --min-cheating-hold-sec 0.35 \
  --calibrate-sec 12 \
  --alarm-cooldown-sec 0.7
```

This profile is heavier on CPU/GPU but usually improves detection stability and recall.

## Important Limitation

If a phone is fully hidden from the camera view, no vision model can confirm it directly. The app uses `POSSIBLE_HIDDEN_PHONE` as a risk heuristic, not hard proof.
