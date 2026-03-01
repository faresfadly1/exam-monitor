# Architecture Flow

```text
Camera Input
  -> Frame Capture (OpenCV)
  -> Object Detection (YOLOv8: person + phone)
  -> Pose Estimation (keypoints)
  -> Tracking (temporary IDs)
  -> Behavior Heuristics
  -> Risk Engine
  -> Alerts & Logging
```

Detailed stages:

1. Camera captures live frames from webcam/CCTV.
2. OpenCV reads and forwards frames to the inference pipeline.
3. YOLOv8 detects persons and mobile phones.
4. Pose model estimates keypoints for attention/pose cues.
5. Tracking module assigns temporary IDs to detected individuals.
6. Heuristic engine analyzes behavior (moving, talking proximity, phone near head, off-paper focus).
7. Risk engine assigns final verdict (`CHEATING`, `SUSPICIOUS`, `NOT_CHEATING`, `CLEAR`).
8. Alert and logging module outputs overlays, alarms, CSV logs, and evidence frames.
