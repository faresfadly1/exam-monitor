# Architecture

```text
CCTV/Webcam -> OpenCV Capture -> YOLO Detection + Pose -> Behavior Analyzer -> Alerts/Logs -> Dashboard (optional)
```

Core pipeline:

1. Video stream ingestion with OpenCV.
2. Person/phone detection with YOLO.
3. Pose-based attention checks (paper vs screen/camera).
4. Temporal risk scoring for stable verdicts.
5. Alerting + evidence logging + incident reporting.
