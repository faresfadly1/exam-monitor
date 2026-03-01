import argparse
import csv
import os
import sys
import time
from collections import Counter
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Dict, List, Optional, Tuple

if sys.version_info >= (3, 13) or sys.version_info.releaselevel != "final":
    raise RuntimeError(
        "Use a stable Python version (3.10-3.12). "
        f"Current version: {sys.version.split()[0]}"
    )

import cv2
import numpy as np
from ultralytics import YOLO


PERSON_LABEL = "person"
PHONE_LABEL = "cell phone"


@dataclass
class Track:
    track_id: int
    center_history: Deque[Tuple[float, Tuple[int, int]]] = field(default_factory=lambda: deque(maxlen=120))
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    smooth_bbox: Optional[Tuple[int, int, int, int]] = None
    last_seen: float = 0.0
    risk_ema: float = 0.0
    cheating_since: Optional[float] = None
    suspicious_since: Optional[float] = None


def decision_from_events(confirmed_events: List[str]) -> Tuple[str, int, str]:
    events = set(confirmed_events)
    if "LOOKING_AT_PHONE" in events:
        return "CHEATING", 99, "LOOKING_AT_PHONE"
    if "PHONE_SIDEWAYS" in events:
        return "CHEATING", 98, "PHONE_SIDEWAYS_OR_HIDDEN"
    if "POSSIBLE_HIDDEN_PHONE" in events:
        return "CHEATING", 97, "POSSIBLE_HIDDEN_PHONE"
    if "NOT_LOOKING_AT_PAPER" in events:
        return "CHEATING", 93, "NOT_LOOKING_AT_PAPER"
    if "LOOKING_AT_SCREEN_OR_CAMERA" in events:
        return "CHEATING", 94, "LOOKING_AT_SCREEN_OR_CAMERA"
    if "HOLDING_PHONE" in events or "PHONE_DETECTED" in events:
        return "CHEATING", 95, "PHONE_USE"
    if "POSSIBLE_TALKING" in events and "MOVING" in events:
        return "CHEATING", 85, "TALKING_AND_MOVING"
    if "LOOKING_AT_PAPER" in events:
        return "NOT_CHEATING", 95, "FOCUSED_ON_PAPER"
    if "POSSIBLE_TALKING" in events:
        return "SUSPICIOUS", 75, "POSSIBLE_TALKING"
    if "MOVING" in events:
        return "SUSPICIOUS", 65, "EXCESSIVE_MOVEMENT"
    if "STATIONARY" in events:
        return "NOT_CHEATING", 10, "STATIONARY"
    return "CLEAR", 0, "NO_RISK_SIGNALS"


def instantaneous_risk(confirmed_events: List[str]) -> Tuple[float, str]:
    events = set(confirmed_events)
    # Hard-evidence cues with very high weight.
    if "LOOKING_AT_PHONE" in events:
        return 1.0, "LOOKING_AT_PHONE"
    if "PHONE_SIDEWAYS" in events:
        return 0.97, "PHONE_SIDEWAYS_OR_HIDDEN"
    if "HOLDING_PHONE" in events or "PHONE_DETECTED" in events:
        return 0.93, "PHONE_USE"
    if "POSSIBLE_HIDDEN_PHONE" in events:
        return 0.9, "POSSIBLE_HIDDEN_PHONE"
    if "LOOKING_AT_SCREEN_OR_CAMERA" in events:
        return 0.9, "LOOKING_AT_SCREEN_OR_CAMERA"

    score = 0.0
    reason = "NO_RISK_SIGNALS"
    if "NOT_LOOKING_AT_PAPER" in events:
        score += 0.72
        reason = "NOT_LOOKING_AT_PAPER"
    if "FACE_TURNED_AWAY" in events:
        score += 0.18
        reason = "FACE_TURNED_AWAY"
    if "POSSIBLE_TALKING" in events:
        score += 0.2
        reason = "POSSIBLE_TALKING"
    if "MOVING" in events:
        score += 0.1
        reason = "EXCESSIVE_MOVEMENT"
    if "LOOKING_AT_PAPER" in events:
        score -= 0.35
        reason = "FOCUSED_ON_PAPER"
    if "STATIONARY" in events and score < 0.4:
        score -= 0.15
        reason = "STATIONARY"
    return float(np.clip(score, 0.0, 1.0)), reason


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exam monitoring MVP with YOLOv8 + behavior heuristics.")
    parser.add_argument(
        "--camera",
        type=int,
        default=-1,
        help="Camera index for OpenCV capture. Use -1 to auto-probe index 0..5.",
    )
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model path.")
    parser.add_argument("--pose-model", type=str, default="yolov8n-pose.pt", help="YOLO pose model path.")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold.")
    parser.add_argument("--pose-conf", type=float, default=0.35, help="Pose confidence threshold.")
    parser.add_argument("--person-conf-min", type=float, default=0.35, help="Min confidence for person detections.")
    parser.add_argument("--phone-conf-min", type=float, default=0.18, help="Min confidence for phone detections.")
    parser.add_argument("--window-sec", type=float, default=3.0, help="Time window for movement analysis.")
    parser.add_argument("--stationary-px", type=float, default=25.0, help="<= distance in window means stationary.")
    parser.add_argument("--moving-px", type=float, default=100.0, help=">= distance in window means moving too much.")
    parser.add_argument("--match-px", type=float, default=120.0, help="Max centroid distance for ID association.")
    parser.add_argument("--talk-distance-px", type=float, default=180.0, help="Proximity threshold for talking heuristic.")
    parser.add_argument("--stationary-sec", type=float, default=3.0, help="Seconds stationary before confirmation.")
    parser.add_argument("--moving-sec", type=float, default=1.5, help="Seconds moving before confirmation.")
    parser.add_argument("--talk-sec", type=float, default=2.0, help="Seconds close to another student before confirmation.")
    parser.add_argument("--phone-sec", type=float, default=0.2, help="Seconds of phone overlap before confirmation.")
    parser.add_argument("--paper-sec", type=float, default=1.0, help="Seconds looking at paper before safe confirmation.")
    parser.add_argument("--off-paper-sec", type=float, default=0.8, help="Seconds not looking at paper before cheating.")
    parser.add_argument("--phone-near-head-ratio", type=float, default=0.45, help="Top-body ratio considered near head.")
    parser.add_argument("--phone-sideways-ratio", type=float, default=1.2, help="Width/height ratio for sideways phone.")
    parser.add_argument("--face-away-sec", type=float, default=0.8, help="Seconds facing away before confirmation.")
    parser.add_argument("--hidden-phone-sec", type=float, default=0.8, help="Seconds for hidden-phone heuristic confirmation.")
    parser.add_argument("--risk-alpha", type=float, default=0.3, help="EMA blending factor for temporal risk score.")
    parser.add_argument("--risk-cheating-threshold", type=float, default=0.78, help="Risk threshold for cheating verdict.")
    parser.add_argument("--risk-suspicious-threshold", type=float, default=0.45, help="Risk threshold for suspicious verdict.")
    parser.add_argument("--min-cheating-hold-sec", type=float, default=0.5, help="Seconds risk must stay high before CHEATING.")
    parser.add_argument("--min-suspicious-hold-sec", type=float, default=0.6, help="Seconds risk must stay medium before SUSPICIOUS.")
    parser.add_argument("--bbox-smooth-alpha", type=float, default=0.35, help="BBox smoothing alpha (higher = less smoothing).")
    parser.add_argument("--event-cooldown-sec", type=float, default=5.0, help="Minimum time between repeated logs per event.")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for CSV log and evidence frames.")
    parser.add_argument("--calibrate-sec", type=float, default=0.0, help="Optional baseline calibration seconds before monitoring.")
    parser.add_argument("--alarm-cooldown-sec", type=float, default=1.0, help="Minimum seconds between alarm beeps.")
    parser.add_argument("--alarm-on-suspicious", action="store_true", help="Also alarm on SUSPICIOUS verdict.")
    parser.add_argument("--mute-alarm", action="store_true", help="Disable audible alarm.")
    return parser.parse_args()


def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter_area / float(area_a + area_b - inter_area)


def centroid(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def expand_bbox(bbox: Tuple[int, int, int, int], pad_ratio: float, width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    pad_x = int(w * pad_ratio)
    pad_y = int(h * pad_ratio)
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width - 1, x2 + pad_x),
        min(height - 1, y2 + pad_y),
    )


def point_in_bbox(point: Tuple[int, int], bbox: Tuple[int, int, int, int]) -> bool:
    px, py = point
    x1, y1, x2, y2 = bbox
    return x1 <= px <= x2 and y1 <= py <= y2


def looking_at_paper_from_keypoints(keypoints_xy: np.ndarray, keypoints_conf: np.ndarray) -> bool:
    # COCO keypoints: 0=nose, 1=left eye, 2=right eye, 5=left shoulder, 6=right shoulder
    needed = [0, 1, 2, 5, 6]
    if any(keypoints_conf[idx] < 0.25 for idx in needed):
        return False
    nose_y = float(keypoints_xy[0][1])
    eye_y = float((keypoints_xy[1][1] + keypoints_xy[2][1]) / 2.0)
    shoulder_y = float((keypoints_xy[5][1] + keypoints_xy[6][1]) / 2.0)
    denom = max(1.0, shoulder_y - eye_y)
    down_ratio = (nose_y - eye_y) / denom
    return down_ratio >= 0.52


def face_turned_away_from_keypoints(keypoints_xy: np.ndarray, keypoints_conf: np.ndarray) -> bool:
    # COCO keypoints: 1/2 eyes, 3/4 ears
    if keypoints_conf[1] < 0.25 or keypoints_conf[2] < 0.25:
        return False
    left_eye_x = float(keypoints_xy[1][0])
    right_eye_x = float(keypoints_xy[2][0])
    eye_dist = abs(right_eye_x - left_eye_x)
    if eye_dist < 2.0:
        return False
    left_ear_ok = keypoints_conf[3] >= 0.2
    right_ear_ok = keypoints_conf[4] >= 0.2
    if left_ear_ok and not right_ear_ok:
        return True
    if right_ear_ok and not left_ear_ok:
        return True
    if left_ear_ok and right_ear_ok:
        left_ear_x = float(keypoints_xy[3][0])
        right_ear_x = float(keypoints_xy[4][0])
        left_span = abs(left_eye_x - left_ear_x)
        right_span = abs(right_ear_x - right_eye_x)
        ratio = (max(left_span, right_span) + 1.0) / (min(left_span, right_span) + 1.0)
        return ratio >= 2.2
    return False


def wrists_hidden_from_keypoints(keypoints_conf: np.ndarray) -> bool:
    # COCO keypoints: 5/6 shoulders, 9/10 wrists
    shoulders_visible = keypoints_conf[5] >= 0.25 and keypoints_conf[6] >= 0.25
    wrists_missing = keypoints_conf[9] < 0.2 and keypoints_conf[10] < 0.2
    return shoulders_visible and wrists_missing


def facing_front_from_keypoints(keypoints_xy: np.ndarray, keypoints_conf: np.ndarray) -> bool:
    # Approximate "facing screen/camera" using symmetric eyes and visible nose.
    if keypoints_conf[0] < 0.25 or keypoints_conf[1] < 0.25 or keypoints_conf[2] < 0.25:
        return False
    left_eye_x = float(keypoints_xy[1][0])
    right_eye_x = float(keypoints_xy[2][0])
    nose_x = float(keypoints_xy[0][0])
    eye_mid_x = (left_eye_x + right_eye_x) / 2.0
    eye_dist = abs(right_eye_x - left_eye_x)
    if eye_dist < 2.0:
        return False
    nose_center_offset = abs(nose_x - eye_mid_x) / eye_dist
    return nose_center_offset <= 0.22


def smooth_bbox(
    prev_bbox: Optional[Tuple[int, int, int, int]],
    curr_bbox: Tuple[int, int, int, int],
    alpha: float,
) -> Tuple[int, int, int, int]:
    if prev_bbox is None:
        return curr_bbox
    a = float(np.clip(alpha, 0.05, 0.95))
    out = []
    for p, c in zip(prev_bbox, curr_bbox):
        out.append(int(round((1.0 - a) * p + a * c)))
    return tuple(out)  # type: ignore[return-value]


def movement_in_window(history: Deque[Tuple[float, Tuple[int, int]]], now_ts: float, window_sec: float) -> float:
    points = [(t, p) for t, p in history if now_ts - t <= window_sec]
    if len(points) < 2:
        return 0.0
    return distance(points[0][1], points[-1][1])


def ensure_log_files(log_dir: str) -> Tuple[str, str]:
    evidence_dir = os.path.join(log_dir, "evidence")
    os.makedirs(evidence_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, "events.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "track_id", "event", "verdict", "confidence", "details", "evidence_file"])
    return csv_path, evidence_dir


def append_event(
    csv_path: str,
    track_id: int,
    event_name: str,
    verdict: str,
    confidence: int,
    details: str,
    evidence_file: str,
) -> None:
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                datetime.now().isoformat(timespec="seconds"),
                track_id,
                event_name,
                verdict,
                confidence,
                details,
                evidence_file,
            ]
        )


def trigger_alarm(muted: bool) -> None:
    if muted:
        return
    # Terminal bell fallback, works without extra dependencies.
    print("\a", end="", flush=True)


def export_incident_report(csv_path: str, log_dir: str, session_start_ts: float, session_end_ts: float) -> str:
    summary_path = os.path.join(log_dir, "incident_report.txt")
    if not os.path.exists(csv_path):
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("No events log found.\n")
        return summary_path

    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    verdict_counts = Counter(row["verdict"] for row in rows if row.get("verdict"))
    event_counts = Counter(row["event"] for row in rows if row.get("event"))
    track_counts = Counter(row["track_id"] for row in rows if row.get("track_id"))

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Exam Monitoring Incident Report\n")
        f.write("=" * 32 + "\n")
        f.write(f"Session start: {datetime.fromtimestamp(session_start_ts).isoformat(timespec='seconds')}\n")
        f.write(f"Session end:   {datetime.fromtimestamp(session_end_ts).isoformat(timespec='seconds')}\n")
        f.write(f"Duration sec:  {int(session_end_ts - session_start_ts)}\n")
        f.write(f"Total events:  {len(rows)}\n\n")

        f.write("Verdict counts:\n")
        for verdict in ["CHEATING", "SUSPICIOUS", "NOT_CHEATING", "CLEAR"]:
            f.write(f"- {verdict}: {verdict_counts.get(verdict, 0)}\n")

        f.write("\nEvent counts:\n")
        for event_name, count in sorted(event_counts.items(), key=lambda x: (-x[1], x[0])):
            f.write(f"- {event_name}: {count}\n")

        f.write("\nTop track IDs by event count:\n")
        for track_id, count in track_counts.most_common(10):
            f.write(f"- ID {track_id}: {count}\n")

    return summary_path


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    pose_model = YOLO(args.pose_model)

    cap = None
    candidate_indices = [args.camera] if args.camera >= 0 else list(range(0, 6))
    selected_index: Optional[int] = None
    for idx in candidate_indices:
        test_cap = cv2.VideoCapture(idx)
        if test_cap.isOpened():
            cap = test_cap
            selected_index = idx
            break
        test_cap.release()

    if cap is None or selected_index is None:
        raise RuntimeError(
            "Could not open any camera (tried indices 0..5).\n"
            "On macOS, allow camera access for the app running Python (Terminal or VS Code):\n"
            "System Settings -> Privacy & Security -> Camera -> enable your terminal app.\n"
            "Then fully close and reopen that app and run again.\n"
            "You can also try: python monitor.py --camera 1"
        )

    csv_path, evidence_dir = ensure_log_files(args.log_dir)
    tracks: Dict[int, Track] = {}
    next_track_id = 1
    event_last_time: Dict[Tuple[int, str], float] = defaultdict(float)
    event_active_since: Dict[Tuple[int, str], float] = {}
    session_start_ts = time.time()
    last_alarm_ts = 0.0
    calibration_done = args.calibrate_sec <= 0.0
    calibration_start_ts = time.time()
    calibration_movement_samples: List[float] = []
    calibration_neighbor_samples: List[float] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now_ts = time.time()

        result = model(frame, conf=args.conf, verbose=False)[0]
        pose_result = pose_model(frame, conf=args.pose_conf, verbose=False)[0]
        person_boxes: List[Tuple[int, int, int, int]] = []
        phone_boxes: List[Tuple[int, int, int, int]] = []

        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox = (x1, y1, x2, y2)
            if label == PERSON_LABEL and conf >= args.person_conf_min:
                person_boxes.append(bbox)
            elif label == PHONE_LABEL and conf >= args.phone_conf_min:
                phone_boxes.append(bbox)

        pose_persons: List[Tuple[Tuple[int, int, int, int], np.ndarray, np.ndarray]] = []
        if pose_result.boxes is not None and pose_result.keypoints is not None:
            for i, pbox in enumerate(pose_result.boxes):
                cls = int(pbox.cls[0])
                if pose_model.names[cls] != PERSON_LABEL:
                    continue
                px1, py1, px2, py2 = map(int, pbox.xyxy[0])
                kxy = pose_result.keypoints.xy[i].cpu().numpy()
                if pose_result.keypoints.conf is None:
                    kconf = np.ones((kxy.shape[0],), dtype=np.float32)
                else:
                    kconf = pose_result.keypoints.conf[i].cpu().numpy()
                pose_persons.append(((px1, py1, px2, py2), kxy, kconf))

        assigned_tracks: Dict[int, Tuple[int, int, int, int]] = {}
        used_track_ids = set()
        for pb in person_boxes:
            pc = centroid(pb)
            best_tid = None
            best_dist = float("inf")
            for tid, tr in tracks.items():
                if tid in used_track_ids or now_ts - tr.last_seen > 2.0:
                    continue
                d = distance(pc, centroid(tr.bbox))
                if d < best_dist and d <= args.match_px:
                    best_dist = d
                    best_tid = tid
            if best_tid is None:
                best_tid = next_track_id
                next_track_id += 1
                tracks[best_tid] = Track(track_id=best_tid)
            used_track_ids.add(best_tid)
            assigned_tracks[best_tid] = pb

        for tid, pb in assigned_tracks.items():
            tr = tracks[tid]
            tr.bbox = pb
            tr.last_seen = now_ts
            tr.center_history.append((now_ts, centroid(pb)))
            tr.smooth_bbox = smooth_bbox(tr.smooth_bbox, pb, args.bbox_smooth_alpha)

        stale_ids = [tid for tid, tr in tracks.items() if now_ts - tr.last_seen > 10.0]
        for tid in stale_ids:
            del tracks[tid]
            keys_to_drop = [key for key in event_active_since if key[0] == tid]
            for key in keys_to_drop:
                event_active_since.pop(key, None)

        person_centers: Dict[int, Tuple[int, int]] = {tid: centroid(pb) for tid, pb in assigned_tracks.items()}
        proximity_pairs = set()
        ids = list(person_centers.keys())
        nearest_dist_per_id: Dict[int, float] = {}
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a_id, b_id = ids[i], ids[j]
                d = distance(person_centers[a_id], person_centers[b_id])
                if d <= args.talk_distance_px:
                    proximity_pairs.add((a_id, b_id))
                nearest_dist_per_id[a_id] = min(d, nearest_dist_per_id.get(a_id, float("inf")))
                nearest_dist_per_id[b_id] = min(d, nearest_dist_per_id.get(b_id, float("inf")))

        if not calibration_done:
            elapsed = now_ts - calibration_start_ts
            for tid, pb in assigned_tracks.items():
                moved = movement_in_window(tracks[tid].center_history, now_ts, args.window_sec)
                if moved > 0:
                    calibration_movement_samples.append(moved)
                nearest = nearest_dist_per_id.get(tid, float("inf"))
                if np.isfinite(nearest):
                    calibration_neighbor_samples.append(nearest)

            if elapsed >= args.calibrate_sec:
                if calibration_movement_samples:
                    base_move = float(np.percentile(np.array(calibration_movement_samples), 75))
                    args.stationary_px = max(12.0, base_move * 0.55)
                    args.moving_px = max(args.stationary_px + 18.0, base_move * 2.1)
                if calibration_neighbor_samples:
                    base_neighbor = float(np.percentile(np.array(calibration_neighbor_samples), 20))
                    args.talk_distance_px = max(90.0, base_neighbor * 0.85)
                calibration_done = True

        red_count = 0
        verdict_counts = defaultdict(int)
        for tid, pb in assigned_tracks.items():
            tr = tracks[tid]
            draw_bbox = tr.smooth_bbox if tr.smooth_bbox is not None else pb
            x1, y1, x2, y2 = draw_bbox
            moved_dist = movement_in_window(tr.center_history, now_ts, args.window_sec)

            has_history = len(tr.center_history) > 6
            is_stationary = moved_dist <= args.stationary_px and has_history
            is_moving = moved_dist >= args.moving_px
            frame_h, frame_w = frame.shape[:2]
            expanded = expand_bbox(pb, pad_ratio=0.15, width=frame_w, height=frame_h)
            related_phones = []
            for ph in phone_boxes:
                ph_center = centroid(ph)
                if iou(pb, ph) > 0.01 or point_in_bbox(ph_center, expanded):
                    related_phones.append(ph)
            has_phone = len(related_phones) > 0
            phone_sideways = False
            phone_near_head = False
            for ph in related_phones:
                px1, py1, px2, py2 = ph
                pw = max(1, px2 - px1)
                ph_h = max(1, py2 - py1)
                ratio = pw / float(ph_h)
                if ratio >= args.phone_sideways_ratio:
                    phone_sideways = True
                cpx, cpy = centroid(ph)
                person_h = max(1, y2 - y1)
                if cpy <= y1 + args.phone_near_head_ratio * person_h:
                    phone_near_head = True

            looking_at_paper = False
            face_turned_away = False
            facing_front = False
            wrists_hidden = False
            pose_available = False
            best_pose_iou = 0.0
            for pose_bbox, kxy, kconf in pose_persons:
                overlap = iou(pb, pose_bbox)
                if overlap > best_pose_iou:
                    best_pose_iou = overlap
                    looking_at_paper = looking_at_paper_from_keypoints(kxy, kconf)
                    face_turned_away = face_turned_away_from_keypoints(kxy, kconf)
                    facing_front = facing_front_from_keypoints(kxy, kconf)
                    wrists_hidden = wrists_hidden_from_keypoints(kconf)
                    pose_available = True
            possible_talking = any(tid in pair for pair in proximity_pairs)
            not_looking_at_paper = pose_available and (not looking_at_paper)
            looking_at_screen_or_camera = not_looking_at_paper and facing_front and (not has_phone)
            possible_hidden_phone = (not has_phone) and face_turned_away and wrists_hidden and not_looking_at_paper

            raw_events = {
                "STATIONARY": is_stationary,
                "MOVING": is_moving,
                "POSSIBLE_TALKING": possible_talking,
                "PHONE_DETECTED": has_phone,
                "HOLDING_PHONE": has_phone,
                "PHONE_SIDEWAYS": phone_sideways,
                "LOOKING_AT_PHONE": phone_near_head and has_phone,
                "LOOKING_AT_PAPER": looking_at_paper and (not has_phone),
                "NOT_LOOKING_AT_PAPER": not_looking_at_paper,
                "LOOKING_AT_SCREEN_OR_CAMERA": looking_at_screen_or_camera,
                "FACE_TURNED_AWAY": face_turned_away and not_looking_at_paper,
                "POSSIBLE_HIDDEN_PHONE": possible_hidden_phone,
            }
            min_sec = {
                "STATIONARY": args.stationary_sec,
                "MOVING": args.moving_sec,
                "POSSIBLE_TALKING": args.talk_sec,
                "PHONE_DETECTED": args.phone_sec,
                "HOLDING_PHONE": args.phone_sec,
                "PHONE_SIDEWAYS": args.phone_sec,
                "LOOKING_AT_PHONE": args.phone_sec,
                "LOOKING_AT_PAPER": args.paper_sec,
                "NOT_LOOKING_AT_PAPER": args.off_paper_sec,
                "LOOKING_AT_SCREEN_OR_CAMERA": args.off_paper_sec,
                "FACE_TURNED_AWAY": args.face_away_sec,
                "POSSIBLE_HIDDEN_PHONE": args.hidden_phone_sec,
            }
            confirmed_events: List[str] = []
            for event_name, is_active in raw_events.items():
                key = (tid, event_name)
                if is_active:
                    if key not in event_active_since:
                        event_active_since[key] = now_ts
                    duration = now_ts - event_active_since[key]
                    if duration >= min_sec[event_name]:
                        confirmed_events.append(event_name)
                else:
                    event_active_since.pop(key, None)

            if not calibration_done:
                verdict, confidence, primary_reason = ("CALIBRATING", 0, "LEARNING_BASELINE")
            else:
                inst_risk, risk_reason = instantaneous_risk(confirmed_events)
                alpha = float(np.clip(args.risk_alpha, 0.01, 0.95))
                tr.risk_ema = (1.0 - alpha) * tr.risk_ema + alpha * inst_risk

                if tr.risk_ema >= args.risk_cheating_threshold:
                    if tr.cheating_since is None:
                        tr.cheating_since = now_ts
                else:
                    tr.cheating_since = None

                if tr.risk_ema >= args.risk_suspicious_threshold:
                    if tr.suspicious_since is None:
                        tr.suspicious_since = now_ts
                else:
                    tr.suspicious_since = None

                hard_cheating = any(
                    ev in confirmed_events
                    for ev in ["LOOKING_AT_PHONE", "PHONE_SIDEWAYS", "HOLDING_PHONE", "PHONE_DETECTED"]
                )
                cheating_ready = tr.cheating_since is not None and (now_ts - tr.cheating_since) >= args.min_cheating_hold_sec
                if hard_cheating or cheating_ready:
                    verdict = "CHEATING"
                    confidence = int(np.clip(70 + tr.risk_ema * 30, 70, 99))
                    primary_reason = risk_reason
                else:
                    suspicious_ready = (
                        tr.suspicious_since is not None
                        and (now_ts - tr.suspicious_since) >= args.min_suspicious_hold_sec
                    )
                    if suspicious_ready:
                        verdict = "SUSPICIOUS"
                        confidence = int(np.clip(45 + tr.risk_ema * 45, 45, 89))
                        primary_reason = risk_reason
                    elif "LOOKING_AT_PAPER" in confirmed_events:
                        verdict = "NOT_CHEATING"
                        confidence = 95
                        primary_reason = "FOCUSED_ON_PAPER"
                    elif tr.risk_ema < 0.15 and "STATIONARY" in confirmed_events:
                        verdict = "NOT_CHEATING"
                        confidence = 70
                        primary_reason = "STATIONARY"
                    else:
                        verdict = "CLEAR"
                        confidence = int(np.clip((1.0 - tr.risk_ema) * 60, 0, 60))
                        primary_reason = "NO_RISK_SIGNALS"
            verdict_counts[verdict] += 1
            alert = verdict in {"CHEATING", "SUSPICIOUS"}
            if alert:
                red_count += 1
            color = (0, 0, 255) if alert else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"ID {tid} {verdict} ({confidence}%)",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
            detail_text = primary_reason if primary_reason else "NO_RISK_SIGNALS"
            cv2.putText(
                frame,
                detail_text.replace("_", " "),
                (x1, min(frame.shape[0] - 8, y2 + 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )

            for reason in confirmed_events:
                if not calibration_done:
                    continue
                if reason in {"STATIONARY", "LOOKING_AT_PAPER"}:
                    continue
                key = (tid, reason)
                if now_ts - event_last_time[key] < args.event_cooldown_sec:
                    continue
                event_last_time[key] = now_ts
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{stamp}_id{tid}_{verdict}_{reason}.jpg"
                out_path = os.path.join(evidence_dir, filename)
                cv2.imwrite(out_path, frame)
                append_event(
                    csv_path=csv_path,
                    track_id=tid,
                    event_name=reason,
                    verdict=verdict,
                    confidence=confidence,
                    details=f"reason={primary_reason}; moved_dist={moved_dist:.1f}px in {args.window_sec:.1f}s",
                    evidence_file=out_path,
                )

        should_alarm = verdict_counts["CHEATING"] > 0
        if args.alarm_on_suspicious:
            should_alarm = should_alarm or verdict_counts["SUSPICIOUS"] > 0
        if calibration_done and should_alarm and now_ts - last_alarm_ts >= args.alarm_cooldown_sec:
            trigger_alarm(muted=args.mute_alarm)
            last_alarm_ts = now_ts

        for ph in phone_boxes:
            x1, y1, x2, y2 = ph
            cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 255), 1)
            cv2.putText(frame, "PHONE", (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 255), 1)

        cv2.putText(
            frame,
            (
                f"People: {len(assigned_tracks)} | Alerts: {red_count} | "
                f"Cheating: {verdict_counts['CHEATING']} | Suspicious: {verdict_counts['SUSPICIOUS']} | q=quit"
            ),
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255) if red_count else (50, 220, 50),
            2,
        )
        if not calibration_done:
            remaining = max(0.0, args.calibrate_sec - (now_ts - calibration_start_ts))
            cv2.putText(
                frame,
                f"Calibrating... {remaining:.1f}s",
                (12, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 215, 255),
                2,
            )
        cv2.imshow("Exam Monitoring MVP", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    report_path = export_incident_report(
        csv_path=csv_path,
        log_dir=args.log_dir,
        session_start_ts=session_start_ts,
        session_end_ts=time.time(),
    )
    print(f"Incident report exported: {report_path}")


if __name__ == "__main__":
    main()
