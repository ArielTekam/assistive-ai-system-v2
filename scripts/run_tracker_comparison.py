import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import csv
import time
import argparse
import psutil
from collections import defaultdict

from core.detector import ObjectDetector
from core.bytetrack_detector import ByteTrackDetector
from core.scene_memory import SceneMemory
from core.context_manager import ContextManager
from core.decision_engine import DecisionEngine


parser = argparse.ArgumentParser()

parser.add_argument("--tracker", required=True, choices=["old", "bytetrack"])
parser.add_argument("--scenario", required=True)
parser.add_argument("--run", type=int, required=True)
parser.add_argument("--video", required=True)

args = parser.parse_args()

TRACKER_TYPE = args.tracker
SCENARIO = args.scenario
RUN_ID = args.run
VIDEO_PATH = args.video

results_dir = Path("Experiments/results_csv")
results_dir.mkdir(parents=True, exist_ok=True)

output_name = f"TRACK_{TRACKER_TYPE}_{SCENARIO}_run{RUN_ID:02d}"
csv_path = results_dir / f"{output_name}.csv"
summary_path = results_dir / f"{output_name}_summary.csv"


old_detector = ObjectDetector(
    model_path="yolo11n.pt",
    conf_threshold=0.25,
    img_size=320
)

bt_detector = ByteTrackDetector(
    model_path="yolo11n.pt",
    conf_threshold=0.25,
    img_size=320,
    tracker_config="bytetrack.yaml"
)

memory = SceneMemory(
    frame_width=640,
    frame_height=480
)

context_manager = ContextManager(
    cooldown_seconds=10.0,
    min_proximity_change=0.15
)

decision_engine = DecisionEngine(
    priority_threshold=0.45,
    high_priority_threshold=0.70
)


unique_ids = set()
track_frames = defaultdict(int)


def detections_to_objects(detections):
    return memory.update(detections)


def bytetrack_to_objects(tracked_objects):
    objects = {}

    for obj in tracked_objects:
        track_id = obj.get("track_id")

        if track_id is None:
            continue

        bbox = obj["bbox"]
        x1, y1, x2, y2 = bbox

        cx = (x1 + x2) / 2
        if cx < 640 * 0.33:
            direction = "left"
        elif cx > 640 * 0.66:
            direction = "right"
        else:
            direction = "center"

        box_area = max(0, x2 - x1) * max(0, y2 - y1)
        frame_area = 640 * 480
        area_norm = box_area / frame_area
        y_bottom = y2 / 480
        proximity = min(1.0, 0.65 * area_norm * 8 + 0.35 * y_bottom)

        objects[track_id] = {
            "id": track_id,
            "label": obj["label"],
            "confidence": obj["confidence"],
            "bbox": bbox,
            "direction": direction,
            "proximity": round(proximity, 3),
            "previous_proximity": round(proximity, 3),
            "risk_score": round(proximity, 3),
            "missing_frames": 0,
        }

        unique_ids.add(track_id)
        track_frames[track_id] += 1

    return objects


cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Erreur: impossible d'ouvrir la vidéo: {VIDEO_PATH}")
    exit()


csv_file = open(csv_path, mode="w", newline="")
writer = csv.writer(csv_file)

writer.writerow([
    "timestamp",
    "tracker",
    "scenario",
    "run_id",
    "frame",
    "fps_instant",
    "latency_ms",
    "cpu_percent",
    "detections_count",
    "objects_count",
    "unique_ids",
    "raw_messages",
    "filtered_messages",
    "decision_messages",
    "mean_priority_score",
    "final_messages"
])


frame_id = 0

fps_values = []
latency_values = []
cpu_values = []

total_detections = 0
total_objects = 0
total_raw_messages = 0
total_filtered_messages = 0
total_decision_messages = 0
priority_scores_all = []

print("===================================")
print("TRACKER COMPARISON STARTED")
print(f"Tracker: {TRACKER_TYPE}")
print(f"Scenario: {SCENARIO}")
print(f"Run: {RUN_ID}")
print(f"Video: {VIDEO_PATH}")
print("===================================")


while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_start = time.time()
    pipeline_start = time.time()

    if TRACKER_TYPE == "old":
        detections = old_detector.detect(frame)
        objects = detections_to_objects(detections)

        detections_count = len(detections)

        for obj_id in objects.keys():
            unique_ids.add(obj_id)
            track_frames[obj_id] += 1

    else:
        tracked_objects = bt_detector.track(frame)
        objects = bytetrack_to_objects(tracked_objects)

        detections_count = len(tracked_objects)

    context_result = context_manager.filter_messages(objects)
    context_messages = context_result["messages"]

    decisions = decision_engine.decide(
        context_messages=context_messages,
        objects=objects
    )

    raw_messages = context_result["raw_messages_count"]
    filtered_messages = len(context_messages)
    decision_messages = len(decisions)

    final_messages = [d["message"] for d in decisions]
    priority_scores = [d["priority_score"] for d in decisions]
    priority_scores_all.extend(priority_scores)

    latency_ms = (time.time() - pipeline_start) * 1000
    fps = 1.0 / max(time.time() - frame_start, 1e-6)
    cpu_percent = psutil.cpu_percent()

    mean_priority_score = (
        sum(priority_scores) / len(priority_scores)
        if priority_scores else 0
    )

    fps_values.append(fps)
    latency_values.append(latency_ms)
    cpu_values.append(cpu_percent)

    total_detections += detections_count
    total_objects += len(objects)
    total_raw_messages += raw_messages
    total_filtered_messages += filtered_messages
    total_decision_messages += decision_messages

    writer.writerow([
        round(time.time(), 3),
        TRACKER_TYPE,
        SCENARIO,
        RUN_ID,
        frame_id,
        round(fps, 2),
        round(latency_ms, 2),
        round(cpu_percent, 2),
        detections_count,
        len(objects),
        len(unique_ids),
        raw_messages,
        filtered_messages,
        decision_messages,
        round(mean_priority_score, 3),
        " | ".join(final_messages)
    ])

    if frame_id % 30 == 0:
        print(
            f"Frame:{frame_id} | "
            f"FPS:{fps:.2f} | "
            f"Latency:{latency_ms:.1f} ms | "
            f"Det:{detections_count} | "
            f"Obj:{len(objects)} | "
            f"UID:{len(unique_ids)} | "
            f"Raw:{raw_messages} | "
            f"Filtered:{filtered_messages} | "
            f"Decision:{decision_messages}"
        )

    frame_id += 1


cap.release()
csv_file.close()


def mean(values):
    return sum(values) / len(values) if values else 0


total_tracks = len(unique_ids)
avg_track_duration = (
    sum(track_frames.values()) / total_tracks
    if total_tracks > 0 else 0
)

stable_tracks = sum(
    1 for frames in track_frames.values()
    if frames >= 30
)

stable_tracks_ratio = (
    stable_tracks / total_tracks
    if total_tracks > 0 else 0
)


summary_file = open(summary_path, mode="w", newline="")
summary_writer = csv.writer(summary_file)

summary_writer.writerow([
    "tracker",
    "scenario",
    "run_id",
    "frames",
    "mean_fps",
    "mean_latency_ms",
    "mean_cpu_percent",
    "total_detections",
    "total_objects",
    "unique_ids",
    "avg_track_duration",
    "stable_tracks_ratio",
    "raw_messages",
    "filtered_messages",
    "decision_messages",
    "mean_priority_score",
    "output_csv"
])

summary_writer.writerow([
    TRACKER_TYPE,
    SCENARIO,
    RUN_ID,
    frame_id,
    round(mean(fps_values), 2),
    round(mean(latency_values), 2),
    round(mean(cpu_values), 2),
    total_detections,
    total_objects,
    total_tracks,
    round(avg_track_duration, 2),
    round(stable_tracks_ratio, 3),
    total_raw_messages,
    total_filtered_messages,
    total_decision_messages,
    round(mean(priority_scores_all), 3),
    str(csv_path)
])

summary_file.close()

print("===================================")
print("TRACKER COMPARISON FINISHED")
print(f"SUMMARY: {summary_path}")
print(f"Frames: {frame_id}")
print(f"Mean FPS: {mean(fps_values):.2f}")
print(f"Mean latency: {mean(latency_values):.2f} ms")
print(f"Unique IDs: {total_tracks}")
print(f"Avg track duration: {avg_track_duration:.2f}")
print(f"Stable tracks ratio: {stable_tracks_ratio:.3f}")
print(f"Decision messages: {total_decision_messages}")
print("===================================")
