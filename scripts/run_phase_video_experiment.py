import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import csv
import time
import argparse
import psutil

from core.detector import ObjectDetector
from core.scene_memory import SceneMemory
from core.context_manager import ContextManager
from core.decision_engine import DecisionEngine
from core.safe_decision_filter import SafeDecisionFilter


parser = argparse.ArgumentParser()

parser.add_argument("--phase", required=True, choices=["C0", "C1", "C2", "C3"])
parser.add_argument("--scenario", required=True)
parser.add_argument("--run", type=int, required=True)
parser.add_argument("--video", required=True)

args = parser.parse_args()

PHASE = args.phase
SCENARIO = args.scenario
RUN_ID = args.run
VIDEO_PATH = args.video


results_dir = Path("Experiments/results_csv")
results_dir.mkdir(parents=True, exist_ok=True)

output_name = f"{PHASE}_{SCENARIO}_run{RUN_ID:02d}"
csv_path = results_dir / f"{output_name}.csv"
summary_path = results_dir / f"{output_name}_summary.csv"


detector = ObjectDetector(
    model_path="yolo11n.pt",
    conf_threshold=0.25,
    img_size=320
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

safe_filter = SafeDecisionFilter(
    global_cooldown=7.0,
    same_message_cooldown=15.0,
    same_object_family_cooldown=10.0,
    max_messages_per_cycle=1
)


cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Erreur: impossible d'ouvrir la vidéo: {VIDEO_PATH}")
    exit()


csv_file = open(csv_path, mode="w", newline="")
writer = csv.writer(csv_file)

writer.writerow([
    "timestamp",
    "phase",
    "scenario",
    "run_id",
    "frame",
    "fps_instant",
    "latency_ms",
    "cpu_percent",
    "detections_count",
    "raw_messages",
    "filtered_messages",
    "decision_messages",
    "mean_priority_score",
    "final_messages",
    "raw_decision_messages"
])


frame_id = 0

fps_values = []
latency_values = []
cpu_values = []

total_detections = 0
total_raw_messages = 0
total_filtered_messages = 0
total_decision_messages = 0
total_raw_decision_messages = 0

priority_scores_all = []

print("===================================")
print("EXPERIMENT STARTED")
print(f"Phase: {PHASE}")
print(f"Scenario: {SCENARIO}")
print(f"Run: {RUN_ID}")
print(f"Video: {VIDEO_PATH}")
print("SAFE FILTER ENABLED IN METRICS")
print("===================================")


while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_start = time.time()
    pipeline_start = time.time()

    detections = detector.detect(frame)

    raw_messages = 0
    filtered_messages = 0
    decision_messages = 0
    raw_decision_messages = 0
    final_messages = []
    priority_scores = []

    if PHASE == "C0":
        pass

    elif PHASE == "C1":
        objects = memory.update(detections)

    elif PHASE == "C2":
        objects = memory.update(detections)
        context_result = context_manager.filter_messages(objects)

        raw_messages = context_result["raw_messages_count"]
        filtered_messages = len(context_result["messages"])

    elif PHASE == "C3":
        objects = memory.update(detections)
        context_result = context_manager.filter_messages(objects)

        context_messages = context_result["messages"]

        decisions = decision_engine.decide(
            context_messages=context_messages,
            objects=objects
        )

        raw_messages = context_result["raw_messages_count"]
        filtered_messages = len(context_messages)

        raw_final_messages = [d["message"] for d in decisions]
        raw_priority_scores = [d["priority_score"] for d in decisions]

        raw_decision_messages = len(raw_final_messages)

        safe_messages = safe_filter.filter(raw_final_messages)

        final_messages = safe_messages
        decision_messages = len(final_messages)

        for msg in final_messages:
            for d in decisions:
                if d["message"] == msg:
                    priority_scores.append(d["priority_score"])
                    break

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

    total_detections += len(detections)
    total_raw_messages += raw_messages
    total_filtered_messages += filtered_messages
    total_decision_messages += decision_messages
    total_raw_decision_messages += raw_decision_messages

    writer.writerow([
        round(time.time(), 3),
        PHASE,
        SCENARIO,
        RUN_ID,
        frame_id,
        round(fps, 2),
        round(latency_ms, 2),
        round(cpu_percent, 2),
        len(detections),
        raw_messages,
        filtered_messages,
        decision_messages,
        round(mean_priority_score, 3),
        " | ".join(final_messages),
        raw_decision_messages
    ])

    if frame_id % 30 == 0:
        print(
            f"Frame:{frame_id} | "
            f"FPS:{fps:.2f} | "
            f"Latency:{latency_ms:.1f} ms | "
            f"Det:{len(detections)} | "
            f"Raw:{raw_messages} | "
            f"Filtered:{filtered_messages} | "
            f"Decision:{decision_messages} | "
            f"RawDecision:{raw_decision_messages} | "
            f"Msg:{final_messages}"
        )

    frame_id += 1


cap.release()
csv_file.close()


def mean(values):
    return sum(values) / len(values) if values else 0


summary_file = open(summary_path, mode="w", newline="")
summary_writer = csv.writer(summary_file)

summary_writer.writerow([
    "phase",
    "scenario",
    "run_id",
    "frames",
    "mean_fps",
    "mean_latency_ms",
    "mean_cpu_percent",
    "total_detections",
    "raw_messages",
    "filtered_messages",
    "decision_messages",
    "raw_decision_messages",
    "mean_priority_score",
    "output_csv"
])

summary_writer.writerow([
    PHASE,
    SCENARIO,
    RUN_ID,
    frame_id,
    round(mean(fps_values), 2),
    round(mean(latency_values), 2),
    round(mean(cpu_values), 2),
    total_detections,
    total_raw_messages,
    total_filtered_messages,
    total_decision_messages,
    total_raw_decision_messages,
    round(mean(priority_scores_all), 3),
    str(csv_path)
])

summary_file.close()

print("===================================")
print("EXPERIMENT FINISHED")
print(f"CSV: {csv_path}")
print(f"SUMMARY: {summary_path}")
print(f"Frames: {frame_id}")
print(f"Mean FPS: {mean(fps_values):.2f}")
print(f"Mean latency: {mean(latency_values):.2f} ms")
print(f"SAFE decisions: {total_decision_messages}")
print(f"Raw C3 decisions: {total_raw_decision_messages}")
print("===================================")