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


parser = argparse.ArgumentParser()

parser.add_argument(
    "--scenario",
    type=str,
    required=True,
    help="Название сценария"
)

parser.add_argument(
    "--duration",
    type=int,
    default=60,
    help="Длительность теста"
)

args = parser.parse_args()

SCENARIO = args.scenario
DURATION = args.duration


results_dir = Path("results/c3")
results_dir.mkdir(parents=True, exist_ok=True)

csv_path = results_dir / f"{SCENARIO}_c3.csv"
summary_path = results_dir / f"{SCENARIO}_c3_summary.csv"


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
    cooldown_seconds=5.0,
    min_proximity_change=0.08
)

decision_engine = DecisionEngine(
    priority_threshold=0.45,
    high_priority_threshold=0.70
)


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Ошибка: камера не открыта")
    exit()


csv_file = open(csv_path, mode="w", newline="")
csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "frame",
    "fps",
    "latency_ms",
    "cpu_percent",
    "raw_messages",
    "context_messages",
    "decision_messages",
    "priority_scores",
    "priority_levels",
    "final_messages"
])


frame_id = 0

fps_values = []
latencies = []

total_raw = 0
total_context = 0
total_decisions = 0
total_low_filtered = 0

priority_scores_all = []

start_global = time.time()

print("===================================")
print(f"Сценарий: {SCENARIO}")
print("Тест C3 Decision Engine запущен")
print("===================================")


while time.time() - start_global < DURATION:

    frame_start = time.time()

    ret, frame = cap.read()

    if not ret:
        print("Ошибка чтения кадра")
        break

    pipeline_start = time.time()

    detections = detector.detect(frame)
    objects = memory.update(detections)

    context_result = context_manager.filter_messages(objects)

    context_messages = context_result["messages"]

    decisions = decision_engine.decide(
        context_messages=context_messages,
        objects=objects
    )

    latency_ms = (time.time() - pipeline_start) * 1000
    fps = 1.0 / max(time.time() - frame_start, 1e-6)
    cpu_percent = psutil.cpu_percent()

    raw_messages = context_result["raw_messages_count"]
    context_messages_count = len(context_messages)
    decision_messages_count = len(decisions)

    total_raw += raw_messages
    total_context += context_messages_count
    total_decisions += decision_messages_count

    low_filtered = max(0, context_messages_count - decision_messages_count)
    total_low_filtered += low_filtered

    priority_scores = [
        str(item["priority_score"])
        for item in decisions
    ]

    priority_levels = [
        item["priority_level"]
        for item in decisions
    ]

    final_messages = [
        item["message"]
        for item in decisions
    ]

    for item in decisions:
        priority_scores_all.append(item["priority_score"])

    fps_values.append(fps)
    latencies.append(latency_ms)

    csv_writer.writerow([
        frame_id,
        round(fps, 2),
        round(latency_ms, 2),
        cpu_percent,
        raw_messages,
        context_messages_count,
        decision_messages_count,
        ",".join(priority_scores),
        ",".join(priority_levels),
        " | ".join(final_messages)
    ])

    print(
        f"Frame:{frame_id} | "
        f"Raw:{raw_messages} | "
        f"Context:{context_messages_count} | "
        f"Decision:{decision_messages_count} | "
        f"Messages:{final_messages}"
    )

    frame_id += 1


cap.release()
csv_file.close()


mean_fps = sum(fps_values) / len(fps_values) if fps_values else 0
mean_latency = sum(latencies) / len(latencies) if latencies else 0

context_reduction = 0
decision_reduction = 0
total_reduction = 0

if total_raw > 0:
    context_reduction = ((total_raw - total_context) / total_raw) * 100
    total_reduction = ((total_raw - total_decisions) / total_raw) * 100

if total_context > 0:
    decision_reduction = ((total_context - total_decisions) / total_context) * 100

mean_priority_score = (
    sum(priority_scores_all) / len(priority_scores_all)
    if priority_scores_all else 0
)


summary_file = open(summary_path, mode="w", newline="")
summary_writer = csv.writer(summary_file)

summary_writer.writerow([
    "scenario",
    "frames",
    "mean_fps",
    "mean_latency_ms",
    "raw_messages",
    "context_messages",
    "decision_messages",
    "context_reduction_percent",
    "decision_reduction_percent",
    "total_reduction_percent",
    "mean_priority_score"
])

summary_writer.writerow([
    SCENARIO,
    frame_id,
    round(mean_fps, 2),
    round(mean_latency, 2),
    total_raw,
    total_context,
    total_decisions,
    round(context_reduction, 2),
    round(decision_reduction, 2),
    round(total_reduction, 2),
    round(mean_priority_score, 3)
])

summary_file.close()

print("===================================")
print("Тест завершён")
print(f"Frames: {frame_id}")
print(f"Средний FPS: {mean_fps:.2f}")
print(f"Средняя латентность: {mean_latency:.2f} ms")
print(f"Raw messages: {total_raw}")
print(f"Context messages: {total_context}")
print(f"Decision messages: {total_decisions}")
print(f"Context reduction: {context_reduction:.2f}%")
print(f"Decision reduction: {decision_reduction:.2f}%")
print(f"Total reduction: {total_reduction:.2f}%")
print(f"Mean priority score: {mean_priority_score:.3f}")
print("===================================")
