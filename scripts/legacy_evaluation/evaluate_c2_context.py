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


# =========================================
# АРГУМЕНТЫ
# =========================================

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


# =========================================
# ПАПКА РЕЗУЛЬТАТОВ
# =========================================

results_dir = Path("results/c2")
results_dir.mkdir(parents=True, exist_ok=True)

csv_path = results_dir / f"{SCENARIO}_c2.csv"
summary_path = results_dir / f"{SCENARIO}_c2_summary.csv"


# =========================================
# ИНИЦИАЛИЗАЦИЯ МОДУЛЕЙ
# =========================================

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


# =========================================
# КАМЕРА
# =========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Ошибка: камера не открыта")
    exit()


# =========================================
# CSV
# =========================================

csv_file = open(csv_path, mode="w", newline="")
csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "frame",
    "fps",
    "latency_ms",
    "cpu_percent",
    "raw_messages",
    "filtered_messages",
    "blocked_messages",
    "messages"
])


# =========================================
# ПЕРЕМЕННЫЕ
# =========================================

frame_id = 0

fps_values = []
latencies = []

total_raw = 0
total_filtered = 0
total_blocked = 0

start_global = time.time()

print("===================================")
print(f"Сценарий: {SCENARIO}")
print("Тест C2 Context Manager запущен")
print("===================================")


# =========================================
# ОСНОВНОЙ ЦИКЛ
# =========================================

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

    latency_ms = (time.time() - pipeline_start) * 1000
    fps = 1.0 / max(time.time() - frame_start, 1e-6)
    cpu_percent = psutil.cpu_percent()

    raw_messages = context_result["raw_messages_count"]
    blocked_messages = context_result["filtered_count"]
    filtered_messages = len(context_result["messages"])

    message_texts = [
        msg["message"]
        for msg in context_result["messages"]
    ]

    total_raw += raw_messages
    total_filtered += filtered_messages
    total_blocked += blocked_messages

    fps_values.append(fps)
    latencies.append(latency_ms)

    csv_writer.writerow([
        frame_id,
        round(fps, 2),
        round(latency_ms, 2),
        cpu_percent,
        raw_messages,
        filtered_messages,
        blocked_messages,
        " | ".join(message_texts)
    ])

    print(
        f"Frame:{frame_id} | "
        f"Raw:{raw_messages} | "
        f"Filtered:{filtered_messages} | "
        f"Blocked:{blocked_messages}"
    )

    frame_id += 1


# =========================================
# ЗАВЕРШЕНИЕ
# =========================================

cap.release()
csv_file.close()

mean_fps = sum(fps_values) / len(fps_values)
mean_latency = sum(latencies) / len(latencies)

reduction_percent = 0

if total_raw > 0:
    reduction_percent = (
        total_blocked / total_raw
    ) * 100


summary_file = open(summary_path, mode="w", newline="")
summary_writer = csv.writer(summary_file)

summary_writer.writerow([
    "scenario",
    "frames",
    "mean_fps",
    "mean_latency_ms",
    "raw_messages",
    "filtered_messages",
    "blocked_messages",
    "reduction_percent"
])

summary_writer.writerow([
    SCENARIO,
    frame_id,
    round(mean_fps, 2),
    round(mean_latency, 2),
    total_raw,
    total_filtered,
    total_blocked,
    round(reduction_percent, 2)
])

summary_file.close()

print("===================================")
print("Тест завершён")
print(f"Frames: {frame_id}")
print(f"Средний FPS: {mean_fps:.2f}")
print(f"Средняя латентность: {mean_latency:.2f} ms")
print(f"Raw messages: {total_raw}")
print(f"Filtered messages: {total_filtered}")
print(f"Blocked messages: {total_blocked}")
print(f"Reduction: {reduction_percent:.2f}%")
print("===================================")

