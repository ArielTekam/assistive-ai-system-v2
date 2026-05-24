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
    help="Длительность теста в секундах"
)

args = parser.parse_args()

SCENARIO = args.scenario
DURATION = args.duration


results_dir = Path("results/c1")
results_dir.mkdir(parents=True, exist_ok=True)

csv_path = results_dir / f"{SCENARIO}_c1.csv"
summary_path = results_dir / f"{SCENARIO}_c1_summary.csv"


detector = ObjectDetector(
    model_path="yolo11n.pt",
    conf_threshold=0.25,
    img_size=320
)

memory = SceneMemory(
    frame_width=640,
    frame_height=480,
    max_distance=140,
    max_missing_frames=10,
    smoothing_alpha=0.3
)


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Ошибка: камера не открыта")
    exit()


csv_file = open(csv_path, mode="w", newline="")
csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "frame",
    "timestamp",
    "fps",
    "latency_ms",
    "cpu_percent",
    "detections_count",
    "active_tracks",
    "track_ids",
    "labels",
    "directions",
    "proximities",
    "risk_scores",
    "missing_tracks"
])


frame_id = 0
fps_values = []
latencies = []
active_tracks_values = []
detections_total = 0
unique_ids = set()
id_history = []

start_global = time.time()

print("===================================")
print(f"Сценарий: {SCENARIO}")
print(f"Длительность: {DURATION} секунд")
print("Тест C1 Scene Memory / Tracking запущен")
print("===================================")


while time.time() - start_global < DURATION:

    frame_start = time.time()

    ret, frame = cap.read()

    if not ret:
        print("Ошибка чтения кадра")
        break

    step_start = time.time()

    detections = detector.detect(frame)
    objects = memory.update(detections)

    latency_ms = (time.time() - step_start) * 1000
    fps = 1.0 / max(time.time() - frame_start, 1e-6)
    cpu_percent = psutil.cpu_percent()

    visible_objects = [
        obj for obj in objects.values()
        if obj.get("missing_frames", 0) == 0
    ]

    missing_objects = [
        obj for obj in objects.values()
        if obj.get("missing_frames", 0) > 0
    ]

    track_ids = [obj["id"] for obj in visible_objects]
    labels = [obj["label"] for obj in visible_objects]
    directions = [obj["direction"] for obj in visible_objects]
    proximities = [round(obj["proximity"], 2) for obj in visible_objects]
    risk_scores = [round(obj["risk_score"], 2) for obj in visible_objects]

    for track_id in track_ids:
        unique_ids.add(track_id)

    id_history.append(track_ids)

    detections_total += len(detections)
    active_tracks_values.append(len(visible_objects))
    fps_values.append(fps)
    latencies.append(latency_ms)

    csv_writer.writerow([
        frame_id,
        round(time.time(), 2),
        round(fps, 2),
        round(latency_ms, 2),
        cpu_percent,
        len(detections),
        len(visible_objects),
        ",".join(map(str, track_ids)),
        ",".join(labels),
        ",".join(directions),
        ",".join(map(str, proximities)),
        ",".join(map(str, risk_scores)),
        len(missing_objects)
    ])

    print(
        f"Frame:{frame_id} | "
        f"FPS:{fps:.2f} | "
        f"Latency:{latency_ms:.1f} ms | "
        f"Detections:{len(detections)} | "
        f"Tracks:{len(visible_objects)} | "
        f"IDs:{track_ids}"
    )

    frame_id += 1


cap.release()
csv_file.close()


mean_fps = sum(fps_values) / len(fps_values) if fps_values else 0
mean_latency = sum(latencies) / len(latencies) if latencies else 0
mean_tracks = sum(active_tracks_values) / len(active_tracks_values) if active_tracks_values else 0

summary_file = open(summary_path, mode="w", newline="")
summary_writer = csv.writer(summary_file)

summary_writer.writerow([
    "scenario",
    "frames",
    "mean_fps",
    "mean_latency_ms",
    "total_detections",
    "mean_active_tracks",
    "unique_track_ids"
])

summary_writer.writerow([
    SCENARIO,
    frame_id,
    round(mean_fps, 2),
    round(mean_latency, 2),
    detections_total,
    round(mean_tracks, 2),
    len(unique_ids)
])

summary_file.close()

print("===================================")
print("Тест завершён")
print(f"Frames: {frame_id}")
print(f"Средний FPS: {mean_fps:.2f}")
print(f"Средняя латентность: {mean_latency:.2f} ms")
print(f"Всего детекций: {detections_total}")
print(f"Среднее число активных треков: {mean_tracks:.2f}")
print(f"Уникальных ID: {len(unique_ids)}")
print("===================================")
