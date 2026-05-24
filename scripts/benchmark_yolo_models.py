import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import csv
import time
import argparse
import psutil
from ultralytics import YOLO


parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="Ex: yolov5nu.pt, yolov8n.pt, yolo11n.pt")
parser.add_argument("--scenario", required=True)
parser.add_argument("--run", type=int, required=True)
parser.add_argument("--video", required=True)
parser.add_argument("--imgsz", type=int, default=320)

args = parser.parse_args()

MODEL_PATH = args.model
SCENARIO = args.scenario
RUN_ID = args.run
VIDEO_PATH = args.video
IMG_SIZE = args.imgsz

results_dir = Path("Experiments/results_csv")
results_dir.mkdir(parents=True, exist_ok=True)

model_name = Path(MODEL_PATH).stem
output_name = f"YOLO_{model_name}_{SCENARIO}_img{IMG_SIZE}_run{RUN_ID:02d}"

csv_path = results_dir / f"{output_name}.csv"
summary_path = results_dir / f"{output_name}_summary.csv"

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Erreur: impossible d'ouvrir la vidéo: {VIDEO_PATH}")
    exit()

csv_file = open(csv_path, mode="w", newline="")
writer = csv.writer(csv_file)

writer.writerow([
    "timestamp",
    "model",
    "scenario",
    "run_id",
    "frame",
    "imgsz",
    "fps_instant",
    "latency_ms",
    "cpu_percent",
    "detections_count"
])

frame_id = 0
fps_values = []
latency_values = []
cpu_values = []
detections_values = []

print("===================================")
print("YOLO MODEL BENCHMARK STARTED")
print(f"Model: {MODEL_PATH}")
print(f"Scenario: {SCENARIO}")
print(f"Run: {RUN_ID}")
print(f"Image size: {IMG_SIZE}")
print(f"Video: {VIDEO_PATH}")
print("===================================")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_start = time.time()
    inference_start = time.time()

    results = model.predict(
        frame,
        imgsz=IMG_SIZE,
        conf=0.25,
        verbose=False
    )[0]

    latency_ms = (time.time() - inference_start) * 1000
    fps = 1.0 / max(time.time() - frame_start, 1e-6)
    cpu_percent = psutil.cpu_percent()

    detections_count = 0

    if results.boxes is not None:
        detections_count = len(results.boxes)

    fps_values.append(fps)
    latency_values.append(latency_ms)
    cpu_values.append(cpu_percent)
    detections_values.append(detections_count)

    writer.writerow([
        round(time.time(), 3),
        model_name,
        SCENARIO,
        RUN_ID,
        frame_id,
        IMG_SIZE,
        round(fps, 2),
        round(latency_ms, 2),
        round(cpu_percent, 2),
        detections_count
    ])

    if frame_id % 30 == 0:
        print(
            f"Frame:{frame_id} | "
            f"FPS:{fps:.2f} | "
            f"Latency:{latency_ms:.1f} ms | "
            f"CPU:{cpu_percent:.1f}% | "
            f"Det:{detections_count}"
        )

    frame_id += 1

cap.release()
csv_file.close()


def mean(values):
    return sum(values) / len(values) if values else 0


def min_value(values):
    return min(values) if values else 0


def max_value(values):
    return max(values) if values else 0


summary_file = open(summary_path, mode="w", newline="")
summary_writer = csv.writer(summary_file)

summary_writer.writerow([
    "model",
    "scenario",
    "run_id",
    "imgsz",
    "frames",
    "mean_fps",
    "min_fps",
    "max_fps",
    "mean_latency_ms",
    "min_latency_ms",
    "max_latency_ms",
    "mean_cpu_percent",
    "mean_detections_per_frame",
    "total_detections",
    "output_csv"
])

summary_writer.writerow([
    model_name,
    SCENARIO,
    RUN_ID,
    IMG_SIZE,
    frame_id,
    round(mean(fps_values), 2),
    round(min_value(fps_values), 2),
    round(max_value(fps_values), 2),
    round(mean(latency_values), 2),
    round(min_value(latency_values), 2),
    round(max_value(latency_values), 2),
    round(mean(cpu_values), 2),
    round(mean(detections_values), 2),
    sum(detections_values),
    str(csv_path)
])

summary_file.close()

print("===================================")
print("YOLO MODEL BENCHMARK FINISHED")
print(f"SUMMARY: {summary_path}")
print(f"Frames: {frame_id}")
print(f"Mean FPS: {mean(fps_values):.2f}")
print(f"Mean latency: {mean(latency_values):.2f} ms")
print(f"Mean CPU: {mean(cpu_values):.2f}%")
print(f"Total detections: {sum(detections_values)}")
print("===================================")
