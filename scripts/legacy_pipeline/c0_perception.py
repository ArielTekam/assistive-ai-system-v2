import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import csv
import time
import argparse
import psutil

from core.detector import ObjectDetector


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

parser.add_argument(
    "--show",
    action="store_true",
    help="Показать изображение с камеры во время теста"
)

args = parser.parse_args()

SCENARIO = args.scenario
DURATION = args.duration


results_dir = Path("results/c0")
results_dir.mkdir(parents=True, exist_ok=True)

csv_path = results_dir / f"{SCENARIO}_c0.csv"
summary_path = results_dir / f"{SCENARIO}_c0_summary.csv"


detector = ObjectDetector(
    model_path="yolo11n.pt",
    conf_threshold=0.25,
    img_size=320
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
    "labels"
])


frame_id = 0
fps_values = []
latencies = []
detections_total = 0

start_global = time.time()

print("===================================")
print(f"Сценарий: {SCENARIO}")
print(f"Длительность: {DURATION} секунд")
print("Тест C0 Perception запущен")
print("===================================")


while time.time() - start_global < DURATION:

    frame_start = time.time()

    ret, frame = cap.read()

    if not ret:
        print("Ошибка чтения кадра")
        break

    infer_start = time.time()
    detections = detector.detect(frame)
    infer_end = time.time()

    latency_ms = (infer_end - infer_start) * 1000

    total_frame_time = time.time() - frame_start
    fps = 1.0 / max(total_frame_time, 1e-6)

    cpu_percent = psutil.cpu_percent()

    labels = [d["label"] for d in detections]
    detections_count = len(detections)

    fps_values.append(fps)
    latencies.append(latency_ms)
    detections_total += detections_count

    csv_writer.writerow([
        frame_id,
        round(time.time(), 2),
        round(fps, 2),
        round(latency_ms, 2),
        cpu_percent,
        detections_count,
        ",".join(labels)
    ])

    print(
        f"Frame:{frame_id} | "
        f"FPS:{fps:.2f} | "
        f"Latency:{latency_ms:.1f} ms | "
        f"CPU:{cpu_percent}% | "
        f"Detections:{detections_count}"
    )

    if args.show:
        display_frame = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            label = det["label"]
            conf = det["confidence"]

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                display_frame,
                f"{label} {conf:.2f}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )

        cv2.imshow("C0 Perception Test", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    frame_id += 1


cap.release()
cv2.destroyAllWindows()
csv_file.close()


mean_fps = sum(fps_values) / len(fps_values) if fps_values else 0
mean_latency = sum(latencies) / len(latencies) if latencies else 0

summary_file = open(summary_path, mode="w", newline="")
summary_writer = csv.writer(summary_file)

summary_writer.writerow([
    "scenario",
    "frames",
    "mean_fps",
    "mean_latency_ms",
    "total_detections"
])

summary_writer.writerow([
    SCENARIO,
    frame_id,
    round(mean_fps, 2),
    round(mean_latency, 2),
    detections_total
])

summary_file.close()

print("===================================")
print("Тест завершён")
print(f"Frames: {frame_id}")
print(f"Средний FPS: {mean_fps:.2f}")
print(f"Средняя латентность: {mean_latency:.2f} ms")
print(f"Всего детекций: {detections_total}")
print("===================================")