from ultralytics import YOLO
import cv2
import time
import csv
import psutil
import subprocess
from pathlib import Path
from statistics import mean
import argparse


def get_temp():
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"]).decode()
        return float(out.replace("temp=", "").replace("'C\n", ""))
    except Exception:
        return None


def box_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    union = areaA + areaB - inter_area
    return inter_area / union if union > 0 else 0


class SimpleTracker:
    def __init__(self, iou_threshold=0.3, max_age=10):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks = {}
        self.next_id = 1

    def update(self, detections):
        updated_tracks = {}
        used_detections = set()

        for track_id, track in self.tracks.items():
            best_iou = 0
            best_idx = None

            for idx, det in enumerate(detections):
                if idx in used_detections:
                    continue
                if det["class_name"] != track["class_name"]:
                    continue

                iou = box_iou(track["box"], det["box"])

                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_idx is not None and best_iou >= self.iou_threshold:
                det = detections[best_idx]
                updated_tracks[track_id] = {
                    "id": track_id,
                    "class_name": det["class_name"],
                    "box": det["box"],
                    "confidence": det["confidence"],
                    "age": 0,
                    "hits": track["hits"] + 1,
                    "iou": best_iou,
                }
                used_detections.add(best_idx)
            else:
                track["age"] += 1
                if track["age"] <= self.max_age:
                    updated_tracks[track_id] = track

        for idx, det in enumerate(detections):
            if idx not in used_detections:
                updated_tracks[self.next_id] = {
                    "id": self.next_id,
                    "class_name": det["class_name"],
                    "box": det["box"],
                    "confidence": det["confidence"],
                    "age": 0,
                    "hits": 1,
                    "iou": 0,
                }
                self.next_id += 1

        self.tracks = updated_tracks
        return list(self.tracks.values())


def benchmark_tracking(model_path, resolution, duration, output):
    model = YOLO(model_path)
    tracker = SimpleTracker(iou_threshold=0.3, max_age=10)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur : caméra non ouverte")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution)

    output_path = Path(output)

    fps_values = []
    latency_values = []
    cpu_values = []
    ram_values = []
    temp_values = []
    detections_values = []
    tracks_values = []

    print("\n=== Benchmark C1 Tracking ===")
    print(f"Modèle      : {model_path}")
    print(f"Résolution  : {resolution}")
    print(f"Durée       : {duration} s")
    print(f"Sortie CSV  : {output}")
    print("=============================\n")

    start_global = time.time()
    frame_id = 0

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame",
            "time_s",
            "latency_ms",
            "fps",
            "cpu_percent",
            "ram_percent",
            "temperature_c",
            "detections",
            "active_tracks",
            "track_ids"
        ])

        try:
            while True:
                elapsed = time.time() - start_global
                if elapsed >= duration:
                    break

                ret, frame = cap.read()
                if not ret:
                    print("Erreur : frame non lue")
                    break

                frame = cv2.resize(frame, (resolution, resolution))

                start = time.time()
                results = model(frame, imgsz=resolution, verbose=False)

                detections = []
                names = results[0].names

                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    class_name = names[cls_id]
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    detections.append({
                        "class_name": class_name,
                        "confidence": confidence,
                        "box": [x1, y1, x2, y2]
                    })

                tracks = tracker.update(detections)

                latency = (time.time() - start) * 1000
                fps = 1000 / latency if latency > 0 else 0

                cpu = psutil.cpu_percent(interval=None)
                ram = psutil.virtual_memory().percent
                temp = get_temp()

                track_ids = [t["id"] for t in tracks]

                fps_values.append(fps)
                latency_values.append(latency)
                cpu_values.append(cpu)
                ram_values.append(ram)
                if temp is not None:
                    temp_values.append(temp)
                detections_values.append(len(detections))
                tracks_values.append(len(tracks))

                writer.writerow([
                    frame_id,
                    round(elapsed, 3),
                    round(latency, 2),
                    round(fps, 2),
                    round(cpu, 2),
                    round(ram, 2),
                    round(temp, 2) if temp is not None else "",
                    len(detections),
                    len(tracks),
                    str(track_ids)
                ])

                if frame_id % 10 == 0:
                    print(
                        f"Frame {frame_id} | "
                        f"FPS {fps:.2f} | "
                        f"Latence {latency:.1f} ms | "
                        f"Detections {len(detections)} | "
                        f"Tracks {len(tracks)} | "
                        f"IDs {track_ids}"
                    )

                frame_id += 1

        except KeyboardInterrupt:
            print("\nArrêt manuel.")

    cap.release()

    summary_path = output_path.with_name(output_path.stem + "_summary.csv")

    summary = {
        "model": model_path,
        "resolution": resolution,
        "duration_s": duration,
        "frames": frame_id,
        "fps_mean": round(mean(fps_values), 2) if fps_values else 0,
        "latency_mean_ms": round(mean(latency_values), 2) if latency_values else 0,
        "cpu_mean_percent": round(mean(cpu_values), 2) if cpu_values else 0,
        "ram_mean_percent": round(mean(ram_values), 2) if ram_values else 0,
        "temperature_mean_c": round(mean(temp_values), 2) if temp_values else "",
        "detections_mean": round(mean(detections_values), 2) if detections_values else 0,
        "active_tracks_mean": round(mean(tracks_values), 2) if tracks_values else 0,
    }

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(summary.keys())
        writer.writerow(summary.values())

    print("\n=== Résumé C1 ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print(f"\nCSV détaillé : {output_path}")
    print(f"CSV résumé   : {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--resolution", type=int, default=320)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--output", default="c1_tracking.csv")
    args = parser.parse_args()

    benchmark_tracking(args.model, args.resolution, args.duration, args.output)
