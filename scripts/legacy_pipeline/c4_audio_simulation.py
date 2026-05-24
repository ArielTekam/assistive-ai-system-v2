from ultralytics import YOLO
import cv2
import time
import csv
import psutil
import subprocess
from pathlib import Path
from statistics import mean
import argparse
import threading
import queue


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
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0


class SimpleTracker:
    def __init__(self, iou_threshold=0.3, max_age=10):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks = {}
        self.next_id = 1

    def update(self, detections):
        updated = {}
        used = set()

        for tid, track in self.tracks.items():
            best_iou = 0
            best_idx = None

            for idx, det in enumerate(detections):
                if idx in used:
                    continue
                if det["class_name"] != track["class_name"]:
                    continue

                iou = box_iou(track["box"], det["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_idx is not None and best_iou >= self.iou_threshold:
                det = detections[best_idx]
                updated[tid] = {
                    "id": tid,
                    "class_name": det["class_name"],
                    "box": det["box"],
                    "confidence": det["confidence"],
                    "age": 0,
                    "hits": track["hits"] + 1,
                }
                used.add(best_idx)
            else:
                track["age"] += 1
                if track["age"] <= self.max_age:
                    updated[tid] = track

        for idx, det in enumerate(detections):
            if idx not in used:
                updated[self.next_id] = {
                    "id": self.next_id,
                    "class_name": det["class_name"],
                    "box": det["box"],
                    "confidence": det["confidence"],
                    "age": 0,
                    "hits": 1,
                }
                self.next_id += 1

        self.tracks = updated
        return list(self.tracks.values())

CLASS_WEIGHTS = {
    "person": 1.0,
    "car": 1.0,
    "bus": 1.0,
    "truck": 1.0,
    "motorcycle": 0.9,
    "bicycle": 0.8,
    "dog": 0.7,
    "chair": 0.5,
    "bench": 0.5,
    "cat": 0.4,
    "bottle": 0.2,
}

DECISION_THRESHOLD = 0.45

def compute_proximity(box, frame_size):
    x1, y1, x2, y2 = box

    area = (x2 - x1) * (y2 - y1)
    frame_area = frame_size * frame_size

    return min(area / frame_area, 1.0)


def compute_centrality(box, frame_size):
    x1, y1, x2, y2 = box

    center_x = (x1 + x2) / 2
    frame_center = frame_size / 2

    distance = abs(center_x - frame_center)

    return max(0, 1 - (distance / frame_center))


def compute_danger_score(track, frame_size):
    proximity = compute_proximity(track["box"], frame_size)
    centrality = compute_centrality(track["box"], frame_size)

    class_weight = CLASS_WEIGHTS.get(track["class_name"], 0.3)

    score = (
        0.5 * proximity +
        0.3 * centrality +
        0.2 * class_weight
    )

    return {
        "score": score,
        "proximity": proximity,
        "centrality": centrality,
        "class_weight": class_weight,
    }

class SimulatedAudioEngine:
    def __init__(self, speech_duration=1.5):
        self.speech_duration = speech_duration
        self.queue = queue.Queue(maxsize=1)
        self.spoken_count = 0
        self.dropped_count = 0
        self.running = True

        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def speak(self, message):
        if self.queue.full():
            try:
                self.queue.get_nowait()
                self.dropped_count += 1
            except queue.Empty:
                pass

        self.queue.put(message)

    def _run(self):
        while self.running:
            try:
                message = self.queue.get(timeout=0.1)
                print(f"[AUDIO_SIM] {message}")
                self.spoken_count += 1
                time.sleep(self.speech_duration)
            except queue.Empty:
                continue

    def stop(self):
        self.running = False
        self.worker.join(timeout=1)

class ContextMemory:
    def __init__(self, cooldown=5.0):
        self.cooldown = cooldown
        self.last_spoken = {}

    def position_label(self, box, width):
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2

        if center_x < width / 3:
            return "on the left"
        elif center_x > 2 * width / 3:
            return "on the right"
        return "ahead"

    def should_announce(self, track, frame_width, now):
        tid = track["id"]
        class_name = track["class_name"]
        position = self.position_label(track["box"], frame_width)
        
        danger = compute_danger_score(track, frame_width)
        score = danger["score"]
        
        if score < DECISION_THRESHOLD:
                return False, f"{class_name} {position}", "low_score"

        message = f"{class_name} {position}"

        if tid not in self.last_spoken:
            self.last_spoken[tid] = {
                "time": now,
                "message": message,
            }
            return True, message, "new_object"

        previous = self.last_spoken[tid]
        time_since = now - previous["time"]

        if message != previous["message"] and time_since >= 1.0:
            self.last_spoken[tid] = {
                "time": now,
                "message": message,
            }
            return True, message, "position_changed"

        if time_since >= self.cooldown:
            self.last_spoken[tid] = {
                "time": now,
                "message": message,
            }
            return True, message, "cooldown"

        return False, message, "filtered"


def benchmark_context(model_path, resolution, duration, output):
    model = YOLO(model_path)
    tracker = SimpleTracker(iou_threshold=0.3, max_age=10)
    context = ContextMemory(cooldown=5.0)
    audio = SimulatedAudioEngine(speech_duration=1.5)

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

    total_messages_raw = 0
    total_messages_context = 0

    print("\n=== Benchmark C2 Context Memory ===")
    print(f"Modèle      : {model_path}")
    print(f"Résolution  : {resolution}")
    print(f"Durée       : {duration} s")
    print(f"Sortie CSV  : {output}")
    print("===================================\n")

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
            "tracks",
            "raw_messages",
            "context_messages",
            "message",
            "reason"
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

                names = results[0].names
                detections = []

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
                now = time.time()

                raw_messages = len(tracks)
                context_messages = 0
                spoken_messages = []
                reasons = []

                for track in tracks:
                    speak, message, reason = context.should_announce(track, resolution, now)
                    if speak:
                            context_messages += 1
                            spoken_messages.append(message)
                            reasons.append(reason)
                            audio.speak(message)

                total_messages_raw += raw_messages
                total_messages_context += context_messages

                latency = (time.time() - start) * 1000
                fps = 1000 / latency if latency > 0 else 0
                cpu = psutil.cpu_percent(interval=None)
                ram = psutil.virtual_memory().percent
                temp = get_temp()

                fps_values.append(fps)
                latency_values.append(latency)
                cpu_values.append(cpu)
                ram_values.append(ram)
                if temp is not None:
                    temp_values.append(temp)

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
                    raw_messages,
                    context_messages,
                    " | ".join(spoken_messages),
                    " | ".join(reasons)
                ])

                if frame_id % 10 == 0:
                    print(
                        f"Frame {frame_id} | FPS {fps:.2f} | "
                        f"Tracks {len(tracks)} | Raw {raw_messages} | "
                        f"Context {context_messages} | "
                        f"Messages: {spoken_messages}"
                    )

                frame_id += 1

        except KeyboardInterrupt:
            print("\nArrêt manuel.")

    cap.release()
    audio.stop()

    reduction = 0
    if total_messages_raw > 0:
        reduction = (1 - total_messages_context / total_messages_raw) * 100

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
        "raw_messages_total": total_messages_raw,
        "context_messages_total": total_messages_context,
        "message_reduction_percent": round(reduction, 2),
    }

    summary_path = output_path.with_name(output_path.stem + "_summary.csv")

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(summary.keys())
        writer.writerow(summary.values())

    print("\n=== Résumé C2 ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print(f"\nCSV détaillé : {output_path}")
    print(f"CSV résumé   : {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--resolution", type=int, default=320)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--output", default="c2_context.csv")
    args = parser.parse_args()

    benchmark_context(args.model, args.resolution, args.duration, args.output)
