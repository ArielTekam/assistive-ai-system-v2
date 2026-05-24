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
from core.audio_manager import AudioManager


parser = argparse.ArgumentParser()

parser.add_argument("--video", required=True)
parser.add_argument("--experiment", required=True)

args = parser.parse_args()

VIDEO_PATH = args.video
EXPERIMENT_NAME = args.experiment


results_dir = Path("Experiments/results_csv")
results_dir.mkdir(parents=True, exist_ok=True)

csv_path = results_dir / f"{EXPERIMENT_NAME}.csv"
summary_path = results_dir / f"{EXPERIMENT_NAME}_summary.csv"


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

audio_manager = AudioManager(
    speech_duration=0.2,
    max_queue_size=2,
    min_repeat_interval=8.0,
    simulation=False,
    voice="en",
    speed=145
)


cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Ошибка открытия видео")
    audio_manager.stop()
    exit()


csv_file = open(csv_path, mode="w", newline="")
csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "frame",
    "fps",
    "latency_ms",
    "cpu_percent",
    "detections",
    "raw_messages",
    "context_messages",
    "decision_messages",
    "audio_spoken",
    "final_messages"
])


frame_id = 0

fps_values = []
latencies = []

total_raw = 0
total_context = 0
total_decisions = 0

start_global = time.time()

print("===================================")
print("VIDEO EXPERIMENT")
print(f"Video: {VIDEO_PATH}")
print(f"Experiment: {EXPERIMENT_NAME}")
print("===================================")


try:
    while True:

        frame_start = time.time()

        ret, frame = cap.read()

        if not ret:
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

        final_messages = []

        for decision in decisions:
            msg = decision["message"]
            final_messages.append(msg)
            audio_manager.speak(msg)

        latency_ms = (time.time() - pipeline_start) * 1000

        fps = 1.0 / max(time.time() - frame_start, 1e-6)

        cpu_percent = psutil.cpu_percent()

        raw_messages = context_result["raw_messages_count"]

        total_raw += raw_messages
        total_context += len(context_messages)
        total_decisions += len(decisions)

        fps_values.append(fps)
        latencies.append(latency_ms)

        csv_writer.writerow([
            frame_id,
            round(fps, 2),
            round(latency_ms, 2),
            cpu_percent,
            len(detections),
            raw_messages,
            len(context_messages),
            len(decisions),
            audio_manager.get_stats()["spoken_count"],
            " | ".join(final_messages)
        ])

        print(
            f"Frame:{frame_id} | "
            f"Det:{len(detections)} | "
            f"Decision:{len(decisions)} | "
            f"Msg:{final_messages}"
        )

        frame_id += 1

except KeyboardInterrupt:
    print("Остановка вручную")


cap.release()
csv_file.close()

time.sleep(2.0)

audio_stats = audio_manager.get_stats()

audio_manager.stop()

mean_fps = sum(fps_values) / len(fps_values)
mean_latency = sum(latencies) / len(latencies)

reduction = 0

if total_raw > 0:
    reduction = ((total_raw - total_decisions) / total_raw) * 100


summary_file = open(summary_path, mode="w", newline="")
summary_writer = csv.writer(summary_file)

summary_writer.writerow([
    "experiment",
    "frames",
    "mean_fps",
    "mean_latency_ms",
    "raw_messages",
    "decision_messages",
    "reduction_percent",
    "audio_spoken"
])

summary_writer.writerow([
    EXPERIMENT_NAME,
    frame_id,
    round(mean_fps, 2),
    round(mean_latency, 2),
    total_raw,
    total_decisions,
    round(reduction, 2),
    audio_stats["spoken_count"]
])

summary_file.close()

print("===================================")
print("EXPERIMENT FINISHED")
print(f"Frames: {frame_id}")
print(f"FPS: {mean_fps:.2f}")
print(f"Latency: {mean_latency:.2f} ms")
print(f"Reduction: {reduction:.2f}%")
print("===================================")
