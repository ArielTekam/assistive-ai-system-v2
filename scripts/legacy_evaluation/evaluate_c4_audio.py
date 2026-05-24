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


results_dir = Path("results/c4")
results_dir.mkdir(parents=True, exist_ok=True)

csv_path = results_dir / f"{SCENARIO}_c4.csv"
summary_path = results_dir / f"{SCENARIO}_c4_summary.csv"


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

audio_manager = AudioManager(
    speech_duration=0.2,
    max_queue_size=3,
    min_repeat_interval=2.0,
    simulation=False,
    voice="en",
    speed=145
)


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Ошибка: камера не открыта")
    audio_manager.stop()
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
    "audio_accepted",
    "audio_spoken_total",
    "audio_dropped_total",
    "audio_repeated_blocked_total",
    "final_messages"
])


frame_id = 0

fps_values = []
latencies = []

total_raw = 0
total_context = 0
total_decisions = 0
total_audio_accepted = 0

start_global = time.time()

print("===================================")
print(f"Сценарий: {SCENARIO}")
print("Тест C4 Audio Engine запущен")
print("===================================")


try:
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

        audio_accepted = 0
        final_messages = []

        for decision in decisions:
            message = decision["message"]
            final_messages.append(message)

            accepted = audio_manager.speak(message)

            if accepted:
                audio_accepted += 1

        latency_ms = (time.time() - pipeline_start) * 1000
        fps = 1.0 / max(time.time() - frame_start, 1e-6)
        cpu_percent = psutil.cpu_percent()

        raw_messages = context_result["raw_messages_count"]
        context_messages_count = len(context_messages)
        decision_messages_count = len(decisions)

        total_raw += raw_messages
        total_context += context_messages_count
        total_decisions += decision_messages_count
        total_audio_accepted += audio_accepted

        fps_values.append(fps)
        latencies.append(latency_ms)

        audio_stats = audio_manager.get_stats()

        csv_writer.writerow([
            frame_id,
            round(fps, 2),
            round(latency_ms, 2),
            cpu_percent,
            raw_messages,
            context_messages_count,
            decision_messages_count,
            audio_accepted,
            audio_stats["spoken_count"],
            audio_stats["dropped_count"],
            audio_stats["repeated_blocked_count"],
            " | ".join(final_messages)
        ])

        print(
            f"Frame:{frame_id} | "
            f"Raw:{raw_messages} | "
            f"Context:{context_messages_count} | "
            f"Decision:{decision_messages_count} | "
            f"Audio accepted:{audio_accepted} | "
            f"Messages:{final_messages}"
        )

        frame_id += 1

except KeyboardInterrupt:
    print("\nТест остановлен вручную")


cap.release()
csv_file.close()

# On laisse au thread audio le temps de finir le message en cours
time.sleep(2.0)

audio_stats = audio_manager.get_stats()
audio_manager.stop()


mean_fps = sum(fps_values) / len(fps_values) if fps_values else 0
mean_latency = sum(latencies) / len(latencies) if latencies else 0

total_reduction = 0

if total_raw > 0:
    total_reduction = ((total_raw - total_decisions) / total_raw) * 100


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
    "audio_accepted",
    "audio_spoken",
    "audio_dropped",
    "audio_repeated_blocked",
    "total_reduction_percent"
])

summary_writer.writerow([
    SCENARIO,
    frame_id,
    round(mean_fps, 2),
    round(mean_latency, 2),
    total_raw,
    total_context,
    total_decisions,
    total_audio_accepted,
    audio_stats["spoken_count"],
    audio_stats["dropped_count"],
    audio_stats["repeated_blocked_count"],
    round(total_reduction, 2)
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
print(f"Audio accepted: {total_audio_accepted}")
print(f"Audio spoken: {audio_stats['spoken_count']}")
print(f"Audio dropped: {audio_stats['dropped_count']}")
print(f"Audio repeated blocked: {audio_stats['repeated_blocked_count']}")
print(f"Total reduction: {total_reduction:.2f}%")
print("===================================")
