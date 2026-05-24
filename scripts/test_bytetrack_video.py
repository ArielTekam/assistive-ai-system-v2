import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import argparse
import time

from core.bytetrack_detector import ByteTrackDetector


parser = argparse.ArgumentParser()

parser.add_argument("--video", required=True)
parser.add_argument("--duration", type=int, default=0)

args = parser.parse_args()

tracker = ByteTrackDetector(
    model_path="yolo11n.pt",
    conf_threshold=0.25,
    img_size=320,
    tracker_config="bytetrack.yaml"
)

cap = cv2.VideoCapture(args.video)

if not cap.isOpened():
    print("Ошибка: невозможно открыть видео")
    exit()

frame_id = 0
start_time = time.time()

unique_ids = set()

while True:
    if args.duration > 0 and time.time() - start_time > args.duration:
        break

    ret, frame = cap.read()

    if not ret:
        break

    objects = tracker.track(frame)

    ids = []
    labels = []

    for obj in objects:
        if obj["track_id"] is not None:
            unique_ids.add(obj["track_id"])
            ids.append(str(obj["track_id"]))
        else:
            ids.append("None")

        labels.append(obj["label"])

    print(
        f"Frame:{frame_id} | "
        f"Objects:{len(objects)} | "
        f"IDs:{ids} | "
        f"Labels:{labels} | "
        f"Unique IDs:{len(unique_ids)}"
    )

    frame_id += 1

cap.release()

print("===================================")
print("ByteTrack test finished")
print(f"Frames: {frame_id}")
print(f"Unique IDs: {len(unique_ids)}")
print("===================================")
