import cv2
import time

from core.bytetrack_detector import ByteTrackDetector
from core.context_manager import ContextManager
from core.decision_engine import DecisionEngine
from core.audio_manager import AudioManager
from core.safe_decision_filter import SafeDecisionFilter
from core.temporal_stabilizer import TemporalStabilizer


CAMERA_ID = 0
PHASE = "C3"

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

PRIORITY_THRESHOLD = 0.45
HIGH_PRIORITY_THRESHOLD = 0.70


def compute_direction(bbox):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2

    if cx < FRAME_WIDTH * 0.33:
        return "left"
    elif cx > FRAME_WIDTH * 0.66:
        return "right"
    return "center"


def compute_proximity(bbox):
    x1, y1, x2, y2 = bbox

    box_area = max(0, x2 - x1) * max(0, y2 - y1)
    frame_area = FRAME_WIDTH * FRAME_HEIGHT

    area_norm = box_area / frame_area
    y_bottom = y2 / FRAME_HEIGHT

    proximity = min(
        1.0,
        0.65 * area_norm * 8 + 0.35 * y_bottom
    )

    return round(proximity, 3)


tracker = ByteTrackDetector(
    model_path="yolo11n.pt",
    conf_threshold=0.25,
    img_size=320,
    tracker_config="bytetrack.yaml"
)

context_manager = ContextManager(
    cooldown_seconds=10.0,
    min_proximity_change=0.15
)

decision_engine = DecisionEngine(
    priority_threshold=PRIORITY_THRESHOLD,
    high_priority_threshold=HIGH_PRIORITY_THRESHOLD
)

audio_manager = AudioManager(
    speech_duration=0.2,
    max_queue_size=2,
    min_repeat_interval=8.0,
    simulation=False,
    voice="en",
    speed=145
)

safe_filter = SafeDecisionFilter(
    global_cooldown=7.0,
    same_message_cooldown=15.0,
    same_object_family_cooldown=10.0,
    max_messages_per_cycle=1
)

temporal_stabilizer = TemporalStabilizer(
    min_seen_frames=8,
    direction_window=8,
    proximity_alpha=0.65
)

previous_proximity = {}

raw_decision_total = 0
safe_decision_total = 0

cap = cv2.VideoCapture(CAMERA_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    print("ERROR: camera not accessible")
    audio_manager.stop()
    exit()


print("=================================")
print("ASSISTIVE AI SYSTEM STARTED")
print("ByteTrack ENABLED")
print("Audio async ENABLED")
print("SAFE TRUE counters ENABLED")
print("Phase:", PHASE)
print("Press Q to stop")
print("=================================")


try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("ERROR: camera frame not received")
            break

        frame_start = time.time()

        tracked_objects = tracker.track(frame)

        objects = {}

        for obj in tracked_objects:
            track_id = obj.get("track_id")

            if track_id is None:
                continue

            bbox = obj["bbox"]
            proximity = compute_proximity(bbox)

            objects[track_id] = {
                "id": track_id,
                "label": obj["label"],
                "confidence": obj["confidence"],
                "bbox": bbox,
                "direction": compute_direction(bbox),
                "proximity": proximity,
                "previous_proximity": previous_proximity.get(track_id, proximity),
                "risk_score": proximity,
                "missing_frames": 0,
            }

            previous_proximity[track_id] = proximity

        objects = temporal_stabilizer.update(objects)

        final_messages = []
        safe_messages = []

        if PHASE == "C1":
            final_messages = []

        elif PHASE == "C2":
            context_result = context_manager.filter_messages(objects)
            final_messages = context_result["messages"]

        elif PHASE == "C3":
            context_result = context_manager.filter_messages(objects)
            context_messages = context_result["messages"]

            decisions = decision_engine.decide(
                context_messages=context_messages,
                objects=objects
            )

            final_messages = [d["message"] for d in decisions]

            raw_decision_total += len(final_messages)

            safe_messages = safe_filter.filter(final_messages)

            safe_decision_total += len(safe_messages)

        else:
            final_messages = []

        for obj in tracked_objects:
            bbox = obj["bbox"]
            x1, y1, x2, y2 = map(int, bbox)

            label = obj["label"]
            track_id = obj.get("track_id", -1)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"{label} ID:{track_id}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        y_offset = 30

        for msg in safe_messages:
            cv2.putText(
                frame,
                msg,
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            y_offset += 30

            print("[AUDIO_QUEUE]", msg)
            audio_manager.speak(msg)

        fps = 1.0 / max(time.time() - frame_start, 1e-6)

        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (20, FRAME_HEIGHT - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"RAW_C3: {raw_decision_total} | SAFE: {safe_decision_total}",
            (20, FRAME_HEIGHT - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        print(
            f"FPS:{fps:.2f} | "
            f"Objects:{len(objects)} | "
            f"RawC3:{raw_decision_total} | "
            f"SAFE:{safe_decision_total} | "
            f"CurrentSafe:{safe_messages}"
        )

        cv2.imshow("Assistive AI System", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

except KeyboardInterrupt:
    print("SYSTEM INTERRUPTED")


cap.release()
cv2.destroyAllWindows()
audio_manager.stop()

print("=================================")
print("SYSTEM STOPPED")
print(f"Final RAW_C3 decisions: {raw_decision_total}")
print(f"Final SAFE decisions: {safe_decision_total}")
print("=================================")