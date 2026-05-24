from ultralytics import YOLO
import cv2
import time

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : caméra non ouverte")
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Erreur lecture frame")
        break

    start = time.time()

    results = model(frame, verbose=False)

    latency = (time.time() - start) * 1000
    fps = 1000 / latency if latency > 0 else 0

    detections = len(results[0].boxes)

    print(
        f"Frame: {frame_count} | "
        f"FPS: {fps:.2f} | "
        f"Latence: {latency:.1f} ms | "
        f"Détections: {detections}"
    )

    frame_count += 1

except_count = 0

cap.release()
print("Test terminé.")