from ultralytics import YOLO
import cv2
import time
import os

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : caméra non ouverte")
    exit()

os.makedirs("outputs", exist_ok=True)

frame_count = 0

try:
    while frame_count < 100:
        ret, frame = cap.read()

        if not ret:
            print("Erreur : frame non lue")
            break

        start = time.time()
        results = model(frame, verbose=False)
        latency = (time.time() - start) * 1000
        fps = 1000 / latency if latency > 0 else 0

        detections = len(results[0].boxes)

        print(f"Frame: {frame_count} | FPS: {fps:.2f} | Latence: {latency:.1f} ms | Détections: {detections}")

        if frame_count % 10 == 0:
            annotated = results[0].plot()
            cv2.imwrite(f"outputs/frame_{frame_count:03d}.jpg", annotated)

        frame_count += 1

except KeyboardInterrupt:
    print("\nArrêt manuel du test.")

finally:
    cap.release()
    print("Caméra libérée. Test terminé.")