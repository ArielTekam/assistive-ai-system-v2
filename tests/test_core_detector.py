import cv2
from core.detector import ObjectDetector

detector = ObjectDetector(
    model_path="yolo11n.pt",
    conf_threshold=0.25,
    img_size=320
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : caméra non ouverte")
    exit()

ret, frame = cap.read()
cap.release()

if not ret:
    print("Erreur : image non lue")
    exit()

detections = detector.detect(frame)

print("Détections :")
for det in detections:
    print(det)
