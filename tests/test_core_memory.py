import cv2
import time

from core.detector import ObjectDetector
from core.scene_memory import SceneMemory


# =========================================
# ИНИЦИАЛИЗАЦИЯ МОДУЛЕЙ
# =========================================

detector = ObjectDetector(
    model_path="yolo11n.pt",
    conf_threshold=0.25,
    img_size=320
)

memory = SceneMemory(
    frame_width=640,
    frame_height=480
)


# =========================================
# ПОДКЛЮЧЕНИЕ КАМЕРЫ
# =========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Ошибка: камера не открыта")
    exit()


# =========================================
# ПАРАМЕТРЫ ТЕСТА
# =========================================

duration_seconds = 120

print("===================================")
print("Тест модуля Scene Memory запущен")
print(f"Длительность теста: {duration_seconds} секунд")
print("===================================")


# =========================================
# ОСНОВНОЙ ЦИКЛ
# =========================================

frame_id = 0
start_time = time.time()

while time.time() - start_time < duration_seconds:

    ret, frame = cap.read()

    if not ret:
        print("Ошибка чтения кадра")
        break

    # Детекция объектов
    detections = detector.detect(frame)

    # Обновление памяти сцены
    objects = memory.update(detections)

    print(f"\nКадр {frame_id}")

    for obj in objects.values():

        # Пропускаем исчезнувшие объекты
        if obj["missing_frames"] > 0:
            continue

        print(
            f"ID:{obj['id']} | "
            f"{obj['label']} | "
            f"dir:{obj['direction']} | "
            f"prox:{obj['proximity']:.2f} | "
            f"risk:{obj['risk_score']:.2f}"
        )

    frame_id += 1

    # Небольшая пауза для стабильности CPU
    time.sleep(0.05)


# =========================================
# ЗАВЕРШЕНИЕ ТЕСТА
# =========================================

cap.release()
cv2.destroyAllWindows()

print("\n===================================")
print("Тест завершён")
print(f"Обработано кадров: {frame_id}")
print("===================================")