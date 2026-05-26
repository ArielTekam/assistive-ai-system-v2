import cv2
import time
import argparse
import subprocess
import psutil

from core.bytetrack_detector import ByteTrackDetector
from core.context_manager import ContextManager
from core.decision_engine import DecisionEngine
from core.audio_manager import AudioManager
from core.safe_decision_filter import SafeDecisionFilter
from core.temporal_stabilizer import TemporalStabilizer


# ============================================================
# Основной видеотест полного когнитивного пайплайна
# Raspberry Pi 5 + YOLO11n + ByteTrack + C2 + C3 + SAFE + Audio
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True, help="Путь к тестовому видео")
parser.add_argument("--display", action="store_true", help="Показать окно OpenCV")
args = parser.parse_args()

VIDEO_PATH = args.video
PHASE = "C4_SAFE_AUDIO"

# ------------------------------------------------------------
# Параметры видеопотока
# ------------------------------------------------------------

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ------------------------------------------------------------
# Пороговые значения для модуля принятия решений
# ------------------------------------------------------------

PRIORITY_THRESHOLD = 0.45
HIGH_PRIORITY_THRESHOLD = 0.70

# ------------------------------------------------------------
# Параметры повторного объявления в статичной сцене
# ------------------------------------------------------------

STATIC_REANNOUNCE_COOLDOWN = 18.0
VERY_CLOSE_THRESHOLD = 0.75


def get_temperature():
    """
    Получает текущую температуру Raspberry Pi через vcgencmd.

    Если команда недоступна, возвращается -1.0.
    """
    try:
        temp = subprocess.check_output(
            ["vcgencmd", "measure_temp"]
        ).decode()

        temp = temp.replace("temp=", "")
        temp = temp.replace("'C\n", "")
        return float(temp)

    except Exception:
        return -1.0


def compute_direction(bbox):
    """
    Определяет приблизительное направление объекта в кадре.

    Возвращает:
    - left   : объект слева
    - center : объект по центру
    - right  : объект справа
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2

    if cx < FRAME_WIDTH * 0.33:
        return "left"
    elif cx > FRAME_WIDTH * 0.66:
        return "right"

    return "center"


def compute_proximity(bbox):
    """
    Оценивает относительную близость объекта.

    Используются два признака:
    1. нормализованная площадь bounding box;
    2. нижняя координата объекта в кадре.

    Это не метрическая дистанция в метрах,
    а эвристическая оценка близости.
    """
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


def is_scene_static(objects, previous_objects, proximity_delta_threshold=0.04):
    """
    Проверяет, является ли сцена относительно статичной.

    Логика:
    - если большинство объектов сохранили похожую близость,
      сцена считается статичной;
    - если объекты существенно изменились,
      сцена считается динамичной.

    Эта функция используется для уменьшения повторных аудиообъявлений.
    """
    if not objects:
        return True

    stable_count = 0
    total_count = 0

    for obj_id, obj in objects.items():
        if obj_id in previous_objects:
            total_count += 1

            prev_prox = previous_objects[obj_id].get(
                "proximity",
                obj["proximity"]
            )

            curr_prox = obj.get("proximity", prev_prox)

            if abs(curr_prox - prev_prox) < proximity_delta_threshold:
                stable_count += 1

    if total_count == 0:
        return False

    return stable_count / total_count >= 0.75


# ============================================================
# Инициализация основных модулей системы
# ============================================================

# Детекция объектов + трекинг ByteTrack
tracker = ByteTrackDetector(
    model_path="yolo11n.pt",
    conf_threshold=0.25,
    img_size=320,
    tracker_config="bytetrack.yaml"
)

# Контекстный фильтр C2:
# уменьшает повторы и микроколебания сцены
context_manager = ContextManager(
    cooldown_seconds=10.0,
    min_proximity_change=0.15
)

# Модуль принятия решений C3:
# выбирает события, достаточно важные для озвучивания
decision_engine = DecisionEngine(
    priority_threshold=PRIORITY_THRESHOLD,
    high_priority_threshold=HIGH_PRIORITY_THRESHOLD
)

# Асинхронный аудиомодуль:
# отделяет генерацию речи от обработки видео
audio_manager = AudioManager(
    speech_duration=0.2,
    max_queue_size=2,
    min_repeat_interval=8.0,
    simulation=False,
    voice="en",
    speed=145
)

# SAFE-фильтр:
# последняя когнитивная защита перед аудио
safe_filter = SafeDecisionFilter(
    global_cooldown=7.0,
    same_message_cooldown=15.0,
    same_object_family_cooldown=10.0,
    max_messages_per_cycle=1
)

# Временная стабилизация объектов:
# уменьшает скачки направления и близости
temporal_stabilizer = TemporalStabilizer(
    min_seen_frames=8,
    direction_window=8,
    proximity_alpha=0.65
)


# ============================================================
# Переменные состояния эксперимента
# ============================================================

previous_proximity = {}
previous_objects_state = {}
last_static_announcement_time = 0

raw_decision_total = 0
safe_decision_total = 0
frame_count = 0
total_objects_detected = 0

fps_values = []
latency_values = []
cpu_values = []
ram_values = []
temperature_values = []

unique_track_ids = set()


# ============================================================
# Открытие видеопотока
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"ОШИБКА: видео недоступно: {VIDEO_PATH}")
    audio_manager.stop()
    exit()


print("============================================")
print("ЗАПУСК ВИДЕОТЕСТА СИСТЕМЫ")
print("============================================")
print(f"Фаза тестирования: {PHASE}")
print("Детекция: YOLO11n")
print("Трекинг: ByteTrack включён")
print("Контекстный фильтр C2 включён")
print("Модуль принятия решений C3 включён")
print("SAFE-фильтр включён")
print("Аудио: асинхронное TTS включено")
print(f"Видео: {VIDEO_PATH}")
print("Нажмите Q для остановки при включённом display")
print("============================================")


try:
    # ========================================================
    # Главный цикл обработки видеопотока
    # ========================================================

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame_start = time.time()

        # ----------------------------------------------------
        # C1: детекция объектов + трекинг ByteTrack
        # ----------------------------------------------------

        tracked_objects = tracker.track(frame)

        objects = {}

        for obj in tracked_objects:
            track_id = obj.get("track_id")

            if track_id is None:
                continue

            unique_track_ids.add(track_id)

            bbox = obj["bbox"]
            proximity = compute_proximity(bbox)

            objects[track_id] = {
                "id": track_id,
                "label": obj["label"],
                "confidence": obj["confidence"],
                "bbox": bbox,
                "direction": compute_direction(bbox),
                "proximity": proximity,
                "previous_proximity": previous_proximity.get(
                    track_id,
                    proximity
                ),
                "risk_score": proximity,
                "missing_frames": 0,
            }

            previous_proximity[track_id] = proximity

        total_objects_detected += len(objects)

        # ----------------------------------------------------
        # Временная стабилизация объектов
        # ----------------------------------------------------

        objects = temporal_stabilizer.update(objects)

        # ----------------------------------------------------
        # C2: контекстная фильтрация
        # ----------------------------------------------------

        context_result = context_manager.filter_messages(objects)
        context_messages = context_result["messages"]

        # ----------------------------------------------------
        # C3: принятие решений
        # ----------------------------------------------------

        decisions = decision_engine.decide(
            context_messages=context_messages,
            objects=objects
        )

        final_messages = [d["message"] for d in decisions]
        raw_decision_total += len(final_messages)

        # ----------------------------------------------------
        # SAFE: финальная фильтрация перед аудио
        # ----------------------------------------------------

        candidate_safe_messages = safe_filter.filter(final_messages)

        # ----------------------------------------------------
        # Проверка статичности сцены
        # ----------------------------------------------------

        scene_static = is_scene_static(
            objects,
            previous_objects_state
        )

        current_time = time.time()
        safe_messages = []

        # ----------------------------------------------------
        # Логика повторного объявления:
        # - если сцена статична, сообщения подавляются;
        # - если объект очень близко, сообщение разрешается;
        # - если сцена динамична, SAFE-сообщения проходят дальше.
        # ----------------------------------------------------

        if scene_static:
            has_very_close_object = any(
                obj.get("proximity", 0) >= VERY_CLOSE_THRESHOLD
                for obj in objects.values()
            )

            if has_very_close_object:
                safe_messages = candidate_safe_messages
            else:
                if (
                    current_time - last_static_announcement_time
                    >= STATIC_REANNOUNCE_COOLDOWN
                ):
                    safe_messages = candidate_safe_messages[:1]

                    if safe_messages:
                        last_static_announcement_time = current_time
                else:
                    safe_messages = []
        else:
            safe_messages = candidate_safe_messages

        safe_decision_total += len(safe_messages)

        # ----------------------------------------------------
        # Сохранение состояния объектов для следующего кадра
        # ----------------------------------------------------

        previous_objects_state = {
            obj_id: obj.copy()
            for obj_id, obj in objects.items()
        }

        # ----------------------------------------------------
        # Асинхронная генерация аудио
        # ----------------------------------------------------

        for msg in safe_messages:
            print("[АУДИО_ОЧЕРЕДЬ]", msg)
            audio_manager.speak(msg)

        # ----------------------------------------------------
        # Расчёт производительности
        # ----------------------------------------------------

        latency_ms = (time.time() - frame_start) * 1000
        fps = 1000.0 / max(latency_ms, 1e-6)

        fps_values.append(fps)
        latency_values.append(latency_ms)

        # ----------------------------------------------------
        # Сбор системных метрик Raspberry Pi
        # ----------------------------------------------------

        if frame_count % 30 == 0:
            cpu_values.append(psutil.cpu_percent(interval=None))

            ram = psutil.virtual_memory()
            ram_used_gb = ram.used / (1024 ** 3)
            ram_values.append(ram_used_gb)

            temperature = get_temperature()
            if temperature > 0:
                temperature_values.append(temperature)

        # ----------------------------------------------------
        # Текущий лог обработки кадра
        # ----------------------------------------------------

        print(
            f"Кадр:{frame_count} | "
            f"FPS:{fps:.2f} | "
            f"Задержка:{latency_ms:.1f} мс | "
            f"Объекты:{len(objects)} | "
            f"Сцена_статична:{scene_static} | "
            f"RawC3:{raw_decision_total} | "
            f"SAFE:{safe_decision_total} | "
            f"Текущие_SAFE:{safe_messages}"
        )

        # ----------------------------------------------------
        # Отображение окна OpenCV при параметре --display
        # ----------------------------------------------------

        if args.display:
            for obj in tracked_objects:
                x1, y1, x2, y2 = map(int, obj["bbox"])
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

            cv2.putText(
                frame,
                (
                    f"FPS:{fps:.2f} "
                    f"RAW_C3:{raw_decision_total} "
                    f"SAFE:{safe_decision_total}"
                ),
                (20, FRAME_HEIGHT - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"STATIC:{scene_static}",
                (20, FRAME_HEIGHT - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2
            )

            cv2.imshow("Assistive AI Video Test", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        frame_count += 1

except KeyboardInterrupt:
    print("СИСТЕМА ОСТАНОВЛЕНА ПОЛЬЗОВАТЕЛЕМ")


# ============================================================
# Завершение видеопотока и аудиосистемы
# ============================================================

cap.release()
cv2.destroyAllWindows()
audio_manager.stop()


# ============================================================
# Расчёт итоговых метрик
# ============================================================

mean_fps = sum(fps_values) / len(fps_values) if fps_values else 0
mean_latency = sum(latency_values) / len(latency_values) if latency_values else 0
max_latency = max(latency_values) if latency_values else 0

mean_cpu = sum(cpu_values) / len(cpu_values) if cpu_values else 0
mean_ram = sum(ram_values) / len(ram_values) if ram_values else 0
mean_temp = sum(temperature_values) / len(temperature_values) if temperature_values else -1

audio_stats = audio_manager.get_stats() if hasattr(audio_manager, "get_stats") else {}

spoken_count = audio_stats.get("spoken_count", "N/A")
dropped_count = audio_stats.get("dropped_count", "N/A")
repeated_blocked_count = audio_stats.get("repeated_blocked_count", "N/A")
tts_error_count = audio_stats.get("tts_error_count", "N/A")
queue_size = audio_stats.get("queue_size", "N/A")


# ============================================================
# Финальная статистика эксперимента
# ============================================================

print("\n============================================")
print("ИТОГОВАЯ СТАТИСТИКА ВИДЕОТЕСТА")
print("============================================")

print(f"Видео: {VIDEO_PATH}")
print(f"Фаза: {PHASE}")

print("\n--- Производительность ---")
print(f"Обработано кадров: {frame_count}")
print(f"Средний FPS: {mean_fps:.2f}")
print(f"Средняя задержка: {mean_latency:.2f} мс")
print(f"Максимальная задержка: {max_latency:.2f} мс")
print(f"Средняя загрузка CPU: {mean_cpu:.1f} %")
print(f"Среднее использование RAM: {mean_ram:.2f} GB")

if mean_temp > 0:
    print(f"Средняя температура Raspberry Pi: {mean_temp:.1f} °C")
else:
    print("Средняя температура Raspberry Pi: недоступна")

print("\n--- Перцепция и трекинг ---")
print(f"Всего обнаруженных объектов: {total_objects_detected}")
print(f"Уникальные ID трекинга: {len(unique_track_ids)}")

print("\n--- Когнитивный пайплайн ---")
print(f"Финальные RAW C3 решения: {raw_decision_total}")
print(f"Финальные SAFE решения: {safe_decision_total}")

if raw_decision_total > 0:
    reduction = 100 * (1 - safe_decision_total / raw_decision_total)
    print(f"Снижение после SAFE: {reduction:.1f} %")
else:
    print("Снижение после SAFE: N/A")

print("\n--- Аудио ---")
print(f"Озвученные сообщения: {spoken_count}")
print(f"Отброшенные сообщения: {dropped_count}")
print(f"Заблокированные повторы: {repeated_blocked_count}")
print(f"Ошибки TTS: {tts_error_count}")
print(f"Размер аудиоочереди: {queue_size}")

print("============================================")