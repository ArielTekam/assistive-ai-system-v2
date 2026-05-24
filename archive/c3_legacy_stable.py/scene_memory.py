import time
import math


class SceneMemory:
    """
    Модуль временной памяти сцены.

    Назначение:
    - связывает обнаруженные объекты между последовательными кадрами;
    - присваивает объектам устойчивые идентификаторы;
    - отслеживает направление, приближение и исчезновение объектов;
    - вычисляет простую оценку близости и риска;
    - подготавливает данные для Decision Engine.
    """

    def __init__(
        self,
        frame_width=640,
        frame_height=480,
        max_distance=140,
        max_missing_frames=10,
        smoothing_alpha=0.3
    ):
        """
        Инициализация памяти сцены.

        :param frame_width: ширина кадра
        :param frame_height: высота кадра
        :param max_distance: максимальная дистанция для сопоставления объектов
        :param max_missing_frames: число кадров, после которого объект удаляется
        :param smoothing_alpha: коэффициент сглаживания координат
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.max_distance = max_distance
        self.max_missing_frames = max_missing_frames
        self.smoothing_alpha = smoothing_alpha

        self.objects = {}
        self.next_id = 1

    def _center(self, bbox):
        """Вычисляет центр ограничивающей рамки."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _distance(self, p1, p2):
        """Вычисляет евклидово расстояние между двумя точками."""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def _direction(self, center):
        """
        Определяет положение объекта в кадре:
        left / center / right.
        """
        x, _ = center

        if x < self.frame_width / 3:
            return "left"
        if x > 2 * self.frame_width / 3:
            return "right"
        return "center"

    def _proximity(self, bbox):
        """
        Оценивает близость объекта по высоте bounding box.
        Чем выше объект в кадре, тем он считается ближе.
        """
        x1, y1, x2, y2 = bbox
        height = max(0, y2 - y1)
        return min(height / self.frame_height, 1.0)

    def _risk_score(self, obj):
        """
        Вычисляет простую оценку риска.

        Учитываются:
        - близость объекта;
        - изменение близости;
        - скорость движения;
        - тип объекта.
        """
        proximity = obj.get("proximity", 0.0)
        previous_proximity = obj.get("previous_proximity", proximity)
        proximity_delta = max(0.0, proximity - previous_proximity)

        vx, vy = obj.get("velocity", (0.0, 0.0))
        speed = min(math.sqrt(vx * vx + vy * vy) / 300.0, 1.0)

        label = obj.get("label", "")

        class_weight = {
            "person": 1.0,
            "car": 1.0,
            "bus": 1.0,
            "truck": 1.0,
            "motorcycle": 0.9,
            "bicycle": 0.8,
            "dog": 0.7,
            "chair": 0.5,
            "bench": 0.5,
            "bottle": 0.2,
        }.get(label, 0.3)

        score = (
            0.5 * proximity +
            0.3 * proximity_delta +
            0.2 * class_weight +
            0.1 * speed
        )

        return min(score, 1.0)

    def _match_detection(self, detection, used_ids):
        """
        Сопоставляет новое обнаружение с уже известным объектом.

        Используется простая эвристика:
        - совпадение класса;
        - минимальное расстояние между центрами;
        - объект не должен быть уже использован в текущем кадре.
        """
        det_center = self._center(detection["bbox"])
        det_label = detection["label"]

        best_id = None
        best_distance = self.max_distance

        for obj_id, obj in self.objects.items():
            if obj_id in used_ids:
                continue

            if obj["label"] != det_label:
                continue

            if obj["missing_frames"] > self.max_missing_frames:
                continue

            dist = self._distance(det_center, obj["center"])

            if dist < best_distance:
                best_distance = dist
                best_id = obj_id

        return best_id

    def update(self, detections):
        """
        Обновляет память сцены на основе текущих обнаружений.

        :param detections: список объектов от детектора
        :return: словарь активных и недавно пропавших объектов
        """
        now = time.time()

        used_ids = set()
        matched_ids = set()

        for det in detections:
            bbox = det["bbox"]
            center = self._center(bbox)
            direction = self._direction(center)
            proximity = self._proximity(bbox)

            matched_id = self._match_detection(det, used_ids)

            if matched_id is None:
                obj_id = self.next_id
                self.next_id += 1

                self.objects[obj_id] = {
                    "id": obj_id,
                    "label": det["label"],
                    "confidence": det["confidence"],
                    "bbox": bbox,
                    "center": center,
                    "smoothed_center": center,
                    "previous_center": None,
                    "direction": direction,
                    "previous_direction": None,
                    "proximity": proximity,
                    "previous_proximity": proximity,
                    "velocity": (0.0, 0.0),
                    "risk_score": 0.0,
                    "missing_frames": 0,
                    "created_at": now,
                    "last_seen": now,
                    "already_announced": False,
                    "last_message": None,
                    "last_message_time": 0.0,
                }

                used_ids.add(obj_id)
                matched_ids.add(obj_id)

            else:
                obj = self.objects[matched_id]

                old_center = obj["smoothed_center"]
                old_time = obj["last_seen"]

                alpha = self.smoothing_alpha
                smoothed = (
                    alpha * center[0] + (1 - alpha) * old_center[0],
                    alpha * center[1] + (1 - alpha) * old_center[1],
                )

                dt = max(now - old_time, 1e-6)
                velocity = (
                    (smoothed[0] - old_center[0]) / dt,
                    (smoothed[1] - old_center[1]) / dt,
                )

                obj["confidence"] = det["confidence"]
                obj["bbox"] = bbox
                obj["previous_center"] = obj["center"]
                obj["center"] = center
                obj["smoothed_center"] = smoothed
                obj["previous_direction"] = obj["direction"]
                obj["direction"] = direction
                obj["previous_proximity"] = obj["proximity"]
                obj["proximity"] = proximity
                obj["velocity"] = velocity
                obj["missing_frames"] = 0
                obj["last_seen"] = now
                obj["risk_score"] = self._risk_score(obj)

                used_ids.add(matched_id)
                matched_ids.add(matched_id)

        for obj_id, obj in list(self.objects.items()):
            if obj_id not in matched_ids:
                obj["missing_frames"] += 1

            if obj["missing_frames"] > self.max_missing_frames:
                del self.objects[obj_id]

        return self.objects
