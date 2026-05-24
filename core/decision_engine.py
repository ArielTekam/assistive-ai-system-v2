class DecisionEngine:
    """
    Модуль оценки приоритета и принятия решений.

    Назначение:
    - оценивает важность объектов;
    - учитывает близость, центральность, класс объекта и динамику приближения;
    - игнорирует второстепенные или нерелевантные классы;
    - выбирает только сообщения, которые имеют практическую значимость для пользователя.
    """

    def __init__(
        self,
        priority_threshold=0.45,
        high_priority_threshold=0.70
    ):
        self.priority_threshold = priority_threshold
        self.high_priority_threshold = high_priority_threshold

        # Классы, которые считаются действительно важными для помощи при движении
        self.important_classes = {
            "person",
            "car",
            "bus",
            "truck",
            "motorcycle",
            "bicycle",
            "dog"
        }

        # Вес класса в общей оценке приоритета
        self.class_weights = {
            "person": 1.0,
            "car": 1.0,
            "bus": 1.0,
            "truck": 1.0,
            "motorcycle": 0.9,
            "bicycle": 0.8,
            "dog": 0.7,
        }

    def _centrality_score(self, obj, frame_width=640):
        """
        Оценивает, насколько объект находится близко к центру кадра.
        Центральные объекты считаются более важными для движения пользователя.
        """
        bbox = obj.get("bbox")

        if not bbox:
            return 0.0

        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        frame_center = frame_width / 2
        distance = abs(center_x - frame_center)

        return max(0.0, 1.0 - distance / frame_center)

    def _approach_score(self, obj):
        """
        Оценивает приближение объекта к пользователю.

        Если показатель близости увеличивается, объект считается более важным.
        Если объект не приближается, значение равно 0.
        """
        current = obj.get("proximity", 0.0)
        previous = obj.get("previous_proximity", current)

        delta = current - previous

        if delta <= 0:
            return 0.0

        return min(delta * 5.0, 1.0)

    def compute_priority_score(self, obj):
        """
        Вычисляет итоговый score приоритета.

        Компоненты:
        - proximity: объект близко;
        - centrality: объект находится перед пользователем;
        - class_weight: тип объекта важен для навигации;
        - approach: объект приближается;
        - risk_score: риск, вычисленный в SceneMemory.
        """
        label = obj.get("label", "")

        # Нерелевантные классы не получают приоритет
        if label not in self.important_classes:
            return 0.0

        proximity = obj.get("proximity", 0.0)
        centrality = self._centrality_score(obj)
        class_weight = self.class_weights.get(label, 0.0)
        approach = self._approach_score(obj)
        risk_score = obj.get("risk_score", 0.0)

        priority_score = (
            0.35 * proximity +
            0.20 * centrality +
            0.20 * class_weight +
            0.15 * approach +
            0.10 * risk_score
        )

        return min(priority_score, 1.0)

    def _priority_level(self, score):
        """
        Переводит числовой score в уровень приоритета.
        """
        if score >= self.high_priority_threshold:
            return "HIGH"

        if score >= self.priority_threshold:
            return "MEDIUM"

        return "LOW"

    def _build_message(self, obj, level):
        """
        Формирует итоговое сообщение на основе объекта и уровня приоритета.
        """
        label = obj.get("label", "object")
        direction = obj.get("direction", "center")
        proximity = obj.get("proximity", 0.0)
        approach = self._approach_score(obj)

        if label == "person":
            if proximity >= 0.75:
                if direction == "center":
                    return "Person very close ahead"
                return f"Person very close on your {direction}"

            if approach >= 0.25:
                if direction == "center":
                    return "Person approaching ahead"
                return f"Person approaching on your {direction}"

            if direction == "center":
                return "Person ahead"

            return f"Person on your {direction}"

        if label in {"car", "bus", "truck", "motorcycle", "bicycle"}:
            if proximity >= 0.65 or level == "HIGH":
                if direction == "center":
                    return f"{label} ahead"
                return f"{label} on your {direction}"

            if approach >= 0.25:
                if direction == "center":
                    return f"{label} approaching ahead"
                return f"{label} approaching on your {direction}"

        if label == "dog":
            if level in ("HIGH", "MEDIUM"):
                if direction == "center":
                    return "Dog ahead"
                return f"Dog on your {direction}"

        return None

    def decide(self, context_messages, objects):
        """
        Принимает сообщения от ContextManager и применяет фильтрацию по приоритету.

        :param context_messages: сообщения, прошедшие контекстную память
        :param objects: объекты из SceneMemory
        :return: список финальных решений
        """
        decisions = []

        for item in context_messages:
            obj_id = item.get("object_id")

            if obj_id not in objects:
                continue

            obj = objects[obj_id]

            if obj.get("missing_frames", 0) > 0:
                continue

            label = obj.get("label", "")

            # Семантическая фильтрация: исключаем нерелевантные классы
            if label not in self.important_classes:
                continue

            score = self.compute_priority_score(obj)
            level = self._priority_level(score)

            if score < self.priority_threshold:
                continue

            message = self._build_message(obj, level)

            if not message:
                continue

            decisions.append({
                "object_id": obj_id,
                "label": label,
                "priority_score": round(score, 3),
                "priority_level": level,
                "message": message,
                "reason": item.get("reason", ""),
                "proximity": round(obj.get("proximity", 0.0), 3),
                "approach": round(self._approach_score(obj), 3),
                "risk_score": round(obj.get("risk_score", 0.0), 3)
            })

        decisions = sorted(
            decisions,
            key=lambda x: x["priority_score"],
            reverse=True
        )

        # Мы сознательно ограничиваем количество сообщений максимум двумя, чтобы избежать перегрузки мозга
        return decisions[:2]