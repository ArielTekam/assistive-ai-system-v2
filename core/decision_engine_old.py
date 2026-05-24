class DecisionEngine:
    """
    Модуль оценки приоритета и принятия решений.

    Назначение:
    - оценивает важность каждого объекта;
    - выбирает только наиболее релевантные объекты;
    - игнорирует второстепенные объекты;
    - формирует сообщения для последующей генерации аудио.
    """

    def __init__(
        self,
        priority_threshold=0.45,
        high_priority_threshold=0.70
    ):
        self.priority_threshold = priority_threshold
        self.high_priority_threshold = high_priority_threshold

        self.class_weights = {
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
            "cup": 0.2,
            "book": 0.2,
            "cell phone": 0.3,
            "laptop": 0.3,
        }

    def _centrality_score(self, obj, frame_width=640):
        """
        Оценивает, насколько объект находится близко к центру кадра.
        """
        bbox = obj.get("bbox")

        if not bbox:
            return 0.0

        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        frame_center = frame_width / 2

        distance = abs(center_x - frame_center)

        return max(0.0, 1.0 - distance / frame_center)

    def compute_priority_score(self, obj):
        """
        Вычисляет итоговый score приоритета.

        Учитываются:
        - близость объекта;
        - центральное положение;
        - класс объекта;
        - риск, вычисленный в SceneMemory.
        """
        label = obj.get("label", "")
        proximity = obj.get("proximity", 0.0)
        risk_score = obj.get("risk_score", 0.0)
        centrality = self._centrality_score(obj)

        class_weight = self.class_weights.get(label, 0.2)

        priority_score = (
            0.40 * proximity +
            0.25 * centrality +
            0.25 * class_weight +
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
        Формирует сообщение на основе объекта и уровня приоритета.
        """
        label = obj.get("label", "object")
        direction = obj.get("direction", "center")
        proximity = obj.get("proximity", 0.0)

        if label == "person":
            if proximity >= 0.75:
                if direction == "center":
                    return "Person very close ahead"
                return f"Person very close on your {direction}"

            if direction == "center":
                return "Person ahead"

            return f"Person on your {direction}"

        if level == "HIGH":
            if direction == "center":
                return f"{label} ahead"
            return f"{label} on your {direction}"

        if level == "MEDIUM":
            if direction == "center":
                return f"{label} ahead"
            return f"{label} on your {direction}"

        return None

    def decide(self, context_messages, objects):
        """
        Prend les messages filtrés par ContextManager et applique une sélection par priorité.

        :param context_messages: messages issus du ContextManager
        :param objects: objets issus de SceneMemory
        :return: liste des décisions finales
        """
        decisions = []

        for item in context_messages:
            obj_id = item.get("object_id")

            if obj_id not in objects:
                continue

            obj = objects[obj_id]

            if obj.get("missing_frames", 0) > 0:
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
                "label": obj.get("label"),
                "priority_score": round(score, 3),
                "priority_level": level,
                "message": message,
                "reason": item.get("reason", "")
            })

        decisions = sorted(
            decisions,
            key=lambda x: x["priority_score"],
            reverse=True
        )

        return decisions[:2]
