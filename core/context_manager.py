import time


class ContextManager:
    """
    Модуль контекстной памяти.

    Назначение:
    - уменьшает количество повторяющихся сообщений;
    - запоминает уже озвученные объекты;
    - пропускает только новые или значимые изменения;
    - снижает когнитивную нагрузку пользователя.
    """

    def __init__(
        self,
        cooldown_seconds=5.0,
        min_proximity_change=0.08
    ):
        """
        Инициализация контекстной памяти.

        :param cooldown_seconds: минимальное время между повторными сообщениями
        :param min_proximity_change: минимальное изменение близости для нового сообщения
        """
        self.cooldown_seconds = cooldown_seconds
        self.min_proximity_change = min_proximity_change
        self.last_messages = {}

    def _build_message(self, obj):
        """
        Формирует простое текстовое сообщение для объекта.
        """
        label = obj.get("label", "object")
        direction = obj.get("direction", "center")

        if direction == "center":
            return f"{label} ahead"

        return f"{label} on your {direction}"

    def _has_significant_change(self, obj, previous_data):
        """
        Проверяет, произошло ли значимое изменение объекта.
        """
        current_direction = obj.get("direction", "center")
        previous_direction = previous_data.get("direction", current_direction)

        if current_direction != previous_direction:
            return True

        current_proximity = obj.get("proximity", 0.0)
        previous_proximity = previous_data.get("proximity", current_proximity)

        if abs(current_proximity - previous_proximity) >= self.min_proximity_change:
            return True

        return False

    def filter_messages(self, objects):
        """
        Фильтрует сообщения на основе контекста.

        :param objects: словарь объектов из SceneMemory
        :return: список сообщений, которые действительно нужно озвучить
        """
        now = time.time()
        messages = []
        raw_messages_count = 0
        filtered_count = 0

        for obj in objects.values():

            if obj.get("missing_frames", 0) > 0:
                continue

            raw_messages_count += 1

            obj_id = obj["id"]
            message = self._build_message(obj)

            if obj_id not in self.last_messages:
                self.last_messages[obj_id] = {
                    "time": now,
                    "message": message,
                    "direction": obj.get("direction", "center"),
                    "proximity": obj.get("proximity", 0.0)
                }

                messages.append({
                    "object_id": obj_id,
                    "message": message,
                    "reason": "new_object"
                })
                continue

            previous = self.last_messages[obj_id]
            time_since_last = now - previous["time"]

            if self._has_significant_change(obj, previous):
                self.last_messages[obj_id] = {
                    "time": now,
                    "message": message,
                    "direction": obj.get("direction", "center"),
                    "proximity": obj.get("proximity", 0.0)
                }

                messages.append({
                    "object_id": obj_id,
                    "message": message,
                    "reason": "significant_change"
                })
                continue

            if time_since_last >= self.cooldown_seconds:
                self.last_messages[obj_id] = {
                    "time": now,
                    "message": message,
                    "direction": obj.get("direction", "center"),
                    "proximity": obj.get("proximity", 0.0)
                }

                messages.append({
                    "object_id": obj_id,
                    "message": message,
                    "reason": "cooldown"
                })
                continue

            filtered_count += 1

        return {
            "messages": messages,
            "raw_messages_count": raw_messages_count,
            "filtered_count": filtered_count
        }
