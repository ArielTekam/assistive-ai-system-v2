import time


class SafeDecisionFilter:
    """
    C3 SAFE MODE v2

    Filtre final avant audio.
    Il réduit :
    - les répétitions exactes ;
    - les messages similaires ;
    - les oscillations left/center/right ;
    - la surcharge audio.
    """

    def __init__(
        self,
        global_cooldown=7.0,
        same_message_cooldown=15.0,
        same_object_family_cooldown=10.0,
        max_messages_per_cycle=1
    ):
        self.global_cooldown = global_cooldown
        self.same_message_cooldown = same_message_cooldown
        self.same_object_family_cooldown = same_object_family_cooldown
        self.max_messages_per_cycle = max_messages_per_cycle

        self.last_global_time = 0.0
        self.last_message_time = {}
        self.last_family_time = {}

    def _family_key(self, msg):
        msg = msg.lower()

        if "person" in msg:
            if "very close" in msg:
                return "person_very_close"
            return "person"

        if "car" in msg:
            return "car"

        if "bicycle" in msg:
            return "bicycle"

        if "motorcycle" in msg:
            return "motorcycle"

        if "chair" in msg:
            return "chair"

        return msg

    def _is_critical(self, msg):
        msg = msg.lower()

        critical_patterns = [
            "very close ahead",
            "danger",
            "obstacle ahead",
            "person very close ahead",
        ]

        return any(pattern in msg for pattern in critical_patterns)

    def filter(self, messages):
        now = time.time()
        accepted = []

        if not messages:
            return accepted

        for msg in messages:
            if not msg:
                continue

            msg = msg.strip()
            family = self._family_key(msg)
            critical = self._is_critical(msg)

            last_msg_time = self.last_message_time.get(msg, 0.0)
            last_family_time = self.last_family_time.get(family, 0.0)

            if not critical:
                if now - self.last_global_time < self.global_cooldown:
                    continue

            if now - last_msg_time < self.same_message_cooldown:
                continue

            if not critical:
                if now - last_family_time < self.same_object_family_cooldown:
                    continue

            accepted.append(msg)

            self.last_message_time[msg] = now
            self.last_family_time[family] = now
            self.last_global_time = now

            if len(accepted) >= self.max_messages_per_cycle:
                break

        return accepted