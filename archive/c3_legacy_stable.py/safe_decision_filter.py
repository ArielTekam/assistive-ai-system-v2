import time


class SafeDecisionFilter:
    """
    Filtre final anti-spam avant audio.

    Objectif :
    - bloquer les micro-mouvements gauche/centre/droite ;
    - éviter les répétitions rapides ;
    - limiter la surcharge audio ;
    - ne laisser passer que les changements vraiment utiles.
    """

    def __init__(
        self,
        global_cooldown=5.0,
        same_message_cooldown=12.0,
        max_messages_per_cycle=1
    ):
        self.global_cooldown = global_cooldown
        self.same_message_cooldown = same_message_cooldown
        self.max_messages_per_cycle = max_messages_per_cycle

        self.last_global_time = 0.0
        self.last_message_time = {}

    def filter(self, messages):
        now = time.time()
        accepted = []

        if not messages:
            return accepted

        if now - self.last_global_time < self.global_cooldown:
            return accepted

        for msg in messages:
            if not msg:
                continue

            msg = msg.strip()

            last_time = self.last_message_time.get(msg, 0.0)

            if now - last_time < self.same_message_cooldown:
                continue

            accepted.append(msg)
            self.last_message_time[msg] = now
            self.last_global_time = now

            if len(accepted) >= self.max_messages_per_cycle:
                break

        return accepted
