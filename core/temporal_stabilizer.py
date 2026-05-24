from collections import defaultdict, deque


class TemporalStabilizer:
    """
    Stabilise les objets trackés avant C2/C3.

    Objectif :
    - empêcher les changements rapides gauche/centre/droite ;
    - lisser la proximité ;
    - ignorer les objets trop instables ;
    - exiger une présence minimale avant annonce.
    """

    def __init__(
        self,
        min_seen_frames=8,
        direction_window=8,
        proximity_alpha=0.65
    ):
        self.min_seen_frames = min_seen_frames
        self.direction_window = direction_window
        self.proximity_alpha = proximity_alpha

        self.seen_count = defaultdict(int)
        self.direction_history = defaultdict(lambda: deque(maxlen=direction_window))
        self.smoothed_proximity = {}

    def update(self, objects):
        stable_objects = {}

        for track_id, obj in objects.items():
            self.seen_count[track_id] += 1

            self.direction_history[track_id].append(obj["direction"])

            if track_id not in self.smoothed_proximity:
                self.smoothed_proximity[track_id] = obj["proximity"]
            else:
                self.smoothed_proximity[track_id] = (
                    self.proximity_alpha * self.smoothed_proximity[track_id]
                    + (1 - self.proximity_alpha) * obj["proximity"]
                )

            if self.seen_count[track_id] < self.min_seen_frames:
                continue

            directions = list(self.direction_history[track_id])
            stable_direction = max(set(directions), key=directions.count)

            stable_obj = obj.copy()
            stable_obj["direction"] = stable_direction
            stable_obj["proximity"] = round(self.smoothed_proximity[track_id], 3)
            stable_obj["risk_score"] = stable_obj["proximity"]

            stable_objects[track_id] = stable_obj

        return stable_objects
