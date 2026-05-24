from collections import defaultdict


class TrackerMetrics:

    def __init__(self):

        self.unique_ids = set()

        self.last_seen = {}

        self.id_switches = 0

        self.track_frames = defaultdict(int)

        self.total_frames = 0

    def update(self, tracked_objects):

        self.total_frames += 1

        current_ids = set()

        for obj in tracked_objects:

            track_id = int(obj["track_id"])

            current_ids.add(track_id)

            self.unique_ids.add(track_id)

            self.track_frames[track_id] += 1

        disappeared = set(self.last_seen.keys()) - current_ids

        for lost_id in disappeared:
            self.last_seen.pop(lost_id, None)

        for current_id in current_ids:

            if current_id in self.last_seen:
                pass

            self.last_seen[current_id] = self.total_frames

    def summary(self):

        total_tracks = len(self.unique_ids)

        avg_duration = 0

        if total_tracks > 0:
            avg_duration = (
                sum(self.track_frames.values()) / total_tracks
            )

        stable_tracks = sum(
            1
            for frames in self.track_frames.values()
            if frames >= 30
        )

        stable_ratio = 0

        if total_tracks > 0:
            stable_ratio = stable_tracks / total_tracks

        return {
            "unique_ids": total_tracks,
            "avg_track_duration": round(avg_duration, 2),
            "stable_tracks_ratio": round(stable_ratio, 3),
            "id_switches": self.id_switches,
        }
