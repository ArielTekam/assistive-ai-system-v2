from ultralytics import YOLO


class ByteTrackDetector:
    """
    Детектор + трекер на основе YOLO и ByteTrack.

    Назначение:
    - выполняет детекцию объектов;
    - присваивает стабильные ID объектам;
    - использует встроенный ByteTrack из Ultralytics;
    - возвращает данные в формате, совместимом с остальным pipeline.
    """

    def __init__(
        self,
        model_path="yolo11n.pt",
        conf_threshold=0.25,
        img_size=320,
        tracker_config="bytetrack.yaml"
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.img_size = img_size
        self.tracker_config = tracker_config

        self.model = YOLO(self.model_path)

        print("[BYTETRACK] Модель загружена:", self.model_path)
        print("[BYTETRACK] Tracker:", self.tracker_config)
        print("[BYTETRACK] Confidence:", self.conf_threshold)
        print("[BYTETRACK] Image size:", self.img_size)

    def track(self, frame):
        """
        Выполняет детекцию и трекинг одного кадра.
        """
        results = self.model.track(
            frame,
            imgsz=self.img_size,
            conf=self.conf_threshold,
            tracker=self.tracker_config,
            persist=True,
            verbose=False
        )[0]

        tracked_objects = []

        if results.boxes is None:
            return tracked_objects

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            track_id = None

            if box.id is not None:
                track_id = int(box.id[0])

            tracked_objects.append({
                "track_id": track_id,
                "label": label,
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2]
            })

        return tracked_objects
