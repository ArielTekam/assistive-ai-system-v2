from ultralytics import YOLO


class ObjectDetector:
    """
    Модуль визуального восприятия.

    Назначение:
    - загружает модель YOLO;
    - выполняет обнаружение объектов на кадре;
    - возвращает список найденных объектов с классом, уверенностью и координатами рамки.
    """

    def __init__(self, model_path="yolo11n.pt", conf_threshold=0.25, img_size=320):
        """
        Инициализация детектора объектов.

        :param model_path: путь к модели YOLO
        :param conf_threshold: минимальный порог уверенности
        :param img_size: размер входного изображения для модели
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.img_size = img_size
        self.model = YOLO(model_path)

        print(f"[ДЕТЕКТОР] Модель загружена: {model_path}")
        print(f"[ДЕТЕКТОР] Порог уверенности: {conf_threshold}")
        print(f"[ДЕТЕКТОР] Размер входного изображения: {img_size}")

    def detect(self, frame):
        """
        Выполняет обнаружение объектов на одном кадре.

        :param frame: кадр с камеры
        :return: список обнаруженных объектов
        """
        results = self.model.predict(
            frame,
            imgsz=self.img_size,
            conf=self.conf_threshold,
            verbose=False
        )[0]

        detections = []

        if results.boxes is None:
            return detections

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "label": label,
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2]
            })

        return detections