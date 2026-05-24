import cv2
import argparse
import time
from pathlib import Path


def record_video(scenario, run, duration, camera_id=0, width=640, height=480, fps=20):
    scenario_dir = Path("Experiments/videos") / scenario
    scenario_dir.mkdir(parents=True, exist_ok=True)

    output_path = scenario_dir / f"{scenario}_run{run:02d}.mp4"

    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print("Ошибка: камера не открыта")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    print("===================================")
    print("Запись экспериментального видео")
    print(f"Сценарий: {scenario}")
    print(f"Run: {run}")
    print(f"Длительность: {duration} секунд")
    print(f"Файл: {output_path}")
    print("===================================")

    start_time = time.time()
    frame_count = 0

    while time.time() - start_time < duration:
        ret, frame = cap.read()

        if not ret:
            print("Ошибка чтения кадра")
            break

        frame = cv2.resize(frame, (width, height))
        writer.write(frame)
        frame_count += 1

    cap.release()
    writer.release()

    print("===================================")
    print("Запись завершена")
    print(f"Кадров записано: {frame_count}")
    print(f"Сохранено: {output_path}")
    print("===================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--scenario", required=True, help="Название сценария")
    parser.add_argument("--run", type=int, required=True, help="Номер запуска")
    parser.add_argument("--duration", type=int, default=45, help="Длительность записи")
    parser.add_argument("--camera", type=int, default=0, help="ID камеры")

    args = parser.parse_args()

    record_video(
        scenario=args.scenario,
        run=args.run,
        duration=args.duration,
        camera_id=args.camera
    )
