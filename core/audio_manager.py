import time
import queue
import threading
import subprocess


class AudioManager:
    """
    Асинхронный аудио-менеджер.

    Назначение:
    - получает финальные сообщения от Decision Engine;
    - помещает их в очередь;
    - озвучивает сообщения через espeak-ng;
    - предотвращает наложение сообщений;
    - не блокирует основной поток камеры и детекции.
    """

    def __init__(
        self,
        speech_duration=0.2,
        max_queue_size=3,
        min_repeat_interval=2.0,
        simulation=False,
        voice="en",
        speed=145
    ):
        """
        Инициализация аудио-менеджера.

        :param speech_duration: пауза после озвучивания сообщения
        :param max_queue_size: максимальный размер очереди сообщений
        :param min_repeat_interval: минимальный интервал между одинаковыми сообщениями
        :param simulation: если True, сообщение только выводится в терминал
        :param voice: язык/голос espeak-ng
        :param speed: скорость речи espeak-ng
        """
        self.speech_duration = speech_duration
        self.min_repeat_interval = min_repeat_interval
        self.simulation = simulation
        self.voice = voice
        self.speed = speed

        self.queue = queue.Queue(maxsize=max_queue_size)

        self.running = True
        self.is_speaking = False

        self.last_message = None
        self.last_message_time = 0.0

        self.spoken_count = 0
        self.dropped_count = 0
        self.repeated_blocked_count = 0
        self.tts_error_count = 0

        self.worker = threading.Thread(
            target=self._worker_loop,
            daemon=True
        )
        self.worker.start()

        print("[АУДИО] AudioManager запущен")

    def _can_accept_message(self, message):
        """
        Проверяет, можно ли принять сообщение в очередь.
        """
        now = time.time()

        if not message:
            return False

        message = message.strip()

        if not message:
            return False

        if (
            self.last_message == message
            and (now - self.last_message_time) < self.min_repeat_interval
        ):
            self.repeated_blocked_count += 1
            return False

        return True

    def speak(self, message):
        """
        Добавляет сообщение в аудио-очередь.

        Если очередь переполнена, удаляется самое старое сообщение.
        Это позволяет сохранить более актуальные предупреждения.
        """
        if not self._can_accept_message(message):
            return False

        if self.queue.full():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
                self.dropped_count += 1
            except queue.Empty:
                pass

        try:
            self.queue.put_nowait(message)
            return True
        except queue.Full:
            self.dropped_count += 1
            return False

    def _speak_simulated(self, message):
        """
        Симуляция озвучивания.
        """
        print(f"[AUDIO_SIM] {message}")
        time.sleep(self.speech_duration)

    def _speak_real(self, message):
        """
        Реальное озвучивание через espeak-ng.
        """
        print(f"[AUDIO] {message}")

        try:
            subprocess.run(
                [
                    "espeak-ng",
                    "-v", self.voice,
                    "-s", str(self.speed),
                    message
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False
            )
        except Exception:
            self.tts_error_count += 1

        time.sleep(self.speech_duration)

    def _worker_loop(self):
        """
        Фоновый поток аудио.
        """
        while self.running:
            try:
                message = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            self.is_speaking = True

            try:
                if self.simulation:
                    self._speak_simulated(message)
                else:
                    self._speak_real(message)

                self.spoken_count += 1
                self.last_message = message
                self.last_message_time = time.time()

            finally:
                self.is_speaking = False
                self.queue.task_done()

        print("[АУДИО] AudioManager остановлен")

    def get_stats(self):
        """
        Возвращает статистику работы аудио-менеджера.
        """
        return {
            "spoken_count": self.spoken_count,
            "dropped_count": self.dropped_count,
            "repeated_blocked_count": self.repeated_blocked_count,
            "tts_error_count": self.tts_error_count,
            "queue_size": self.queue.qsize()
        }

    def stop(self):
        """
        Корректно останавливает аудио-поток.
        """
        self.running = False

        if self.worker.is_alive():
            self.worker.join(timeout=2.0)

        print("[АУДИО] Финальная статистика:", self.get_stats())