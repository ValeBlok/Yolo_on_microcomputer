import os
import time
import cv2
from ultralytics import YOLO
import psutil
from gpiozero import CPUTemperature
from datetime import datetime
from threading import Thread
from ultralytics import YOLO
import os

MODEL = "models/yolov8n.pt"
VIDEO_PATH = "video/street_short.mp4"
OUTPUT_FILE = f"share/results_r5/video_results_{MODEL.split('.')[0]}.txt"

class VideoProcessor:
    def __init__(self):
        self.model = YOLO(MODEL)
        self.frame_count = 0
        self.start_time = 0
        self.system_monitor = SystemMonitor()

    def process_video(self):
        """Основной метод обработки видео"""
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print("❌ Ошибка открытия видеофайла!")
            return

        # Параметры видео
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n🔍 Начало обработки видео (FPS: {fps:.1f}, кадров: {total_frames})")

        # Запуск мониторинга системы
        self.system_monitor.start()
        self.start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.frame_count += 1
            self.process_frame(frame)

            # Вывод прогресса каждые 10 кадров
            if self.frame_count % 10 == 0:
                self.print_progress(total_frames)

        # Финализация
        cap.release()
        self.print_final_stats(total_frames)

    def process_frame(self, frame):
        """Обрабатывает один кадр"""
        results = self.model(frame,
                            imgsz=640,
                            verbose=False)
        
        # Детекция объектов (для статистики)
        detected = []
        for box in results[0].boxes:
            class_id = int(box.cls.item())
            detected.append(self.model.names[class_id])

        # Вывод информации о кадре (опционально)
        if len(detected) > 0:
            print(f"Кадр {self.frame_count}: обнаружено {', '.join(detected)}")

    def print_progress(self, total_frames):
        """Выводит прогресс обработки"""
        elapsed = time.time() - self.start_time
        current_fps = self.frame_count / elapsed
        remaining = (total_frames - self.frame_count) / current_fps
        
        print(
            f"\r📊 Прогресс: {self.frame_count}/{total_frames} | "
            f"FPS: {current_fps:.1f} | "
            f"Время: {elapsed:.1f}с | "
            f"Осталось: {remaining:.1f}с",
            end="", flush=True
        )

    def print_final_stats(self, total_frames):
        """Записывает итоговую статистику в файл stats.txt"""
        total_time = time.time() - self.start_time
        sys_metrics = self.system_monitor.stop()
        
        stats_content = f"""✅ Обработка завершена!
            • Кадров: {self.frame_count}/{total_frames}
            • Время: {total_time:.2f} сек
            • Средний FPS: {self.frame_count/total_time:.2f}
            • CPU: {sys_metrics['cpu_avg']:.1f}% (пик: {sys_metrics['cpu_max']:.1f}%)
            • RAM: {sys_metrics['ram_avg']:.2f} MB (пик: {sys_metrics['ram_max']:.2f} MB)"""
        
        if sys_metrics['temp_avg'] >= 0:
            stats_content += f"\n• Температура CPU: {sys_metrics['temp_avg']:.1f}°C"
        
        # Запись в файл
        with open(OUTPUT_FILE, "w") as f:
            f.write(stats_content)
        
        # Дополнительно выводим в консоль путь к файлу
        print(f"\nСтатистика сохранена в: {OUTPUT_FILE}")

class SystemMonitor:
    def __init__(self):
        self.measurements = {
            'cpu': [], 
            'ram': [],
            'temp': []
        }
        self._running = False

    def start(self):
        self._running = True
        Thread(target=self._monitor).start()

    def _monitor(self):
        while self._running:
            self.measurements['cpu'].append(psutil.cpu_percent(interval=0.5))
            self.measurements['ram'].append(psutil.virtual_memory().used / (1024 ** 2))
            try:
                temp = CPUTemperature().temperature
                self.measurements['temp'].append(float(temp))
            except:
                pass

    def stop(self):
        self._running = False
        time.sleep(0.6)  # Даем последнему измерению завершиться
        
        return {
            'cpu_avg': sum(self.measurements['cpu']) / len(self.measurements['cpu']),
            'cpu_max': max(self.measurements['cpu']),
            'ram_avg': sum(self.measurements['ram']) / len(self.measurements['ram']),
            'ram_max': max(self.measurements['ram']),
            'temp_avg': sum(self.measurements['temp'])/len(self.measurements['temp']) if self.measurements['temp'] else -1
        }

if __name__ == "__main__":
    processor = VideoProcessor()
    processor.process_video()
