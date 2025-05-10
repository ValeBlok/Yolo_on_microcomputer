import os
import time
import cv2
from ultralytics import YOLO
from datetime import datetime
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from system_monitor import start_monitoring, stop_monitoring

SYSTEM_METRICS_FILE = "share/default_metrics.csv"

def main():
    print("Снятие дефолтных метрик...")
    queue, monitor_process = start_monitoring(SYSTEM_METRICS_FILE)

    # Остановка мониторинга и вывод статистики
    print("\nОстановка мониторинга системы...")
    from system_monitor import SystemMonitor
    monitor = SystemMonitor()
    stop_monitoring(queue, monitor_process)
    
    # Чтение и анализ собранных метрик
    import pandas as pd
    try:
        df = pd.read_csv(SYSTEM_METRICS_FILE)
        if not df.empty:
            cpu_avg = df['cpu_usage'].mean()
            ram_avg = df['ram_usage'].mean()
            temp_avg = df[df['cpu_temp'] >= 0]['cpu_temp'].mean()
            
            print("\nСредние значения за период работы:")
            print(f"CPU usage: {cpu_avg:.2f}%")
            print(f"RAM usage: {ram_avg:.2f} MB")
            print(f"CPU temp: {temp_avg:.1f}°C" if temp_avg >= 0 else "CPU temp: N/A")
    except Exception as e:
        print(f"Ошибка при анализе метрик: {e}")

if __name__ == "__main__":
    main()