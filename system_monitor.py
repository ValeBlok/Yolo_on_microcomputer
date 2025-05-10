import psutil
from gpiozero import CPUTemperature
import time
from datetime import datetime
import os
import csv
from multiprocessing import Process, Queue
import signal
import sys

class SystemMonitor:
    def __init__(self, output_file="share/system_metrics.csv", interval=0.5):
        self.output_file = output_file
        self.interval = interval
        self.metrics = []
        self._prepare_output_file()
        
    def _prepare_output_file(self):
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'cpu_usage', 'ram_usage', 'cpu_temp'])
    
    def collect_metrics(self):
        try:
            cpu_temp = float(CPUTemperature().temperature)
        except:
            cpu_temp = -1
            
        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
            'cpu_usage': psutil.cpu_percent(interval=None),
            'ram_usage': psutil.virtual_memory().used / (1024 ** 2),
            'cpu_temp': cpu_temp
        }
    
    def start_monitoring(self, queue):
        """Метод для запуска в отдельном процессе"""
        try:
            while True:
                metrics = self.collect_metrics()
                self.metrics.append(metrics)
                
                # Запись в файл
                with open(self.output_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        metrics['timestamp'],
                        metrics['cpu_usage'],
                        metrics['ram_usage'],
                        metrics['cpu_temp']
                    ])
                
                # Проверка сигнала остановки
                if not queue.empty():
                    if queue.get() == "STOP":
                        break
                
                time.sleep(self.interval)
        except KeyboardInterrupt:
            pass
    
    def get_average_metrics(self):
        if not self.metrics:
            return None
            
        cpu_avg = sum(m['cpu_usage'] for m in self.metrics) / len(self.metrics)
        ram_avg = sum(m['ram_usage'] for m in self.metrics) / len(self.metrics)
        valid_temps = [m['cpu_temp'] for m in self.metrics if m['cpu_temp'] >= 0]
        temp_avg = sum(valid_temps) / len(valid_temps) if valid_temps else -1
        
        return {
            'cpu_avg': cpu_avg,
            'ram_avg': ram_avg,
            'temp_avg': temp_avg
        }

def monitor_process(queue, output_file):
    monitor = SystemMonitor(output_file)
    monitor.start_monitoring(queue)

def start_monitoring(output_file):
    """Запускает мониторинг в отдельном процессе"""
    queue = Queue()
    process = Process(target=monitor_process, args=(queue, output_file))
    process.start()
    return queue, process

def stop_monitoring(queue, process):
    """Останавливает мониторинг"""
    queue.put("STOP")
    process.join(timeout=1)
    if process.is_alive():
        process.terminate()