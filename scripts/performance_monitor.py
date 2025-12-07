import time
import psutil
import torch
import logging

class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.peak_memory = 0
        self.logger = logging.getLogger(__name__)
        
    def start_monitoring(self):
        """Start performance monitoring"""
        self.start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.logger.info("Performance monitoring started")
    
    def log_performance(self, stage_name):
        """Log current performance metrics"""
        if self.start_time is None:
            self.start_monitoring()
            
        elapsed = time.time() - self.start_time
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        self.logger.info(f"{stage_name} - Time: {elapsed:.2f}s, "
                        f"CPU: {cpu_percent:.1f}%, "
                        f"RAM: {memory_percent:.1f}%, "
                        f"GPU Memory: {gpu_memory:.2f}GB")

# Global monitor instance
monitor = PerformanceMonitor()