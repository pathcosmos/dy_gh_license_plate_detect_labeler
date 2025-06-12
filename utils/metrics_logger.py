import json
import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

class MetricsLogger:
    """Enhanced metrics logging for model performance tracking"""
    
    def __init__(self, log_dir: str = "logs/metrics"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize log files
        self.training_log = self.log_dir / "training_metrics.csv"
        self.inference_log = self.log_dir / "inference_metrics.json"
        
        self._init_training_csv()
    
    def _init_training_csv(self):
        """Initialize training metrics CSV file with headers"""
        if not self.training_log.exists():
            headers = ['timestamp', 'epoch', 'phase', 'loss', 'accuracy', 
                      'precision', 'recall', 'f1_score', 'learning_rate']
            with open(self.training_log, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
    
    def log_training_metrics(self, epoch: int, phase: str, metrics: Dict[str, float]):
        """Log training/validation metrics to CSV"""
        timestamp = datetime.now().isoformat()
        
        row_data = [
            timestamp,
            epoch,
            phase,
            metrics.get('loss', 0.0),
            metrics.get('accuracy', 0.0),
            metrics.get('precision', 0.0),
            metrics.get('recall', 0.0),
            metrics.get('f1_score', 0.0),
            metrics.get('learning_rate', 0.0)
        ]
        
        with open(self.training_log, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)
    
    def log_inference_metrics(self, image_path: str, results: Dict[str, Any]):
        """Log inference results to JSON"""
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            'timestamp': timestamp,
            'image_path': str(image_path),
            'processing_time': results.get('processing_time', 0.0),
            'detected_plates': results.get('detected_plates', []),
            'confidence_scores': results.get('confidence_scores', []),
            'avg_confidence': results.get('avg_confidence', 0.0),
            'model_version': results.get('model_version', 'unknown')
        }
        
        # Append to JSON lines file
        with open(self.inference_log, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def log_system_metrics(self, gpu_usage: float = None, memory_usage: float = None):
        """Log system resource usage"""
        system_log = self.log_dir / "system_metrics.json"
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'gpu_usage_percent': gpu_usage,
            'memory_usage_mb': memory_usage
        }
        
        with open(system_log, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
