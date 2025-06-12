import logging
import os
from datetime import datetime
from pathlib import Path

def setup_logger(name: str, log_dir: str = "logs", level: int = logging.INFO):
    """
    Enhanced logger setup with file and console handlers
    
    Args:
        name: Logger name
        log_dir: Directory to save log files
        level: Logging level
    
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    Path(log_dir).mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler - detailed logs
    log_filename = f"{log_dir}/license_plate_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler - simplified logs
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def log_model_metrics(logger, phase: str, metrics: dict, epoch: int = None):
    """
    Log model training/validation metrics
    
    Args:
        logger: Logger instance
        phase: Training phase (train/val/test)
        metrics: Dictionary of metrics
        epoch: Current epoch number
    """
    epoch_str = f"Epoch {epoch} - " if epoch is not None else ""
    metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    logger.info(f"{epoch_str}{phase.upper()} - {metrics_str}")

def log_data_info(logger, data_type: str, count: int, details: str = ""):
    """
    Log data loading and preprocessing information
    
    Args:
        logger: Logger instance
        data_type: Type of data (train/val/test)
        count: Number of samples
        details: Additional details
    """
    detail_str = f" - {details}" if details else ""
    logger.info(f"Data loaded - {data_type}: {count} samples{detail_str}")

def log_inference_results(logger, image_path: str, predictions: list, confidence: float):
    """
    Log inference results for license plate detection
    
    Args:
        logger: Logger instance
        image_path: Path to processed image
        predictions: List of detected license plates
        confidence: Average confidence score
    """
    logger.info(f"Inference - Image: {os.path.basename(image_path)} | "
                f"Detected plates: {len(predictions)} | "
                f"Avg confidence: {confidence:.3f}")
    
    for i, pred in enumerate(predictions):
        logger.debug(f"  Plate {i+1}: {pred}")
