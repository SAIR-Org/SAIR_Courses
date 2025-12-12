"""
Logging setup for the pipeline
"""
import logging
import sys
from pathlib import Path
from typing import Optional


class PipelineLogger:
    """Configure logging for the pipeline"""
    
    @staticmethod
    def setup_logger(
        name: str = "pipeline",
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        console: bool = True
    ) -> logging.Logger:
        """
        Setup and configure logger
        
        Args:
            name: Logger name
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Path to log file (optional)
            log_format: Log message format
            console: Whether to log to console
            
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(name)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Set log level
        level = getattr(logging, log_level.upper())
        logger.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter(log_format)
        
        # Console handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            # Create directory if it doesn't exist
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
        
        return logger
    
    @staticmethod
    def get_logger(name: str = "pipeline") -> logging.Logger:
        """Get existing logger or create new one"""
        return logging.getLogger(name)