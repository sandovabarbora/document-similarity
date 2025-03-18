"""
Logging utility for the Document Relevance Classification System.
Provides a comprehensive logging setup with file and console handlers.
"""

import os
import sys
import time
import logging
import logging.handlers
import traceback
from typing import Optional, Callable, Any
from functools import wraps
import inspect


class CustomFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    grey = "\x1b[38;20m"
    blue = "\x1b[34;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    
    FORMATS = {
        logging.DEBUG: grey + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.INFO: blue + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.WARNING: yellow + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.ERROR: red + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.CRITICAL: bold_red + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


def setup_logger(
    name: str, 
    log_dir: str = "logs", 
    log_level: int = logging.INFO, 
    enable_console: bool = True, 
    log_filename: Optional[str] = None,
    module_name: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Name for the logger
        log_dir: Directory to store log files
        log_level: Logging level (e.g., logging.INFO)
        enable_console: Whether to enable console logging
        log_filename: Name of the log file (default: <name>.log)
        module_name: Module name for context in log messages
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False
    
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # File handler with rotation
    if log_filename is None:
        log_filename = f"{name.replace('.', '_')}.log"
    
    log_file_path = os.path.join(log_dir, log_filename)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file_path, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setLevel(log_level)
    
    # Include module name if provided
    format_str = "%(asctime)s - %(name)s - %(levelname)s"
    if module_name:
        format_str += f" - [{module_name}] - [%(filename)s:%(lineno)d]"
    format_str += " - %(message)s"
    
    file_formatter = logging.Formatter(
        format_str,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler with colors (optional)
    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(CustomFormatter())
        logger.addHandler(console_handler)
    
    # Log system info at startup
    logger.info(f"Logging initialized: {log_file_path}")
    logger.info(f"Log level: {logging.getLevelName(log_level)}")
    
    return logger


def get_logger(
    module_name: str, 
    base_name: str = "document_relevance",
    config: Optional[dict] = None
) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Args:
        module_name: Name of the module (used in logger name)
        base_name: Base name for the logger
        config: Optional configuration dictionary with log_level
        
    Returns:
        Logger instance
    """
    logger_name = f"{base_name}.{module_name}"
    logger = logging.getLogger(logger_name)
    
    # Set log level if specified in config
    if config and 'log_level' in config:
        logger.setLevel(config['log_level'])
        
    return logger


def log_execution_time(logger=None, level=logging.INFO):
    """
    Decorator to log the execution time of a function.
    
    Args:
        logger: Logger to use (if None, will get a logger based on the function's module)
        level: Log level to use for the timing message
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger if not provided
            nonlocal logger
            if logger is None:
                module = inspect.getmodule(func)
                module_name = module.__name__ if module else func.__module__
                logger = get_logger(module_name.split('.')[-1])
            
            # Log function call
            logger.debug(f"Calling {func.__name__}")
            
            # Time the function execution
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.log(level, f"{func.__name__} executed in {elapsed:.4f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"{func.__name__} failed after {elapsed:.4f}s: {str(e)}")
                raise
        
        return wrapper
    
    return decorator


def log_exceptions(logger=None, reraise=True, level=logging.ERROR):
    """
    Decorator to log exceptions raised by a function.
    
    Args:
        logger: Logger to use (if None, will get a logger based on the function's module)
        reraise: Whether to re-raise the exception after logging
        level: Log level to use for the exception message
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger if not provided
            nonlocal logger
            if logger is None:
                module = inspect.getmodule(func)
                module_name = module.__name__ if module else func.__module__
                logger = get_logger(module_name.split('.')[-1])
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Capture traceback
                tb = traceback.format_exc()
                
                # Log the exception with traceback
                logger.log(level, f"Exception in {func.__name__}: {str(e)}\n{tb}")
                
                # Re-raise if requested
                if reraise:
                    raise
                
                # Return None if not re-raising
                return None
        
        return wrapper
    
    return decorator