import logging
import os
from datetime import datetime


def create_logger(log_name, log_dir="logs"):
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    
    today = datetime.now().strftime("%Y-%m-%d")
    
    logger = logging.getLogger(log_name)
    file_handler = logging.FileHandler(f"{log_dir}/{today}.log")
    console_handler = logging.StreamHandler()
    
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_formatter = logging.Formatter("%(message)s")
    
    logger.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
