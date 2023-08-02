"""
https://theaisummer.com/logging-debugging/
"""
import os
import logging.config
import yaml

# get the path of the current file
CONFIG_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(CONFIG_PATH, '../../config/logging.yaml')
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)
    logging.captureWarnings(True)

def get_logger(name: str):
    """Logs a message
    Args:
    name(str): name of logger
    """
    logger = logging.getLogger(name)
    return logger
