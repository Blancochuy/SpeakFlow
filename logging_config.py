import logging
import sys

def setup_logging(level=logging.INFO, log_file: str = None) -> None:
    """
    Configure logging output for the project.
    
    Parameters:
      level    -- Minimum level of messages to record.
      log_file -- If specified, messages will also be logged to this file.
    """
    # Get root logger and set level
    logger = logging.getLogger()
    logger.setLevel(level)

    # Define message formatter
    formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s')

    # Configure console output handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional: Configure handler for log file
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)