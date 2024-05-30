import sys
import logging

LOGGING_FILE = None


class Logger:
    logger = None
    
    @staticmethod
    def get_logger(name: str, log_path:str=None) -> logging.Logger:
        if Logger.logger is None:
            Logger.logger = get_logger(name, log_path)
        return Logger.logger
    
    @staticmethod
    def info(msg: object):
        Logger.logger.info(msg)
    
    @staticmethod
    def error(msg: object):
        Logger.logger.error(msg)
    
    @staticmethod
    def warning(msg: object):
        Logger.logger.warning(msg)

def get_logger(name: str, log_path:str=None) -> logging.Logger:
    global LOGGING_FILE
    formatter = logging.Formatter(
        # fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S"
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    if log_path is not None:
        logger = add_file_handler(logger, log_path)
        LOGGING_FILE = log_path
    elif LOGGING_FILE is not None:
        logger = add_file_handler(logger, LOGGING_FILE)

    return logger

def add_file_handler(logger, file_path):
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S"
    )
    handler_file = logging.FileHandler(file_path,
                                       encoding = 'utf-8',
                                       mode = 'a')
    handler_file.setLevel(logging.INFO)
    handler_file.setFormatter(formatter)
    logger.addHandler(handler_file)
    
    return logger