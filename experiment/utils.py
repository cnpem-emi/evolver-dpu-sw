import logging
import logging.handlers


def get_file_handler(file: str):
    """
    Create a file handler for logging.
    """
    # logger.handlers.clear()
    file_handler = logging.handlers.RotatingFileHandler(
        file, maxBytes=10000000, backupCount=10
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s — %(name)s — %(levelname)s — \
                %(funcName)s:%(lineno)d — %(message)s"
        )
    )
    return file_handler


def get_logger(file_handler):
    """
    Create a logger for logging.
    """
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    log.addHandler(file_handler)
    return log