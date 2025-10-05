import logging
import os

def get_logger(name, level=logging.INFO, to_file=None):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:  # avoid duplicate handlers
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # file handler if path given
        if to_file:
            log_dir = os.path.dirname(to_file)
            if log_dir:  # make sure dir exists
                os.makedirs(log_dir, exist_ok=True)

            fh = logging.FileHandler(to_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger
