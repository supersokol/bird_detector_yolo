import logging

def setup_logger(log_file: str = "logs/bird_detector_log.log") -> logging.Logger:
    """
    Set up a logger to log messages to both console and a log file.

    Args:
        log_file (str): Path to the log file.
       
    """
    logging.basicConfig(filename = log_file,
                        filemode = 'a',
                        format = '%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt = '%H:%M:%S',
                        level = logging.DEBUG)

    log = logging.getLogger(__name__)
    return log
