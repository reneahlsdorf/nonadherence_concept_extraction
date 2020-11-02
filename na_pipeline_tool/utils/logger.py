"""This module manages all loggers in this pipeline tool"""

"""
Medical lexicon NLP extraction pipeline

File contains: Helper methods for using the global logger

-- (c) Rene Ahlsdorf 2019 - Team  D in the HST 953 class
"""

from . import config

import logging
import sys, os

LOGGER_ENABLED = False
LOG_DEBUG = False
_GLOBAL_LOGGER = None

def init_logger():
    """Inits all loggers in this module (do it only once!)
    """

    global LOGGER_ENABLED, LOG_DEBUG, _GLOBAL_LOGGER
    if not config.get('logger_enabled', True):
        return

    LOGGER_ENABLED = True
    if config.get('debugging', False):
        LOG_DEBUG = True

    # Create a custom logger
    logger = logging.getLogger('NLP Extraction Tool')
    # Create handlers
    c_handler = logging.StreamHandler()

    if not LOG_DEBUG:
        logger.setLevel(level=logging.INFO)
        c_handler.setLevel(logging.INFO)
    else:
        logger.setLevel(level=logging.DEBUG)
        c_handler.setLevel(logging.DEBUG)

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    _GLOBAL_LOGGER = logger

    if not config.get('log_file', None) is None:
        filename = config.get('log_file', None)
        if os.path.isdir(filename):
            f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            f_handler = logging.FileHandler(config.get('log_file', None))
            if not LOG_DEBUG:
                f_handler.setLevel(logging.INFO)
            else:
                f_handler.setLevel(logging.DEBUG)
            f_handler.setFormatter(f_format)
            logger.addHandler(f_handler)
        else:
            print('ERR: Invalid log file: {}'.format(filename))

def log_warn(text):
    """Log warn text in global logger.

    Args:
        text (str): Warn message.
    """
    if LOGGER_ENABLED:
        _GLOBAL_LOGGER.warning(text)

def log_error(text, exitcode=1):
    """Log error text in global logger.

    Args:
        text (str): Error message.
        exitcode (int): The exit code to end the tool with.
    """
    if LOGGER_ENABLED:
        _GLOBAL_LOGGER.error(text)
        sys.exit(exitcode)

def log_info(text):
    """Log info text in global logger.

    Args:
        text (str): Info message.
    """
    if LOGGER_ENABLED:
        _GLOBAL_LOGGER.info(text)

def log_debug(text):
    """Log debug text in global logger. Only active in DEBUG mode (see config file)

    Args:
        text (str): Info message.
    """
    if LOG_DEBUG:
        _GLOBAL_LOGGER.debug(text)
