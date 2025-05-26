import logging
from logging import getLogger

logging.basicConfig(
    level=logging.INFO,
    format="  %(asctime)s --[%(levelname)s] - %(name)s - %(message)s",
)


def get_logger(name: str):
    return getLogger(name)
