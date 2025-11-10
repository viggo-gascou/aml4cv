"""Aml4cv project package."""

import logging

from dotenv import load_dotenv
from termcolor import colored

from .constants import BASE_DIR, RESULTS_DIR
from .log_utils import log, set_logging_level

fmt = (
    colored("[%(asctime)s ⋅ %(levelname)s]", "light_blue")
    + " ⋅ "
    + colored("%(message)s", "green")
)
logging.basicConfig(level=logging.WARNING, format=fmt, datefmt="%Y-%m-%d %H:%M:%S")

load_dotenv()

__all__ = ["log", "set_logging_level", "BASE_DIR", "RESULTS_DIR"]
