#!/usr/bin/python3.7
import os
import sys
import yaml
import logging.config
import pathlib

current_path = str(pathlib.Path(__file__).parent.parent.resolve())

logging.config.fileConfig(current_path + "/etc/logging.ini")


def catch_exception(exc_type, exc_value, exc_traceback):
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value,
                                                  exc_traceback))


sys.excepthook = catch_exception

with open(current_path + "/etc/settings.yaml", "r") as stream:
    try:
        settings = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
