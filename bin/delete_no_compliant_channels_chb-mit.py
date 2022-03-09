#!/usr/bin/python3.7
"""
1.- Get a list of files not having channels required by channels.txt
2.- Delete that list from metadata.json
"""
import os
import copy
import sys
import logging
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir, ".."))
from utils import settings
from utils.file_managers import get_json_content, get_text_content, write_json, write_csv
from utils.file_parsers import unwind_meta_json


settings = settings["chb-mit"]
metadata_file = os.path.join(current_dir, settings["metadata_file_json"])
channels_file = os.path.join(current_dir, settings["channels"])
metadata_file_csv = os.path.join(current_dir, settings["metadata_file_csv"])


meta = get_json_content(metadata_file)
modify_meta = copy.deepcopy(meta)
channels = set(get_text_content(channels_file))

for patient, data in meta.items():
    for file, details in data["files"].items():
        _c_channels = set(details["channels"].values())
        if len(_c_channels.intersection(channels)) < len(channels):
            logging.info("Deleted " + file)
            del modify_meta[patient]["files"][file]

unwinded_modify_meta = unwind_meta_json(modify_meta)
write_json(metadata_file, modify_meta)
write_csv(metadata_file_csv, unwinded_modify_meta,
          sort_keys=["patient", "filename"],
          columns=["patient", "filename", "fs", "channels",
                   "number_of_seizure", "start_time", "end_time"])