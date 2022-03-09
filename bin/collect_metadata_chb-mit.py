#!/usr/bin/python3.7
"""
1.- Get files having seizures from RECORDS-WITH-SEIZURES
2.- Read summary file
3.- Read sampling frequency
4.- Read channels in EDF files
5.- Read start/end time of each seizure
6.- Save to metadata.json
"""
import sys
import json
import logging
from collections import defaultdict
sys.path.append("../")
from utils import settings
from utils.file_managers import write_csv
from utils.file_parsers import get_sampl_frequency, get_channels_and_seizures, unwind_meta_json


record_seizure_file = settings["chb-mit"]["seizure_records"]
metadata_folder = settings["chb-mit"]["metadata"]
metadata_file_json = settings["chb-mit"]["metadata_file_json"]
metadata_file_csv = settings["chb-mit"]["metadata_file_csv"]
meta_file_pattern = metadata_folder + "/{}-summary.txt"
full_metadata = defaultdict(dict)


def main():

    with open(record_seizure_file, "r") as f:
        interest_files = {a.replace("\n", "").split("/")[0]
                          for a in f.readlines() if a != "\n"}

    for file in interest_files:
        with open(meta_file_pattern.format(file), "r") as f:
            content = f.readlines()

        full_metadata[file]["fs"] = get_sampl_frequency(content)
        full_metadata[file]["files"] = get_channels_and_seizures(content)

        logging.info("Collected data of " + file)

    with open(metadata_file_json, "w") as f:
        json.dump((dict(full_metadata)), f)

    unwinded_data = unwind_meta_json(full_metadata)
    write_csv(metadata_file_csv, unwinded_data,
              sort_keys=["patient", "filename"],
              columns=["patient", "filename", "fs", "channels",
                       "number_of_seizure", "start_time", "end_time"])


if __name__ == "__main__":
    main()