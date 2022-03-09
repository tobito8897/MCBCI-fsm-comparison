#!/usr/bin/python3.7
"""
1.- Get list of metadata files Seizures-list-{patient}.txt
2.- Read summary file
3.- Read sampling frequency
4.- Read channels in EDF files
5.- Read start/end time of each seizure
6.- Save to metadata.json
"""
import os
import re
import mne
import sys
import json
from collections import defaultdict
sys.path.append("../")
from utils import settings
from utils.file_parsers import get_sampl_frequency,\
                               get_seizures, unwind_meta_json
from utils.file_managers import write_csv


settings = settings["siena"]
database_folder = settings["database"]
metadata_folder = settings["metadata"]
metadata_file_json = settings["metadata_file_json"]
metadata_file_csv = settings["metadata_file_csv"]
full_metadata = defaultdict(dict)


def main():

    metadata_files, edf_files = [], []
    for path, _, files in os.walk(database_folder):
        for name in files:
            if ".edf" in name:
                edf_files.append(os.path.join(path, name))

    for path, _, files in os.walk(metadata_folder):
        for name in files:
            if "Seizures-list" in name:
                metadata_files.append(os.path.join(path, name))

    for file in metadata_files:
        patient = re.search("(?<=-)PN[0-9]{2}(?=\.)", file).group(0)
        patient_files = [x for x in edf_files if patient in x]

        with open(file.format(file), "r") as f:
            content = f.readlines()
        full_metadata[patient]["fs"] = get_sampl_frequency(content)
        full_metadata[patient]["files"] = {}

        for p_file in patient_files:
            edf_file = mne.io.read_raw_edf(p_file)
            channels = [x.split()[1] if len(x.split()) == 2
                        else x for x in edf_file.ch_names]

            edf_file = p_file.split("/")[-1]
            seizures = get_seizures(content, edf_file)
            single_meta = {"channels": {k: v for k, v in enumerate(channels)},
                           "number_of_seizure": len(seizures),
                           "seizures": seizures}
            full_metadata[patient]["files"][edf_file] = single_meta

    with open(metadata_file_json, "w") as f:
        json.dump((dict(full_metadata)), f)

    unwinded_data = unwind_meta_json(full_metadata)
    write_csv(metadata_file_csv, unwinded_data,
              sort_keys=["patient", "filename"],
              columns=["patient", "filename", "fs", "channels",
                       "number_of_seizure", "start_time", "end_time"])


if __name__ == "__main__":
    main()
