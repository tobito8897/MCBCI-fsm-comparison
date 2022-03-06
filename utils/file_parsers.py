#!/usr/bin/python3.7
__author__ = "Sergio E Sanchez Hdez"

import re
from datetime import datetime, timedelta
from collections import defaultdict

record_seizure_file = "../../HelperFiles/RECORDS-WITH-SEIZURES"
metadata_folder = "../../RawData/Metadata/"
meta_file_pattern = metadata_folder + "/{}-summary.txt"
full_metadata = defaultdict(dict)


def get_channels_groups(text_lines: list) -> list:
    channel_index = []
    groups = []
    text_lines.append("")

    for index, line in enumerate(text_lines):
        if "Channels " in line:
            channel_index.append(index)
    channel_index.append(-1)

    for index, _ in enumerate(channel_index[: -1]):
        groups.append(text_lines[channel_index[index]: channel_index[index+1]])

    return groups


def get_files_groups(text_lines: list) -> list:
    file_index = []
    groups = []
    text_lines.append("")

    for index, line in enumerate(text_lines):
        if "File Name" in line:
            file_index.append(index)
    file_index.append(-1)

    for index, _ in enumerate(file_index[: -1]):
        groups.append(text_lines[file_index[index]: file_index[index+1]])
    
    return groups


def get_channels(text_lines: list) -> dict:
    channels =  {}

    for line in text_lines:
        if not "Channel " in line:
            continue
        match = re.search("(\d{1,2}):\s(.+)", line)

        if match:
            groups = match.groups()
            channels[groups[0]] = groups[1]
    return channels


def get_file_meta(text_lines: list) -> tuple:
    clean_lines = []
    seizures = []
    file_meta = {}

    for line in text_lines:
        if "Name" in line or "Seizure" in line:
            clean_lines.append(line)

    file_name = re.search("(?<=:)\s.+", clean_lines[0]).group(0).strip()
    file_meta["number_of_seizure"] = int(re.search("(?<=:)\s.+", clean_lines[1]).group(0).strip())

    if not file_meta["number_of_seizure"]: 
        file_meta["seizures"] = []
        return file_name, file_meta

    for a in range(2, file_meta["number_of_seizure"]*2+2, 2):
        start_seizure = re.search("(?<=:)\s*[0-9]+", clean_lines[a]).group(0).strip()
        end_seizure = re.search("(?<=:)\s*[0-9]+", clean_lines[a+1]).group(0).strip()
        seizures.append((start_seizure, end_seizure))
    
    file_meta["seizures"] = seizures

    return file_name, file_meta


def get_channels_and_seizures(text_lines: list, just_seizure: bool=True) -> dict:
    channels_files_map = {}

    channels_groups = get_channels_groups(text_lines)

    for _c_group in channels_groups:
        channels = get_channels(_c_group)
        file_groups = get_files_groups(_c_group)

        for _f_group in file_groups:
            file_name, file_data = get_file_meta(_f_group)
            if file_data["seizures"]:
                channels_files_map[file_name] = {"channels": channels}
                channels_files_map[file_name].update(file_data)

    return channels_files_map


def get_sampl_frequency(text_lines: list) -> int:
    for line in text_lines:
        match = re.search("(?<=Rate:\s)\d{3}", line)
        if match:
            break

    return int(match.group(0))


def unwind_meta_json(patients_map: dict) -> dict:
    patients_data = []
    for patient, v1 in patients_map.items():
        single_patient_data = {}
        single_patient_data["patient"] = patient
        single_patient_data["fs"] = v1["fs"]

        for file_name, v2 in v1["files"].items():
            _inner_data = {}
            _inner_data["filename"] = file_name
            _inner_data["channels"] = list(v2["channels"].values())
            _inner_data["number_of_seizure"] = v2["number_of_seizure"]

            for seizure in v2["seizures"]:
                _seizure_span = {}
                _seizure_span["start_time"] = seizure[0]
                _seizure_span["end_time"] = seizure[1]
                _seizure_span.update(_inner_data.copy())
                _seizure_span.update(single_patient_data.copy())

                patients_data.append(_seizure_span)
    
    return patients_data


def get_seizure(lines: list):
    assert "Registration start" in lines[0]
    recording_start_time = re.search("(?<=:\s).*", lines[0]).group(0)
    seizure_start_time = re.search("(?<=:\s).*", lines[2]).group(0)
    seizure_end_time = re.search("(?<=:\s).*", lines[3]).group(0)
    recording_start_time = datetime.strptime(recording_start_time.strip(), "%H.%M.%S")
    seizure_start_time = datetime.strptime(seizure_start_time.strip(), "%H.%M.%S")
    seizure_end_time = datetime.strptime(seizure_end_time.strip(), "%H.%M.%S")

    if recording_start_time > seizure_start_time:
        recording_start_time = recording_start_time - timedelta(days=1)

    start_seconds = (seizure_start_time - recording_start_time).seconds
    end_seconds = (seizure_end_time - recording_start_time).seconds
    return (start_seconds, end_seconds)


def get_seizures(text_lines: list, patient: str) -> dict:
    seizures = []
    for index, line in enumerate(text_lines):
        if "File name" in line and patient in line:
            seizures.append(get_seizure(text_lines[index+1: index+5]))
    return seizures
