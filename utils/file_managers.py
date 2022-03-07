#!/usr/bin/python3.7
import os
import mne
import json
import pickle
import numpy as np
import pandas as pd


def get_edf_content(file_name: str) -> np.array:
    data = mne.io.read_raw_edf(file_name)
    return data.get_data()


def get_json_content(file_name: str) -> dict:
    with open(file_name, "r") as f:
        content = json.load(f)
    return content


def write_json(file_name: str, lines: list):
    with open(file_name, "w") as f:
        json.dump(lines, f)


def write_csv(file_name: str, data: list, sort_keys: list,
              columns: list):
    df = pd.DataFrame(data)
    df.sort_values(sort_keys, inplace=True)
    df = df[columns]
    df.to_csv(file_name, index=False)


def write_pickle(file_name: str, data: list):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)


def read_pickle(file_name: str):
    with open(file_name, 'rb') as f:
        content = pickle.load(f)
    return content


def read_pickles(file_names: list, ):
    output = []
    for file in file_names:
        content = read_pickle(file)
        output.extend(content)
    return output


def get_text_content(file_name: str) -> list:
    with open(file_name, "r") as f:
        content = f.readlines()
    return [l.strip().replace("\n", "") for l in content]


def list_dir(path) -> list:
    files = os.listdir(path)
    files = [path + file for file in files]
    return files
