#!/usr/bin/python3.7
"""
1.- Download all summary files from physionet CHB-MIT
"""
import os
import sys
import logging
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir, ".."))
from utils import settings
from utils.http import download_file


record_seizure_file = os.path.join(current_dir, settings["chb-mit"]["seizure_records"])
dst_directory = os.path.join(current_dir, settings["chb-mit"]["metadata"])
chb_url = "https://physionet.org/files/chbmit/1.0.0/"


def main():

    suffix = "-summary.txt"
    with open(record_seizure_file, "r") as f:
        files = {a.split("/")[-2] for a in f.readlines() if a != "\n"}

    for a in files:
        size = download_file(source=chb_url + a + "/" + a + suffix,
                             destine=dst_directory)
        logging.info("File dowloaded: %s, size: %s" % (a, size))


if __name__ == "__main__":
    main()