#!/usr/bin/python3.7
"""
1.- Download all summary files from physionet CHB-MIT
"""
import os
import sys
import logging
sys.path.append("../../Src")
from utils import settings
from utils.http import download_file


base_path = os.path.dirname(os.path.realpath(__file__))
record_seizure_file = settings["chb-mit"]["seizure_records"]
dst_directory = base_path + "/" + settings["chb-mit"]["metadata"]
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