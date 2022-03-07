#!/usr/bin/python3.7
import requests


def download_file(source: str, destine: str) -> int:
    r = requests.get(source, allow_redirects=True, stream=True)
    size = len(r.content)
    with open(destine + source.split("/")[-1],'wb') as f:
        f.write(r.content)
    return size