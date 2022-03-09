#!/usr/bin/python3.7
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


def download_file(source: str, destine: str) -> int:
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    r = session.get(source, allow_redirects=True, stream=True)
    size = len(r.content)
    with open(destine + source.split("/")[-1],'wb') as f:
        f.write(r.content)
    return size
