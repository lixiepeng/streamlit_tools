import os
import subprocess
import time
import zipfile
import json

import pandas as pd
import streamlit as st

TMP_DIR = './tmp'

timestamp_str = str(time.time())
RESUME_DIR = os.path.join(TMP_DIR, timestamp_str)


def dump_zip(bytes):
    timestamp_str = str(time.time())
    file_name = timestamp_str + '.zip'
    file_path = os.path.join(TMP_DIR, file_name)
    with open(file_path, 'wb') as fw:
        fw.write(uploaded_file.read())


uploaded_file = st.file_uploader(
    "Choose a zip file contain your resumes:", type=["zip"])
if uploaded_file is not None:
    zf = zipfile.ZipFile(uploaded_file)
    zf.extractall(RESUME_DIR)
    subprocess.call(["python", "cv_utils.py", RESUME_DIR])
    with open(os.path.join(RESUME_DIR, 'resumes.json'), 'r', encoding='utf-8') as fr:
        resume_res = json.load(fr)
    st.json(resume_res)


# streamlit run cv_parser.py --browser.serverAddress=0.0.0.0 --server.maxUploadSize=100
# limit the uploaded zip file to 100MB