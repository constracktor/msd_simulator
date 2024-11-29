#!/bin/bash
# create python enviroment
python3 -m venv msd_env
# activate enviroment
source msd_env/bin/activate
# install requirements
pip install --no-cache-dir -r requirements.txt
# launch 
python3 generate_msd_data.py
