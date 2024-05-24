# Coupled Mass Spring Dampfer Simulator

STEP 1: 
- Adjust parameters in `msd_experiment.py`

STEP 2:
- Run directly with python: `python3 msd_experiment.py` 
- Run in dedicated environment: `./run_experiment.sh`
- Run in docker container: `sudo docker build . -f docker/Dockerfile -t msd_image && sudo docker run -it --rm --mount type=bind,source="$(pwd)",target=/usr/src/python_workspace msd_image`

Contact: alexander.strack@ipvs.uni-stuttgart.de