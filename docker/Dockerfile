# How to run code:
# sudo docker build . -f docker/Dockerfile -t msd_image && sudo docker run -it --rm --mount type=bind,source="$(pwd)",target=/usr/src/python_workspace msd_image
# sudo docker run -it --rm --mount type=bind,source="$(pwd)",target=/usr/src/python_workspace msd_image

FROM python:3.10

# create working directory
WORKDIR /usr/src/python_workspace

COPY . .
RUN pip install --no-cache-dir -r docker/requirements.txt

CMD ["python", "./msd_experiment.py"]
