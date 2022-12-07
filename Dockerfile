FROM python:3.10-slim-buster

WORKDIR /app
COPY requirements.txt requirements.txt
COPY src src
COPY data data

RUN pip3 install -r requirements.txt
RUN python3 src/model_training.py 
# RUN python3 src/predict_one.py