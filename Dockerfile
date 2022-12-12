FROM python:3.10-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt
COPY src src
COPY data/final data/final
COPY app app

RUN pip3 install --update pip3
RUN pip3 install -r requirements.txt

RUN python3 src/model_training.py 

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]