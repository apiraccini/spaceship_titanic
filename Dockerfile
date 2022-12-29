FROM python:3.10-slim-buster

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y install curl
RUN apt-get install libgomp1

RUN pip3 install -r requirements.txt
RUN python3 src/model_training.py 

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]