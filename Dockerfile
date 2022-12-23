FROM python:3.10

WORKDIR /app

COPY . .

RUN pip3 install -r requirements.txt
RUN python3 src/model_training.py 

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]