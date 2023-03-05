# syntax=docker/dockerfile:1

FROM python:3.8.10-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install flask
RUN pip3 install -r requirements.txt
RUN apt-get install tesseract-ocr -y

COPY . .


CMD [ "python3", "server.py"]
