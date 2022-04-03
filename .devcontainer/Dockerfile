FROM python:3.7

RUN apt update && apt install -y zip htop screen libgl1-mesa-glx

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN mkdir -p /workspace
WORKDIR /workspace