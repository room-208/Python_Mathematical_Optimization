FROM python:3.7

RUN apt update && apt install -y zip htop screen libgl1-mesa-glx

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.8-cp37-cp37m-linux_x86_64.whl

RUN mkdir -p /workspace
WORKDIR /workspace