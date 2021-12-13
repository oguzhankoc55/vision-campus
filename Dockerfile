FROM ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update \
      && apt-get install --no-install-recommends --no-install-suggests -y gnupg2 ca-certificates git build-essential \
      && rm -rf /var/lib/apt/lists/*

RUN apt-get update -y --fix-missing
RUN apt install -y software-properties-common curl git wget
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y python3.8  python3-distutils
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 0

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python get-pip.py --force-reinstall
RUN rm get-pip.py

RUN apt-get update && apt-get install -y python3-opencv

WORKDIR /

RUN git clone https://github.com/AlexeyAB/darknet.git \
     && cd darknet \
     && make -j3 \
     && cp darknet /usr/local/bin/

WORKDIR /opt/project
ADD . /opt/project

RUN pip install -r requirements.txt