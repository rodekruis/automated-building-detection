FROM nvidia/cuda:10.2-runtime-ubuntu18.04

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt-get update && \
	apt-get install -y python3-pip && \
	ln -sfn /usr/bin/python3.7 /usr/bin/python && \
	ln -sfn /usr/bin/pip3 /usr/bin/pip

RUN deps='build-essential cmake gdal-bin python-gdal libgdal-dev kmod wget apache2 vim apt-utils' && \
	apt-get update && \
	apt-get install -y $deps && \
	pip install --upgrade pip && \
	pip install GDAL==$(gdal-config --version)

WORKDIR /abd_model
ADD abd_model .
RUN pip install .

WORKDIR /abd_utils
ADD abd_utils .
RUN pip install .

WORKDIR /
