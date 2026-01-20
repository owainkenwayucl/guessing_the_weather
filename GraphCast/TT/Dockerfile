FROM ghcr.io/tenstorrent/tt-xla-slim:latest

RUN apt-get update --quiet \
    && apt-get dist-upgrade --yes --quiet \
    && apt-get install --yes --quiet software-properties-common \
    && apt-get install --yes --quiet git wget gcc g++

RUN pip install --upgrade pip

RUN pip install git+https://github.com/deepmind/graphcast.git

RUN pip install ai-models ai-models-graphcast

RUN pip install climetlab

RUN pip install "earthkit-data>=0.11.1"
