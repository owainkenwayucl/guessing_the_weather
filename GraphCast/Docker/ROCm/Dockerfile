FROM docker.io/rocm/jax:rocm6.3.2-jax0.4.31-py3.10

RUN apt-get update --quiet \
    && apt-get dist-upgrade --yes --quiet \
    && apt-get install --yes --quiet software-properties-common \
    && apt-get install --yes --quiet git wget gcc g++

RUN pip install --upgrade pip

RUN pip install git+https://github.com/deepmind/graphcast.git@v0.1.1

RUN pip install ai-models ai-models-graphcast

RUN pip install climetlab

RUN pip install "earthkit-data>=0.11.1"