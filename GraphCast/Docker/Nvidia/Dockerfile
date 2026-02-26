FROM nvcr.io/nvidia/jax:26.01-py3

RUN apt-get update --quiet \
    && apt-get dist-upgrade --yes --quiet \
    && apt-get install --yes --quiet software-properties-common \
    && apt-get install --yes --quiet git wget gcc g++

RUN pip install --upgrade pip

RUN pip install git+https://github.com/deepmind/graphcast.git

RUN pip install --upgrade dm-haiku ipywidgets

RUN pip install google-cloud-storage
