# Base image: Ubuntu 18.04 with Python
FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.6 \
    python3.6-dev \
    python3-pip \
    build-essential \
    libsndfile1 \
    fluidsynth \
    wget \
    ffmpeg \
    git \
    && apt-get clean

# Make python3.6 the default
RUN ln -s /usr/bin/python3.6 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

# Upgrade pip
RUN pip install --upgrade pip

# Install required Python packages
RUN pip install \
    jupyter \
    papermill \
    tensorflow==1.14 \
    librosa \
    pandas \
    numpy \
    scipy \
    matplotlib \
    dill \
    mir_eval \
    imageio \
    soundfile \
    pypianoroll==0.4.2 \
    pretty_midi \
    midiutil \
    opencv-python

# Set working directory
WORKDIR /workspace

# Copy notebook execution script
COPY run_all.sh /workspace/run_all.sh

# Make script executable
RUN chmod +x /workspace/run_all.sh

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Run the notebooks when container starts
CMD ["bash", "/workspace/run_all.sh"]