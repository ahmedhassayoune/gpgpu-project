FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    cmake \
    clang \
    ninja-build \
    pkg-config \
    libpng-dev \
    wget \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-tools

WORKDIR /gpgpu

COPY . .

RUN wget https://gstreamer.freedesktop.org/media/sintel_trailer-480p.webm
RUN chmod +x build.sh