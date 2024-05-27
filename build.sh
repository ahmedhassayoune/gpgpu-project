#!/bin/sh

# Check if a URI parameter is provided, otherwise use default
URI_FILE=${1:-sintel_trailer-480p.webm}

cmake -S . -B build --preset release -D USE_CUDA=ON  # 1 (ou debug)
cmake --build build                                  # 1

# Download the default file if it's not present and if the default is used
if [ "$URI_FILE" = "sintel_trailer-480p.webm" ] && [ ! -f sintel_trailer-480p.webm ]; then
    wget https://gstreamer.freedesktop.org/media/sintel_trailer-480p.webm
fi

export GST_PLUGIN_PATH=$(pwd)                                         # 3
ln -s ./build/libgstcudafilter-cpp.so libgstcudafilter.so             # 4

# Construct the URI path
URI_PATH="file://$(pwd)/$URI_FILE"

gst-launch-1.0 uridecodebin uri=$URI_PATH ! videoconvert ! "video/x-raw, format=(string)RGB" ! cudafilter ! videoconvert ! video/x-raw, format=I420 ! x264enc ! mp4mux ! filesink location=video.mp4 #4
