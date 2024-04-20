#!/bin/sh

cmake -S . -B build --preset release -D USE_CUDA=ON  # 1 (ou debug)
cmake --build build                                  # 1


if [ ! -f sintel_trailer-480p.webm ]; then
    wget https://gstreamer.freedesktop.org/media/sintel_trailer-480p.webm
fi
export GST_PLUGIN_PATH=$(pwd)                                         # 3
ln -s ./build/libgstcudafilter-cpp.so libgstcudafilter.so             # 4


gst-launch-1.0 uridecodebin uri=file://$(pwd)/sintel_trailer-480p.webm ! videoconvert ! "video/x-raw, format=(string)RGB" ! cudafilter ! videoconvert ! video/x-raw, format=I420 ! x264enc ! mp4mux ! filesink location=video.mp4 #4