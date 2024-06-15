#!/bin/sh

# Usage:
# ./build.sh [MODE] [URI_VIDEO_FILE]

# Check if a second parameter is provided for mode, otherwise use default (cuda)
MODE=${1:-cuda}
# Check if a URI parameter is provided, otherwise use default
URI_FILE=${2:-sintel_trailer-480p.webm}


# Set the symlink target based on the mode
if [ "$MODE" = "cpp" ]; then
    SYMLINK_TARGET="./build/libgstcudafilter-cpp.so"
else
    SYMLINK_TARGET="./build/libgstcudafilter-cu.so"
fi

# Run cmake and build
cmake -S . -B build --preset release -D USE_CUDA=ON  # 1 (ou debug)
cmake --build build                                  # 1

# Download the default file if it's not present and if the default is used
if [ "$URI_FILE" = "sintel_trailer-480p.webm" ] && [ ! -f sintel_trailer-480p.webm ]; then
    wget https://gstreamer.freedesktop.org/media/sintel_trailer-480p.webm
fi

# Set the plugin path and create the symlink
export GST_PLUGIN_PATH=$(pwd)                         # 3
ln -sf $SYMLINK_TARGET libgstcudafilter.so            # 4

# Construct the URI path
URI_PATH="file://$(pwd)/$URI_FILE"

# Cudafilter params can be set as follows:
# pipeline... ! cudafilter bg="mybg.ppm" th-low="3" th-high="30" bg-sampling-rate="500" bg-number-frames="10" ! ...pipeline

# Execute the gstreamer pipeline
gst-launch-1.0 uridecodebin uri=$URI_PATH ! videoconvert ! "video/x-raw, format=(string)RGB" ! cudafilter ! videoconvert ! video/x-raw, format=I420 ! x264enc ! mp4mux ! filesink location=video.mp4 #4
