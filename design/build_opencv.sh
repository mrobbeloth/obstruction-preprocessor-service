#!/bin/bash
# save as build_opencv.sh

OPENCV_VERSION="4.11.0"
COMPUTE_CAP="5.2" # Change this for different GPUs

cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D WITH_CUDA=ON \
      -D WITH_CUDNN=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D CUDA_ARCH_BIN=$COMPUTE_CAP \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      ..
