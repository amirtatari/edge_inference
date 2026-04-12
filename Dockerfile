################################################
# Build Image
#################################################
FROM ubuntu:22.04 AS host
ENV DEBIAN_FRONTEND=noninteractive

# 1. install required packages
RUN apt-get update && apt-get install -y \
    build-essential cmake git wget unzip curl python3 python3-pip \
    ninja-build \
    autoconf libtool pkg-config

# 2. clone libraries required to build the project to be able to build
RUN git clone -b 4.5.2 --depth 1 https://github.com/opencv/opencv.git
RUN git clone -b v2.15.0 --depth 1 https://github.com/tensorflow/tensorflow.git tflite_src
RUN git clone -b v23.5.26 https://github.com/google/flatbuffers.git && \
    cd flatbuffers && cmake -G "Unix Makefiles" . \
    -DCMAKE_BUILD_TYPE=Release \
    -DFLATBUFFERS_BUILD_TESTS=OFF && \
    make -j$(nproc) install && \
    ldconfig
RUN git clone --single-branch --depth=1 --branch "release-1.10.0" https://github.com/google/googletest.git && \
    mkdir -vp googletest/build && cd googletest && \
    cmake -G "Ninja" -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_GMOCK=OFF && \
    cmake --build build -j"$(nproc)" && \
    cmake --install build && \
    ldconfig

# target build x86
ENV SDK_INSTALL_PATH=/opt/sdk/x86

# build and install opencv
RUN cmake -G "Ninja" -S opencv -B build_opencv_x86 \
    -DCMAKE_INSTALL_PREFIX=$SDK_INSTALL_PATH \
    -DBUILD_OPENCV_DNN=OFF \
    -DBUILD_PROTOBUF=OFF \
    -DWITH_PROTOBUF=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_TESTS=OFF \ 
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_DOCS=OFF \
    -DBUILD_OPENCV_PYTHON3=OFF \
    -DBUILD_OPENCV_JAVA=OFF \
    -DBUILD_OPENCV_APPS=OFF \
    -DBUILD_OPENCV_GAPI=OFF \
    -DBUILD_JPEG=ON \
    -DBUILD_PNG=ON && \
    cmake --build build_opencv_x86 -j$(nproc) --target install

# build TFLite (shared library, manual install)
RUN cmake -G "Ninja" -S tflite_src/tensorflow/lite -B build_tflite_x86 \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DTFLITE_ENABLE_XNNPACK=ON \
    -DABSL_PROPAGATE_CXX_STD=ON \
    -DTFLITE_ENABLE_INSTALL=OFF \
    -DCMAKE_INSTALL_RPATH=$SDK_INSTALL_PATH/lib \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DTENSORFLOW_SOURCE_DIR=/tflite_src && \
    cmake --build build_tflite_x86 -j$(nproc)
# manual install 
RUN mkdir -p $SDK_INSTALL_PATH/lib $SDK_INSTALL_PATH/include/tensorflow/lite && \
    cp -a build_tflite_x86/libtensorflow-lite.so* $SDK_INSTALL_PATH/lib/ && \
    find build_tflite_x86 -type f -name "*.so*" -exec cp -a {} $SDK_INSTALL_PATH/lib/ \; && \
    cd tflite_src/tensorflow/lite && \
    find . -name "*.h" -exec cp --parents {} $SDK_INSTALL_PATH/include/tensorflow/lite/ \;

WORKDIR /workspace
