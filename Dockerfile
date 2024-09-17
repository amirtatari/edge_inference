# Base image
FROM ubuntu:22.04

# set work dir
WORKDIR /tflite_test

# install packages
RUN apt update && apt install -y \
    build-essential \ 
    cmake \
    git \
    wget \
    nano \
    python3

# change the working dir
RUN pwd

# Clone TensorFlow repository
RUN git clone https://github.com/tensorflow/tensorflow.git

# copy the files into working dir
COPY main.cpp .
COPY CMakeLists.txt .


# create build dir and build project
RUN mkdir build && cd build && \
    cmake .. && \
    cmake --build .

COPY model.tflite .

# start the app
CMD ["./build/inference"]
