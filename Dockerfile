################################################
# Build Image
#################################################
FROM ubuntu:22.04 AS builder

# Prevent interactive prompts from apt (e.g., asking for timezone)
ENV DEBIAN_FRONTEND=noninteractive

# install required packages
RUN apt update && apt install -y \
    build-essential \ 
    cmake \
    libopencv-dev \
    libgtest-dev \
    && rm -rvf /var/lib/apt/lists/*
    

# copy the files into working dir
COPY . .

# create build dir and build project
RUN mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=RELEASE -DDELEGATE_TYPE=CPU .. && \
    make -j$(nproc)

# run the tests
RUN ctest --output-on-failure -j$(nproc)