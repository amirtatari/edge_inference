# EdgeAI-TestBench: High-Performance Inference Engine Benchmarking
EdgeAI-TestBench is a C++ application designed for benchmarking and comparing the performance of various computer vision models. It provides a unified interface to evaluate models on different hardware backends (CPU, GPU, NPU).

## Overview
In the world of edge computing, selecting the right inference engine for a specific hardware target is crucial for achieving optimal performance. This project was born out of the need for a standardized, high-fidelity tool to measure and compare the real-world performance of computer vision models on different edge backends.

This test bench allows developers and engineers to:
- Benchmark object detection and semantic segmentation models on edge hardware
- Analyze detailed performance metrics, including pre-processing, inference, and post-processing times.
- Make informed decisions about which engine and hardware combination best suits their needs.

## Prerequisites
-   C++20
-   CMake (v3.16+)
-   Docker

## Building & Deployment

1. Build the docker image using the docker file:
```Bash
docker build -t edge_inference-sdk:latest .
```

2. Build the application using the SDK image:
```Bash
mkdir -vp build
docker run --rm -v $(pwd):/workspace edge_inference-sdk:latest bash -c "
  cmake -G Ninja -S /workspace -B /workspace/build \
    -DCMAKE_BUILD_TYPE=Release \
    -DSDK_PATH=/opt/sdk/x86 && \
  cmake --build /workspace/build -j$(nproc)"
```

3. Run the executable:
```Bash
docker run --rm -v $(pwd):/workspace edge_inference-sdk:latest bash -c "
    build/edge_inference --config configs/config.xml"
```

4. Run the unit tests:
```Bash
docker run --rm -v $(pwd):/workspace edge_inference-sdk:latest bash -c "
  cd /workspace/build && ctest --output-on-failure -j$(nproc)"
```


