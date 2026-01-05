# EdgeAI-TestBench: High-Performance Inference Engine Benchmarking

EdgeAI-TestBench is a C++ application designed for benchmarking and comparing the performance of various edge AI inference engines. It provides a unified interface to evaluate models on different hardware backends (CPU, GPU, NPU), focusing on low latency, memory safety, and zero-copy semantics.

## Overview

In the world of edge computing, selecting the right inference engine for a specific hardware target is crucial for achieving optimal performance. This project was born out of the need for a standardized, high-fidelity tool to measure and compare the real-world performance of computer vision models on different edge backends.

This test bench allows developers and engineers to:
-   Benchmark object detection and semantic segmentation models.
-   Integrate heterogeneous inference backends (TensorFlow Lite, OpenVINO, TensorRT).
-   Analyze detailed performance metrics, including pre-processing, inference, and post-processing times.
-   Make informed decisions about which engine and hardware combination best suits their needs.

## Technology Stack

-   **Core**: C++20
-   **AI / Computer Vision**:
    -   OpenCV: Image pre/post-processing
    -   TensorFlow Lite: Mobile/edge CPU models
    -   OpenVINO: Intel hardware optimization(NOT IMPLEMENTED YET)
    -   TensorRT: NVIDIA hardware optimization(NOT IMPLEMENTED YET)


## Getting Started

### Prerequisites

-   C++20 compatible compiler (e.g., GCC 11+)
-   CMake (v3.16+)
-   (Optional but Recommended) Docker

### Building the Application

The project is designed to be built using CMake and Ninja. The following commands will configure and build the project, including fetching all the required dependencies.

```bash
cmake -S . -B build
cmake --build build/ -j$(nproc)
```
This will create the main executable at `build/bin/edge_inference`.

## Usage

To run the test bench, you need to provide a configuration file. An example can be found in `configs/config.xml`. The application takes the config file path as a command-line argument.

```bash
./build/bin/edge_inference --config /path/to/your/config.xml
```

The application will load the specified model and engine, run the benchmark on the provided dataset, and output the performance metrics.

## Configuration

The application is configured via an XML file. The main settings include:

-   `<type>`: The type of test to run (e.g., `object_detection`).
-   `<engineType>`: The inference engine to use (`tflite`, `openvino`, `tensorrt`).
-   `<datasetDir>`: Path to the dataset for benchmarking.
-   `<engine>`:
    -   `<modelPath>`: Path to the inference model file.
    -   `<classesPath>`: Path to the file containing class names.
    -   `<iou>`: IoU threshold for NMS.
    -   `<confidence>`: Confidence threshold for filtering detections.

Example:
```xml
<?xml version='1.0' encoding='UTF-8'>
<testBenchConfigs>
  <type value='object_detection' />
  <engineType value='tflite' />
  <datasetDir value='/path/to/dataset' />
  <engine>
    <modelPath value='/path/to/model' />
    <classesPath value='/path/to/classes' />
    <iou value='0.7'/>
    <confidence value='0.6'/>
  </engine>
</testBenchConfigs>
```

