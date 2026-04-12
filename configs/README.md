# Configuration
The application is configured via an XML file. The main settings include:
- `<type>`: The type of test to run (e.g., `object_detection`).
- `<engineType>`: The inference engine to use (`tflite`, `openvino`, `tensorrt`).
- `<datasetDir>`: Path to the dataset for benchmarking.
- `<engine>`:
  - `<modelPath>`: Path to the inference model file.
  - `<classesPath>`: Path to the file containing class names.
  - `<iou>`: IoU threshold for NMS.
  - `<confidence>`: Confidence threshold for filtering detections.
