#pragma once

#include <string>

/**
 * @brief EngineType defines the type of inference engine to be used
 */
enum class EngineType {TFLITE, OPENVINO, TENSORRT, UNKNOWN};

/**
 * @brief TestBenchType defines the type of test bench to be used
 */
enum class TestBenchType {OBJECT_DETECTION, SEMANTIC_SEGMENTATION, UNKNOWN};

/**
 * @brief ModelArch defines what kind of model is used and according to that what kind 
 * of post processing function we need
 */
enum class ModelArch {SSD, YOLO5, YOLOV8, YOLO10, UNKNOWN};


/**
 * @brief TestBenchConfig holds the configuration parameters for the test bench
 */
struct TestBenchConfig
{
  std::string m_modelPath;                /// \var path to the model file
  std::string m_classNamesPath;           /// \var path to the class names file
  std::string m_datasetDir;                /// \var path to the dataset directory
  float m_iouThreshold;                   /// \var IOU threshold for non-max suppression
  float m_confidenceThreshold;            /// \var confidence threshold for detections
  EngineType m_engineType;                 /// \var type of the inference engine
  TestBenchType m_benchType;               /// \var type of the test bench
  ModelArch m_arch;

  /**
   * @brief parses the xml configuration file at the given path
   * @param path path to the xml configuration file
   * @return true if parsing was successful, false otherwise
   */
  bool parseConfigFile(const std::string& path);
};

