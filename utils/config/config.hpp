#pragma once

#include <string>
#include <pugi/pugixml.hpp>

/**
 * @brief defines the type of computer vision task to be used
 */
enum class TaskType : int {OBJECT_DETECTION, SEMANTIC_SEGMENTATION};

/**
 * @brief defines what kind of model is used and according to that what kind 
 * of post processing function we need
 */
enum class ModelArch : int {SSD, YOLO5, YOLOV8, YOLO10};

/**
 * @brief holds the configuration parameters
 */
class Config
{
  /**
   * @brief parse engine node in config file
   * @param engineNode xml node
   */
  void parseEngineNode(const pugi::xml_node& engineNode);

  /**
   * @brief parse test bench configs node
   * @param root root xml node
   */
  void parseRootNode(const pugi::xml_node& root);

public:
  std::string m_modelPath;                /// \var path to the model file
  std::string m_classNamesPath;           /// \var path to the class names file
  std::string m_datasetDir;               /// \var path to the dataset directory
  float m_iouThreshold;                   /// \var IOU threshold for non-max suppression
  float m_confidenceThreshold;            /// \var confidence threshold for detections
  TaskType m_taskType;                    /// \var type of the 
  ModelArch m_arch;

  Config() = default;
  Config(const Config&) = delete;
  Config(Config&&) = delete;
  Config& operator=(const Config&) = delete;
  Config& operator=(Config&&) = delete;
  
  /**
   * @brief parses the xml configuration file at the given path
   * @param path path to the xml configuration file
   */
  void parseFromFile (const std::string& path);
};

