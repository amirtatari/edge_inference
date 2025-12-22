#pragma once

#include <opencv2/core/mat.hpp>
#include <vector>
#include <string>

/**
 * @brief ModelArch defines what kind of model is used and according to that what kind 
 * of post processing function we need
 */
enum class ModelArch {SSD, YOLO5, UNKNOWN};

/**
 * @brief Data structure of object detecion output 
 */
struct DetectedObjects
{
  std::vector<float> m_classProbabilities;         /// \var class probabilities
  std::vector<cv::Point> m_firstPoints;            /// \var bbox top-left points
  std::vector<cv::Point> m_secondPoints;           /// \var bbox buttom-right points
  std::vector<std::size_t> m_classNameIdxs;        /// \var class name indexes
};

/**
 * @brief Data structure for semantic segmentation ouput
 */
struct DetectedSemantics
{
  std::vector<cv::Point> m_pixels;                /// \var segmented pixel
  std::vector<std::size_t> m_classNameIdxs;       /// \var class name indicies
};

class AbsEngine
{
protected:
  cv::Mat m_resizedFrame;
  cv::Mat m_normalizedFrame;

  DetectedObjects m_odOutput;
  DetectedSemantics m_semantics;
  
  std::vector<std::string> m_classNames;
  float m_confidenceScore {0.0F};
  float m_iou {0.0f};
  int m_width {0};
  int m_height {0};
  int m_inputChannels {0};
  ModelArch m_arch {ModelArch::UNKNOWN};

  /**
   * @brief runs object detection on the input frame
   * @param frame input frame
   * @return true if successful, false otherwise
   */
  virtual bool runObjectDetection(const cv::Mat& frame) = 0;

  /**
   * @brief runs semantic segmentation on the input frame
   * @param frame input frame
   * @return true if successful, false otherwise
   */
  virtual bool runSemanticDetection(const cv::Mat& frame) = 0;

  /**
   * @brief loads the model from the given binary path
   * @param path path to the model binary
   * @return true if successful, false otherwise
   */
  virtual bool loadModel(const std::string& path) = 0;
  
  /**
   * @brief loads class names from the given file path
   * @param path path to the class names file
   * @return true if successful, false otherwise
   */
  bool loadClassNames(const std::string& path);

  /**
   * @brief resizes and normalizes the input frame
   * @param frame input frame
   */
  void resizeAndNormalize(const cv::Mat& frame);

  /**
   * @brief parses the configuration file
   * @param configPath path to the xml config file
   * @return true if successful, false otherwise
   */
  bool parseConfig(const std::string& path);

  bool yoloFivePostProc(void* data, int frameWidth, int frameHeight);
  bool ssdPostProc(void* data, int frameWidth, int frameHeight);

public:
  virtual ~AbsEngine() = default;
};






