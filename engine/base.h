#pragma once

#include "../utils/config/config.h"

#include <opencv2/core/mat.hpp>
#include <vector>
#include <string>
#include <memory>

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

/**
 * @brief Abstract base class for inference engines
 */
class AbsEngine
{
protected:
  cv::Mat m_resizedFrame;                         /// \var resized input frame        
  cv::Mat m_normalizedFrame;                      /// \var normalized input frame

  EngineConfig m_config;                          /// \var engine configuration
  DetectedObjects m_odOutput;                     /// \var object detection output
  DetectedSemantics m_semantics;                  /// \var semantic segmentation output
  
  std::vector<std::string> m_classNames;          /// \var class names
  int m_width {0};                                /// \var model's input width      
  int m_height {0};                               /// \var model's input height
  int m_inputChannels {0};                        /// \var model's input channels

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
   * @brief run post proccessing algorithm on the output tensor of a YOLOv5 model
   * @param data pointer to the output tensor data
   * @param frameWidth original frame width
   * @param frameHeight original frame height
   * @return true if successful, false otherwise
   */
  bool yoloFivePostProc(void* data, int frameWidth, int frameHeight);

  /**
   * @brief run post proccessing algorithm on the output tensor of an SSD model
   * @param data pointer to the output tensor data
   * @param frameWidth original frame width
   * @param frameHeight original frame height
   * @return true if successful, false otherwise
   */
  bool ssdPostProc(void* data, int frameWidth, int frameHeight);

public:
  /**
   * @brief initializes the engine with the given configuration file
   * @param configPath path to the configuration file
   * @return true if successful, false otherwise
   */
  bool init(const std::string& configPath);

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

  virtual ~AbsEngine() = default;
};






