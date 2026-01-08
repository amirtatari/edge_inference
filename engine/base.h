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

  TestBenchConfig* m_config;                      /// \var ptr to test bench configuration
  DetectedObjects m_odOutput;                     /// \var object detection output
  DetectedSemantics m_semantics;                  /// \var semantic segmentation output
  
  std::vector<std::string> m_classNames;          /// \var class names
  int m_width {0};                                /// \var model's input width      
  int m_height {0};                               /// \var model's input height
  int m_inputChannels {0};                        /// \var model's input channels
  int m_numBoxes {0};                             /// \var number of candidate boxes

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
   * @param input input frame
   * @param output output frame (normalized)
   */
  void resizeAndNormalize(const cv::Mat& input, cv::Mat& output);

  /**
   * @brief run post proccessing algorithm on the output tensor of a YOLOv5 model
   * @param data pointer to the output tensor data
   * @param frameWidth original frame width
   * @param frameHeight original frame height
   * @return true if successful, false otherwise
   */
  bool yoloFivePostProc(void* data, int frameWidth, int frameHeight);

  /**
   * @brief run post proccessing algorithm on the output tensor of a YOLOv8 model
   * @param data pointer to the output tensor data
   * @param frameWidth original frame width
   * @param frameHeight original frame height
   * @return true if successful, false otherwise
   */
  bool yoloEightPostProc(void* data, int frameWidth, int frameHeight);

  /**
   * @brief run post proccessing algorithm on the output tensor of a YOLOv10 model
   * @param data pointer to the output tensor data
   * @param frameWidth original frame width
   * @param frameHeight original frame height
   * @return true if successful, false otherwise
   */
  bool yoloTenPostProc(void* data, int frameWidth, int frameHeight);

  /**
   * @brief run post proccessing algorithm on the output tensor of an SSD model
   * @param data pointer to the output tensor data
   * @param frameWidth original frame width
   * @param frameHeight original frame height
   * @return true if successful, false otherwise
   */
  bool ssdPostProc(void* data, int frameWidth, int frameHeight);

  /**
   * @brief run post proccessing algorithm for semantic segmentation model
   * @param data pointer to the output tensor data
   * @param outW output tensor width
   * @param outH output tensor height
   * @param numClasses number of classes
   * @param frameWidth original frame width
   * @param frameHeight original frame height
   */
  void semanticPostProc(void* data, int outW, int outH, int numClasses,
                        int frameWidth, int frameHeight);

  /**
   * @brief applies non-maximum suppression to filter overlapping boxes
   * @param boxes vector of bounding boxes
   * @param scores vector of confidence scores
   * @param classIds vector of class indices
   */
  void applyNms(const std::vector<cv::Rect>& boxes, const std::vector<float>& scores,
                const std::vector<int>& classIds);
  
  /**
   * @brief calculates Intersection over Union (IoU) between two boxes
   * @param box1 first bounding box
   * @param box2 second bounding box
   * @return IoU value
   */
  float calculateIoU(const cv::Rect& box1, const cv::Rect& box2);

public:
  /**
   * @brief initializes the engine with the given configuration file
   * @param configPath path to the configuration file
   * @return true if successful, false otherwise
   */
  bool init(TestBenchConfig* config);

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






