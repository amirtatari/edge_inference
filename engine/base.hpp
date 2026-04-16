#pragma once

// liteRt
#include <tensorflow/lite/model.h>

// opencv
#include <opencv2/core/mat.hpp>

// stl
#include <memory>
#include <vector>
#include <string>
#include <optional>

namespace modelIo 
{

/**
 * @brief defines what kind of model is used and according to that what kind 
 * of post processing function we need
 */
enum class ModelArch : int {SSD, YOLO5, YOLOV8, YOLO10};

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

}; // namespace modelIo

/**
 * @brief Abstract base class for inference engines
 */
class LiteRtEngine
{
  std::unique_ptr<tflite::FlatBufferModel> m_flatBufferModel {nullptr};
  std::unique_ptr<tflite::Interpreter> m_interpreter {nullptr};
  TfLiteTensor* m_inputTensor {nullptr};
  TfLiteTensor* m_outputTensor {nullptr};
  std::vector<std::string> m_classNames;          /// \var class names    
  modelIo::ModelArch m_arch;                      /// \var model architecture

  /**
   * @brief loads the model from the given binary path
   * @param path path to the model binary
   */
  void loadModel(const std::string& path);
  
  /**
   * @brief loads class names from the given file path
   * @param path path to the class names file
   */
  void loadClassNames(const std::string& path);

  /**
   * @brief runs inference on input frame
   * @param frame input frame
   * @return float* array of detections
   */
  float* runInference(const cv::Mat& frame);

public:
    explicit LiteRtEngine(ModelArch arch, const std::string& modelPath, cosnt std::string& classesPath);
    LiteRtEngine(const LiteRtEngine&) = delete;
    LiteRtEngine(LiteRtEngine&&) = delete;
    LiteRtEngine& operator=(const LiteRtEngine&) = delete;
    LiteRtEngine& operator=(LiteRtEngine&&) = delete;

  /**
   * @brief runs object detection on the input frame
   * @param frame input frame
   * @return true if successful, false otherwise
   */
  std::optional<modelIo::DetectedObjects> runObjectDetection(const cv::Mat& frame);

  /**
   * @brief runs semantic segmentation on the input frame
   * @param frame input frame
   * @return true if successful, false otherwise
   */
  std::optional<modelIo::DetectedSemantics> runSemanticDetection(const cv::Mat& frame);
};






