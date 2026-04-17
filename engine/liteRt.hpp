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
class DetectedObjects
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

typedef OptDetectedObjects std::optional<modelIo::DetectedObjects>;
typedef OptDetectedSemantics std::optional<modelIo::DetectedSemantics>;

/**
 * @brief Abstract base class for inference engines
 */
class LiteRtEngineBase
{
  std::unique_ptr<tflite::FlatBufferModel> m_flatBufferModel {nullptr};
  std::unique_ptr<tflite::Interpreter> m_interpreter {nullptr};
  TfLiteTensor* m_inputTensor {nullptr};
  TfLiteTensor* m_outputTensor {nullptr};
  std::vector<std::string> m_classNames;          /// \var class names    
  modelIo::ModelArch m_arch;                      /// \var model architecture
  float m_confidenceThresh;                       /// \var confidence threshold
  float m_iouThresh;                              /// \var iou threshold

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

  /**
   * @brief
   */
  struct RawDetections 
  {
    float* m_outputTensorData; 
    int m_frameWidth;
    int m_frameHeight; 
  };

  /**
   * @brief contains post processing methods for different model architecures
   */
  class PostProccessing 
  {
    public:
    static OptDetectedObjects yoloFive(const RawDetections& rawData, int numBoxes) const;
    static OptDetectedObjects yoloEight(const RawDetections& rawData, int numBoxes) const;
    static OptDetectedObjects yoloTen(const RawDetections& rawData, int numBoxes) const;
    static OptDetectedSemantics semanticSegment(const RawDetections& rawData, int numBoxes) const;
  };

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
  OptDetectedObjects runObjectDetection(const cv::Mat& frame);

  /**
   * @brief runs semantic segmentation on the input frame
   * @param frame input frame
   * @return true if successful, false otherwise
   */
  OptDetectedSemantics runSemanticDetection(const cv::Mat& frame);
};

class YoloV8