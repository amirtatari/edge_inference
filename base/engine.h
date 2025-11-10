#pragma once

#include <vector>
#include <memory>
#include <tensorflow/lite/model.h>
#include <opencv2/core/mat.hpp>

/**
 * @brief Input represent the data structure that Engine::runObjectDetection or Engine::runSemanticDetection accepts
 */
struct Input
{
  cv::Mat m_frame;                                                       /// \var input frame
  std::vector<int> m_roiXs;                                              /// \var vector of ROI poitns x coordinates
  std::vector<int> m_roiYs;                                              /// \var vector of ROI poitns y coordinates
  std::vector<const char*> m_targets;                                    /// \var vector of target classes that need to be detected
  std::size_t m_modelIdx;                                                /// \var model index that input should run with
};

/**
 * @brief ModelArch defines what kind of model is used and according to that what kind of post processing function we need
 */
enum class ModelArch
{
  SSD ,
  YOLO5
};

/**
 * @brief Models represents data structure of tflite models
 */
struct Models
{
  std::vector<std::vector<const char*>> m_classNames;                    /// \var vector of class names
  std::vector<std::unique_ptr<tflite::FlatBufferModel>> m_flatBufferPtrs; /// \var vector of flat buffer ptrs
  std::vector<std::unique_ptr<tflite::Interpreter>> m_interpreterPtrs;   /// \var vector of interpreter ptrs
  std::vector<TfLiteTensor*> m_inputTensorPtrs;                          /// \var vector of input tensor ptrs
  std::vector<TfLiteTensor*> m_outputTensorPtrs;                         /// \var vector of output tensor ptrs
  std::vector<int> m_widths;                                             /// \var vector of model's widths
  std::vector<int> m_height;                                             /// \var vector of model's heights
  std::vector<int> m_numInputChannels;                                   /// \var vector of model's number of input channels
  std::vector<float> m_confidences;                                      /// \var vector of model's confidence threshold
  std::vector<float> m_IoUs;                                             /// \var vector of model's IoU threshold
  std::vector<ModelArch> m_archs;                                        /// \var vector of model's architecture  
};

/**
 * @brief DetectedObjects represents the data structure of object detecion task's output 
 */
struct DetectedObjects
{
  std::vector<float> m_classProbabilities;                               /// \var vector of class probabilities
  std::vector<int> m_xOnes;                                              /// \var vector of bounding box top-left x coordinate
  std::vector<int> m_xTwos;                                              /// \var vector of bounding box buttom-right x coordinate 
  std::vector<int> m_yOnes;                                              /// \var vector of bounding box top-left y coordinate
  std::vector<int> m_yTwos;                                              /// \var vector of bounding box buttom-right y coordinate
  std::vector<std::size_t> m_classNameIdxs;                              /// \var vector of indicies of vector of class name
};

/**
 * @brief DetectedSemantics represents the data structure semantic segmentation task's ouput
 */
struct DetectedSemantics
{
  std::vector<int> m_pixelXs;                                            /// \var vector of pixel x coordinates
  std::vector<int> m_pixelYs;                                            /// \var vector of pixel y coordinates
  std::vector<std::size_t> m_classNameIdxs;                              /// \var vector of indicies of vector of class name
};

/*
class Engine loads the models and runs the inference on the input frame
*/
struct Engine 
{
  Engine(const std::string& configPath);
  bool runObjectDetection(const Input& input);
  bool runSemanticDetection(const Input& input);
  bool loadModel(const char* modelPath, const char* classNamePath, const char* modelType, float confidence, float iou);

private:
  void parseConfig(const std::string& configPath);
  bool loadFlatBuffer(const char* model, const char* modelType, float confidence, float iou);
  bool loadClassNames(const char* path);

  // ---------------- Pre Processing ---------------- //
  bool resizeFrame(cv::Mat& frame, int x, int y);

  // ---------------- Post Proccessing -------------- //
  bool runYoloPostProc(std::size_t modelIdx, int originalFrameWidth, int originalFrameHeight);
  bool runSsdPostProc(std::size_t modelIdx, int originalFrameWidth, int originalFrameHeight);
 
  // ------------------ Memebers -------------------- //
  Models m_models;
  DetectedObjects m_objsDetected;
  DetectedSemantics m_semanticsDetected;
};
