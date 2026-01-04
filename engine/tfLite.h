#pragma once

#include <memory>
#include <tensorflow/lite/model.h>

#include "base.h"

/**
 * @brief TensorFlow Lite inference engine implementation
 */
class EngineLite : public AbsEngine
{
  std::unique_ptr<tflite::FlatBufferModel> m_flatBufferModel {nullptr};
  std::unique_ptr<tflite::Interpreter> m_interpreter {nullptr};
  TfLiteTensor* m_inputTensor {nullptr};
  TfLiteTensor* m_outputTensor {nullptr};

  float* runInference(const cv::Mat& frame);

public:
  bool runObjectDetection(const cv::Mat& frame);
  bool runSemanticDetection(const cv::Mat& input);
  bool loadModel(const std::string& path);
};
