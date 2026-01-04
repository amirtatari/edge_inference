#include "tfLite.h"
#include "../../utils/profiler/profiler.h"

#include <opencv2/imgproc.hpp>
#include <fstream>
#include <tensorflow/lite/interpreter_builder.h>
#include <tensorflow/lite/kernels/register.h>
#include <spdlog/spdlog.h>

bool EngineLite::loadModel(const std::string& path)
{
  spdlog::info("EngineLite::loadModel: loading model from {}", path);

  m_flatBufferModel = tflite::FlatBufferModel::BuildFromFile(path.c_str());
  if (m_flatBufferModel == nullptr)
  {
    spdlog::error("EngineLite::loadModel: failed to build model from file: {}", path);
    return false;
  }

  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*m_flatBufferModel, resolver);
  if (builder(&m_interpreter) != kTfLiteOk)
  {
    spdlog::error("EngineLite::loadModel: failed to build interpreter");
    return false;
  }

  if (m_interpreter->AllocateTensors() != kTfLiteOk)
  {
    spdlog::error("EngineLite::loadModel: failed to allocate tensors");
    return false;
  }

  m_inputTensor = m_interpreter->tensor(m_interpreter->inputs()[0]);
  m_outputTensor = m_interpreter->tensor(m_interpreter->outputs()[0]);

  if (m_inputTensor == nullptr || m_outputTensor == nullptr)
  {
    spdlog::error("EngineLite::loadModel: failed to get input/output tensors.");
    return false;
  }

  if (m_inputTensor->dims->size == 4)
  {
    m_height = m_inputTensor->dims->data[1];
    m_width = m_inputTensor->dims->data[2];
    m_inputChannels = m_inputTensor->dims->data[3];
  }
  else
  {
    spdlog::warn("EngineLite::loadModel: unexpected input tensor dimension size: {}",
                 m_inputTensor->dims->size);
  }

  return true;
}

float* EngineLite::runInference(const cv::Mat& frame)
{
  // resize and normalize the input frame
  resizeAndNormalize(frame);

  // Copy data to input tensor
  memcpy(m_inputTensor->data.f, m_normalizedFrame.data, 
         m_normalizedFrame.total() * m_normalizedFrame.elemSize());

  // Run inference
  if (m_interpreter->Invoke() != kTfLiteOk)
  {
    spdlog::error("EngineLite::runObjectDetection: failed to invoke interpreter!");
    return nullptr;
  }

  return m_outputTensor->data.f;
}


bool EngineLite::runObjectDetection(const cv::Mat& frame)
{
  float* outputData{runInference(frame)};
  /*
  // Clear previous detections
  m_objsDetected.m_classProbabilities.clear();
  m_objsDetected.m_xOnes.clear();
  m_objsDetected.m_yOnes.clear();
  m_objsDetected.m_xTwos.clear();
  m_objsDetected.m_yTwos.clear();
  m_objsDetected.m_classNameIdxs.clear();

  // Dispatch to the correct post-processing method based on model architecture
  switch (m_models.m_archs[modelIdx])
  {
    case ModelArch::SSD:
      //return ssdPostProc(, input.m_frame.cols, input.m_frame.rows);
      return true;
    case ModelArch::YOLO5:
      //return runYoloPostProc(modelIdx, input.m_frame.cols, input.m_frame.rows);
      return true;
    default:
      spdlog::error("EngineLite::runObjectDetection: Unknown model architecture for post-processing!");
      return false;
  }
  */
  return true;
}

bool EngineLite::runSemanticDetection(const cv::Mat& frame)
{
  // TODO
  return true;
}
