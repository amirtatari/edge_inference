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
  float* outputData {runInference(frame)};
  if (outputData == nullptr)
  {
    spdlog::error("EngineLite::runObjectDetection: inference failed");
    return false;
  }
  m_numBoxes = m_outputTensor->dims->data[1];
  switch (m_config.m_arch)
  {
    case ModelArch::YOLO5:
      return yoloFivePostProc(outputData, frame.cols, frame.rows);
    case ModelArch::YOLOV8:
      return yoloEightPostProc(outputData, frame.cols, frame.rows);
    case ModelArch::YOLO10:
      return yoloTenPostProc(outputData, frame.cols, frame.rows);
    case ModelArch::SSD:
      return ssdPostProc(outputData, frame.cols, frame.rows);
    default:
      spdlog::error("EngineLite::runObjectDetection: unsupported architecture");
      return false;
  }
}

bool EngineLite::runSemanticDetection(const cv::Mat& frame)
{
  float* outputData {runInference(frame)};
  // TODO post processing for semantic segmentation
  return true;
}
