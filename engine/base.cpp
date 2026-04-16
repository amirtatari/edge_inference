#include "../../utils/profiler/profiler.hpp"
#include "postProcess.hpp"

// liteRt
#include <tensorflow/lite/interpreter_builder.h>
#include <tensorflow/lite/kernels/register.h>

// opencv
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// logging
#include <spdlog/spdlog.h>

// stl
#include <algorithm>
#include <numeric>
#include <fstream>
#include <stdexcept>

LiteRtEngine::LiteRtEngine(ModelArch arch, const std::string& modelPath, cosnt std::string& classesPath)
  : m_flatBufferModel{nullptr}
  , m_interpreter{nullptr}
  , m_inputTensor{nullptr}
  , m_outputTensor{nullptr}
{
  loadModel(modelPath);
  loadClassNames(classesPath);
}

void LiteRtEngine::loadClassNames(const std::string& path)
{
  spdlog::info("LiteRtEngine::loadClassNames: loading classes from {}", path);

  std::ifstream file(path);
  if (!file.is_open())
    throw std::runtime_error("AbsEngine::loadClassNames: could not open file " + path);

  std::string line;
  while (std::getline(file, line))
  {
    // remove any trailing whitespace
    line.erase(line.find_last_not_of(" \n\r\t") + 1);
    if (!line.empty())
      m_classNames.push_back(line);
  }
  file.close();
  if (m_calssNames.empty) 
    throw std::runtime_error("AbsEngine::loadClassNames: no class name entries.")
}

void LiteRtEngine::loadModel(const std::string& path)
{
  spdlog::info("LiteRtEngine::loadModel: loading model from {}", path);

  // load model from file
  m_flatBufferModel = tflite::FlatBufferModel::BuildFromFile(path.c_str());
  if (m_flatBufferModel == nullptr)
    throw std::runtime_error("LiteRtEngine::loadModel: failed to build model from file!");

  // build the interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*m_flatBufferModel, resolver);
  if (builder(&m_interpreter) != kTfLiteOk)
    throw std::runtime_error("LiteRtEngine::loadModel: failed to build interpreter");

  if (m_interpreter->AllocateTensors() != kTfLiteOk)
    throw std::runtime_error("LiteRtEngine::loadModel: failed to allocate tensors");

  m_inputTensor = m_interpreter->tensor(m_interpreter->inputs()[0]);
  m_outputTensor = m_interpreter->tensor(m_interpreter->outputs()[0]);
}

float* LiteRtEngine::runInference(const cv::Mat& frame)
{
  const int height {m_inputTensor->dims->data[1]};
  const int width {m_inputTensor->dims->data[2]};
  cv::Mat inputWrapper(height, width, CV_32FC3, m_interpreter->typed_input_tensor<float>(0));

  // check the frame size with model size
  if (frame.cols == width && frame.rows == height)
        frame.convertTo(inputWrapper, CV_32FC3, 1.0f / 255.0f);
  else 
  {
    static cv::Mat staticResized; 
    cv::resize(frame, staticResized, cv::Size(width, height));
    staticResized.convertTo(inputWrapper, CV_32FC3, 1.0f / 255.0f);
  }

  // run inference
  return m_interpreter->Invoke() != kTfLiteOk ? nullptr : m_outputTensor->data.f;
}

std::optional<modelIo::DetectedObjects> LiteRtEngine::runObjectDetection(const cv::Mat& frame)
{
  // run inference and get the output
  float* outputData {runInference(frame)};
  if (outputData == nullptr) return std::nullopt;
  const int numBoxes {m_outputTensor->dims->data[1]};

  // post process methods
  switch (m_arch)
  {
    using enum ModelArch;
    case YOLO5:
      return PostProcess::yoloFivePostProc(numBoxes, outputData, frame.cols, frame.rows);

    case YOLOV8:
      return PostProcess::yoloEightPostProc(numBoxes, outputData, frame.cols, frame.rows);

    case YOLO10:
      return PostProcess::yoloTenPostProc(numBoxes, outputData, frame.cols, frame.rows);

    case SSD:
      return PostProcess::ssdPostProc(numBoxes, outputData, frame.cols, frame.rows);
  }

  return std::nullopt;
}

std::optional<modelIo::DetectedSemantics> LiteRtEngine::runSemanticDetection(const cv::Mat& frame)
{
  float* outputData {runInference(frame)};
  if (outputData == nullptr) return std::nullopt;

  // get output tensor dimensions
  const int outH {m_outputTensor->dims->data[1]};
  const int outW {m_outputTensor->dims->data[2]};
  const int numClasses {m_outputTensor->dims->data[3]};

  semanticPostProc(outputData, outW, outH, numClasses, frame.cols, frame.rows);

  return true;
}