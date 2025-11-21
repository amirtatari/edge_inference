#include <opencv2/imgproc.hpp>

#include <fstream>
#include <string>
#include <tensorflow/lite/op_resolver.h>
#include <tensorflow/lite/kernels/register.h>

#include "engineLite.h"
#include "../../utils/profiler/profiler.h"

bool EngineLite::parseConfig(const char* configPath)
{
    // TODO
  return true;
}

bool EngineLite::loadModel(const char* modelPath, const char* classNamePath,
                           const char* modelType, float confidence, float iou, int numClasses)
{
  // get flatbuffer model data
  FlatBufferModel model;
  if (!loadFlatBufferModel(model, modelPath, modelType, confidence, iou))
  {
    return false;
  }
  // get the class names
  std::vector<const char*> classNames;
  classNames.reserve(numClasses);
  if (!loadClassNames(classNames, classNamePath))
  {
    return false;
  }
  // add flatbuffer model and classNames to loaded models  
  addLoadedModelsEntry(model, classNames);                    
  
  return true;
}

bool EngineLite::loadTensorFlowModel(ModelData& model, const char* modelPath,
                                     const char* modelType, float confidence, float iou)
{
  spdlog::info("EngineLite::loadFlatBufferPtr: loading {}", modelPath);

  // add flatbuffer ptr to models container
  model.m_flatBufferPtr std::move(tflite::FlatBufferModel::BuildFromFile(model));
  if(model.m_flatBufferPtr == nullptr)
  {
    spdlog::error("EngineLite::loadFlatBuffer: could not load the flat buffer model!"); 
    return false;
  }
  
  // add interpreter ptr to models container
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreterPtr;
  tflite::InterpreterBuilder(*flatBufferPtr, resolver)(&interpreterPtr);
  if(interpreterPtr == nullptr)
  {
    spdlog::error("EngineLite::loadFlatBuffer: could not create interpreter!");
    return false;
  }
  if(interpreterPtr->AllocateTensors() != kTfLiteOk)    // update all allocation for io tensors
  {
    spdlog::error("EngineLite::loadFlatBuffer: failed to update allocations for IO tensors.");
    return false;
  }

  // update the pointers to IO tensors of interpreter
  const int inTensorIdx {interpreterPtr->inputs()[0]};
  m_models.m_inputTensorPtrs.push_back(interpreterPtr->tensor(inTensorIdx));
  const int outTensorIdx {interpreterPtr->outputs()[0]};
  m_models.m_outputTensorPtrs.push_back(interpreterPtr->tensor(outTensorIdx));

  m_models.m_flatBufferPtrs.push_back(std::move(flatBufferPtr));
  m_models.m_interpreterPtrs.push_back(std::move(interpreterPtr));

  // add confidence and iou to container
  m_models.m_confidences.push_back(confidence);
  m_models.m_IoUs.push_back(iou);

  // store model architecture
  if (strcmp(modelType, "SSD") == 0)
  {
    m_models.m_archs.push_back(ModelArch::SSD);
  }
  else if (strcmp(modelType, "YOLO") == 0)
  {
    m_models.m_archs.push_back(ModelArch::YOLO5);
  }
  else
  {
    spdlog::error("Unknown model architecture type: {}", modelType);
    return false;
  }
  return true;
}

bool EngineLite::loadClassNames(std::vector<const char*>& classNames, const char* path)
{
  std::ifstream file(path);
  if (!file.is_open())
  {
    spdlog::error("EngineLite::loadClassNames: cannot load the file at: " + path);
    return false;
  }

    std::string line;
    while (std::getline(file, line))
    {
        // Remove any trailing whitespace
        line.erase(line.find_last_not_of(" \n\r\t") + 1);
        if (!line.empty())
        {
          classNames.push_back(line);
        } 
    }
    file.close();
    
  return true;
}

bool EngineLite::runObjectDetection(const Input& input)
{
  const std::size_t modelIdx {input.m_modelIdx};

  // Get model input dimensions
  const int inputWidth {m_models.m_widths[modelIdx]};
  const int inputHeight {m_models.m_height[modelIdx]};

  // Preprocess the input frame
  cv::Mat frame;
  input.m_frame.copyTo(frame);
  cv::resize(frame, frame, cv::Size(inputWidth, inputHeight));

  frame.convertTo(frame, CV_32FC3, 2.0f / 255.0f, -1.0f); // Normalize to [-1, 1]

  // Copy data to input tensor
  float* inputTensorData {m_models.m_inputTensorPtrs[modelIdx]->data.f};
  memcpy(inputTensorData, frame.data, frame.total() * frame.elemSize());

  // Run inference
  if (m_models.m_interpreterPtrs[modelIdx]->Invoke() != kTfLiteOk)
  {
    spdlog::error("EngineLite::runObjectDetection: failed to invoke interpreter!");
    return false;
  }

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

  return true;
}

bool EngineLite::runSemanticDetection(const Input& input)
{
  // TODO
  return true;
}
