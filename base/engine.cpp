#include <../libs/spdlog/spdlog.h>

#include "engine.h"

Engine::Engine(const std::string& configPath)
{
    parseConfig(configPath);      // load the model into memory
}

void Engine::parseConfig(const std::string& parseConfig)
{
    // TODO
}

bool Engine::loadModel(const char* modelPath, const char* classNamePath, float confidence, float iou)
{
  if (!loadFlatBuffer(modelPath, confidence, iou))
  {
    return false;
  }

  if (!loadClassNames(classNamePath))
  {
    return false;
  }
  
  return true;
}

bool Engine::loadFlatBuffer(const char* model, float confidence, float iou)
{
  spdlog::info("Engine::loadFlatBufferPtr: loading {}", path);

  // add flatbuffer ptr to models container
  tflite::FlatBufferModel* flatBufferPtr {tflite::FlatBufferModel::BuildFromFile(path)};
  if(flatBufferPtr == nullptr)
  {
    spdlog::error("Engine::loadFlatBuffer: could not load the flat buffer model!"); 
    return false;
  }
  m_models.m_flatBufferPtrs.push_back(flatBufferPtr);
  
  // add interpreter ptr to models container
  tflite::Interpreter* interpreterPtr;
  tflite::InterpreterBuilder(flatBufferPtr, tflite::ops::builtin::BuiltinOpResolver())(&interpreterPtr);
  if(interpreterPtr == nullptr)
  {
    spdlog::error("Engine::loadFlatBuffer: could not create interpreter!");
    return false;
  }
  if(interpreterPtr->AllocateTensors() != kTfLiteOk)    // update all allocation for io tensors
  {
    spdlog::error("Engine::loadFlatBuffer: failed to update allocations for IO tensors.");
    return false;
  }
  m_models.m_interpreters.push_back(interpreterPtr);

  // update the pointers to IO tensors of interpreter
  const int inTensorIdx {interpreterPtr->inputs()[0]};
  m_models.m_inputTensorPtrs.push_back(interpreterPtr->tensor(inTensorIdx));
  const int outTensorIdx {m_interpreterPtr->outputs()[0]};
  m_models.m_outputTensorPtrs.push_back(interpreterPtr->tensor(outTensorIdx));

  // add confidence and iou to container
  m_models.m_confidences.push_back(confidence);
  m_models.m_IoUs.push_back(iou);
  
  return true;
}

bool Engine::loadClassNames(const char* path)
{
  return true;
}
