#pragma once

#include <memory>
#include <tensorflow/lite/model.h>

#include "../base.h"

/*
class Engine loads the models and runs the inference on the input frame
*/
struct EngineLite : public EngineBase
{
  bool runObjectDetection(const Input& input) override;
  bool runSemanticDetection(const Input& input) override;
  bool parseConfig(const char* configPath) override;

private:
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
  bool loadModel(const char* modelPath, const char* classNamePath, const char* modelType, float confidence, float iou);
  void parseConfig(const std::string& configPath);
  bool loadFlatBuffer(const char* model, const char* modelType, float confidence, float iou);
  bool loadClassNames(const char* path);

  Models m_models;
  DetectedObjects m_objsDetected;
  DetectedSemantics m_semanticsDetected;
};
