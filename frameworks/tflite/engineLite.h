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
   * @brief defines the data structure of a tensorflow model
   */
  struct ModelData
  {
    std::unique_ptr<tflite::FlatBufferModel> m_flatBufferPtr {nullptr};
    std::unique_ptr<tflite::Interpreter> m_interPreterPtr {nullptr};
    TfliteTensor* m_inputTensorPtr {nullptr};
    TfliteTensor* m_outputTensorPtr {nullptr};
    int m_width {0};
    int m_height {0};
    int m_inputChannels {0};
    float m_confidenceScore {0.0f};
    float m_iou {0.0f};
    ModelArch m_arch {ModelArch::YOLO5};
  };
  
  /**
   * @brief Models represents data structure of tflite models
   */
  struct ModelsData
  {
    std::vector<std::vector<const char*>> m_classNames;                    /// \var vector of class names
    std::vector<std::unique_ptr<tflite::FlatBufferModel>> m_flatBufferPtrs; /// \var vector of flat buffer ptrs
    std::vector<std::unique_ptr<tflite::Interpreter>> m_interpreterPtrs;   /// \var vector of interpreter ptrs
    std::vector<TfLiteTensor*> m_inputTensorPtrs;                          /// \var vector of input tensor ptrs
    std::vector<TfLiteTensor*> m_outputTensorPtrs;                         /// \var vector of output tensor ptrs
    std::vector<int> m_widths;                                             /// \var vector of model's widths
    std::vector<int> m_height;                                             /// \var vector of model's heights
    std::vector<int> m_inputChannels;                                      /// \var vector of model's number of input channels
    std::vector<float> m_confidenceScores;                                 /// \var vector of model's confidence threshold
    std::vector<float> m_ious;                                             /// \var vector of model's IoU threshold
    std::vector<ModelArch> m_archs;                                        /// \var vector of model's architecture  
  };


  void parseConfig(const std::string& configPath);

  bool loadModel(const char* modelPath, const char* classNamePath,
                 const char* modelType, float confidence, float iou);
  
  bool loadTensorFlowModel(ModelData& model, const char* modelPath,
                           const char* modelType, float confidence,
                           float iou);
  
  bool loadClassNames(std::vector<const char*>& classNames, const char* path);

  void addLoadedModelsEntry(const ModelData& model, 
                          const std::vector<const char*>& classNames);

  LoadedModels m_models;
  DetectedObjects m_objsDetected;
  DetectedSemantics m_semanticsDetected;
};
