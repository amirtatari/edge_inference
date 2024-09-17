#ifndef INFERENCE_HPP
#define INFERENCE_HPP

#include <string>
#include <memory>

#include <tensorflow/lite/model.h>

#define MODELS_DIR "/etc/inference/models/"

class Inference {
    std::unique_ptr<tflite::FlatBufferModel> _model;
    std::unique_ptr<tflite::Interpreter> _interpreter;
protected:
    // loads the model into memory
    void LoadModel(const std::string& model_name);
public:
    Inference(const std::string& model_name);
};

#endif // INFERENCE_HPP