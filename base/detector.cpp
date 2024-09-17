#include <iostream>

#include "tensorflow/lite/kernels/register.h"

#include "inference.hpp"

Inference::Inference(const std::string& model_name){
    // load the model into memory
    LoadModel(model_name);
}

void Inference::LoadModel(const std::string& model_name){
    using std::runtime_error, std::cout, std::string;

    const string path = MODELS_DIR + model_name;

    // load the model into memory
    _model = tflite::FlatBufferModel::BuildFromFile(path.c_str());
    if(_model == nullptr){
        throw runtime_error("LoadModel: ERROR: could not load the model from: " + path); 
    }

    // creating model interpreter
    tflite::InterpreterBuilder(*_model, tflite::ops::builtin::BuiltinOpResolver())(&_interpreter);
    if(_interpreter == nullptr){
        throw runtime_error("LoadModel: ERROR: could not create tflite interpreter.");
    }

    // update all allocattions for tensor buffers
    if(_interpreter->AllocateTensors() != kTfLiteOk){
        throw runtime_error("LoadModel: ERROR: failed to update allocations for tensors.");
    }

    cout << "LoadModel: LOG: " << model_name << " successfully loaded.\n";
}