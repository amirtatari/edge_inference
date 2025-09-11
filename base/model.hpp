#pragma once

#include <string>
#include <memory>
#include <tensorflow/lite/model.h>
#include <vector>

#define MODELS_DIR "/etc/inference/models/"

/*
Model structure holds the pointers to flatbuffer model and the the interpreter
*/
struct Model
{
    explicit Model(const std::string& modelName);
    Model(Model&& rhs) noexcept;
    Model& operator=(Model&& rhs) noexcept;

    // copy ctor and assignment operator are deleted
    Model(const Model& rhs) = delete;
    Model& operator=(const Model& rhs) = delete;

    std::vector<std::string> m_classes;                                     // class names
    std::string m_name;                                                     // model name
    std::unique_ptr<tflite::FlatBufferModel> m_faltBufferModelPtr;          // flatbuffer modfel
    std::unique_ptr<tflite::Interpreter> m_interpreterPtr;                  // tflite interpreter
    TfLiteTensor* m_inputTensorPtr;                         // input tensor
    TfLiteTensor* m_outputTensorPtr;                         // output tensor

private:
    // load the flatbuffer models from file
    void loadFlatBufferModel();

    // read and add class names from file
    void setClasses();
};