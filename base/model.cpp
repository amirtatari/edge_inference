#include "model.hpp"

#include <tensorflow/lite/kernels/register.h>
#include <iostream>
#include <stdexcept>

Model::Model(const std::string& modelName)
    : m_name {modelName}
    , m_faltBufferModelPtr {nullptr}
    , m_interpreterPtr {nullptr}
    , m_inputTensorPtr {nullptr}
    , m_outputTensorPtr {nullptr}
{
    loadFlatBufferModel();
    setClasses();               
}

Model::Model(Model&& rhs) noexcept
    : m_classes {std::move(rhs.m_classes)}
    , m_name {std::move(rhs.m_name)}
    , m_faltBufferModelPtr {std::move(rhs.m_faltBufferModelPtr)}
    , m_interpreterPtr {std::move(rhs.m_interpreterPtr)}
    , m_inputTensorPtr {std::move(rhs.m_inputTensorPtr)}
    , m_outputTensorPtr {std::move(rhs.m_outputTensorPtr)}
{}

Model& Model::operator=(Model&& rhs) noexcept
{
    if (this == &rhs)
    {
        return *this;
    }

    m_classes = std::move(rhs.m_classes);
    m_name = std::move(rhs.m_name);
    m_faltBufferModelPtr = std::move(rhs.m_faltBufferModelPtr);
    m_interpreterPtr = std::move(rhs.m_interpreterPtr);
    m_inputTensorPtr = std::move(rhs.m_inputTensorPtr);
    m_outputTensorPtr = std::move(rhs.m_outputTensorPtr);
    return *this;
}

void Model::loadFlatBufferModel()
{
    const std::string fullPath {MODELS_DIR + m_name};

    // load the model into memory
    m_faltBufferModelPtr = std::unique_ptr<tflite::FlatBufferModel>(tflite::FlatBufferModel::BuildFromFile(fullPath.c_str()));
    if(m_faltBufferModelPtr == nullptr)
    {
        throw std::runtime_error("Model::loadFlatBufferModel: could not load the flat buffer model at: " + fullPath); 
    }

    // creating model interpreter
    tflite::InterpreterBuilder(*m_faltBufferModelPtr, tflite::ops::builtin::BuiltinOpResolver())(&m_interpreterPtr);
    if(m_interpreterPtr == nullptr)
    {
        throw std::runtime_error("Model::loadFlatBufferModel: could not create tflite interpreter.");
    }

    if(m_interpreterPtr->AllocateTensors() != kTfLiteOk)    // update all allocation for io tensors
    {
        throw std::runtime_error("Model::loadFlatBufferModel: failed to update allocations for io tensors.");
    }

    // update the pointers to IO tensors of interpreter
    const int inTensorIdx {m_interpreterPtr->inputs()[0]};
    //m_inputTensorPtr = std::make_unique<TfLiteTensor>(*m_interpreterPtr->tensor(inTensorIdx));
    m_inputTensorPtr = std::unqiue_ptr<TfLiteTensor>(m_interpreterPtr->tensor(inTensorIdx));
    const int outTensorIdx {m_interpreterPtr->outputs()[0]};
    m_outputTensorPtr = std::make_unique<TfLiteTensor>(*m_interpreterPtr->tensor(outTensorIdx));

    std::cout << "loadFlatBufferModel: successfully loaded the model: " << m_name << '\n';
}

void Model::setClasses()
{
    // TODO
}