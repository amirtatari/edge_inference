#pragma once

#include "model.hpp"

#include <vector>

/*
class Engine loads the models and runs the inference on the input frame
*/
class Engine 
{
    std::vector<Model> m_models;

protected:
    // loads the model into memory
    void parseConfig(const std::string& configPath);

public:
    explicit Engine(const std::string& configPath);
};