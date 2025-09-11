#include <iostream>

#include "engine.hpp"

Engine::Engine(const std::string& configPath)
{
    parseConfig(configPath);      // load the model into memory
}

void Engine::parseConfig(const std::string& parseConfig)
{
    // TODO
}