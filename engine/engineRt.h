#pragma once

#include "../base.h"

struct EngineRt : public EngineBase
{
  bool runObjectDetection(const Input& input) override;
  bool runSemanticDetection(const Input& input) override;
  bool parseConfig(const char* path) override;
  
  // TODO implement other functions 
private:
  DetectedObjects m_objsDetected;
  DetectedSemantics m_semanticsDetected;
};
