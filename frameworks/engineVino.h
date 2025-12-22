#pragma once

#include "../base.h"

struct EngineVino : public EngineBase
{
  bool runObjectDetection(const Input& input) override;
  bool runSemanticDetection(const Input& input) override;
  bool parseConfig(const char* path) override;
  
  // TODO implement EngineVino
private:
  DetectedObjects m_objsDetected;
  DetectedSemantics m_semanticsDetected;
};
