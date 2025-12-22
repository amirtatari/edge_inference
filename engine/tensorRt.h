#pragma once

#include "base.h"

class EngineRt : public AbsEngine
{
  public:
  bool runObjectDetection(const cv::Mat& frame);
  bool runSemanticDetection(const cv::Mat& frame);
  bool loadModel(const std::string& path);
};
