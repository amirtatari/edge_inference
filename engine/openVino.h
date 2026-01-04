#pragma once

#include "base.h"

struct EngineVino : public AbsEngine
{
  bool runObjectDetection(const cv::Mat& frame);
  bool runSemanticDetection(const cv::Mat& frame);
  bool loadModel(const std::string& path);
};
