#include "testBench.hpp"

#include <spdlog/spdlog.h>

TaskFactory::TaskFactory(const std::string& configPath)
{
  start(configPath);
}

void TaskFactory::start(const std::string& path)
{
  if (path.empty())
    throw std::runtime_error("TaskFactory::start: empty config file path!");

  m_config.parseFromFile(path);

  const auto taskRes {getTask(m_config.m_taskType)};
  if (!taskRes.has_value())
    throw std::runtime_error("TaskFactory::start: could not create test bench instance!");
    
  taskRes.value()->runModelBenchmark(&m_config);
}

std::optional<std::unique_ptr<AbsTask>> TaskFactory::getTask(TaskType type) const
{
  switch (type)
  {
    case TaskType::OBJECT_DETECTION:
      return std::make_unique<ObjectDetection>();
    case TaskType::SEMANTIC_SEGMENTATION:
      return std::make_unique<SemanticSegmentation>();
    default:
      spdlog::error("TaskFactory::getTestBench: Unknown test bench type!");
      return std::nullopt;
  }
}

bool AbsTask::runModelBenchmark(Config* config)
{
  EngineLite engine;
  const std::vector<cv::Mat>& dataset {loadDataset(config->m_datasetDir)};
  if (dataset.empty())
  {
    spdlog::error("AbsTask::runModelBenchmark: could not load dataset from path: {}", 
                   config->m_datasetDir);
    return false;
  }

  if (!engine.init(config))  // initialize the engine parameters
  {
    spdlog::error("start: Engine initialization failed!");
    return false;
  }

  for (auto&& frame : dataset)
    runInference(&engine, frame);
  
  evaluateOutput(&engine);

  return true;
}

std::vector<cv::Mat> AbsTask::loadDataset(const std::string& path)
{
  // TODO
  return std::vector<cv::Mat>{};
}

void ObjectDetection::runInference(AbsEngine* engine, const cv::Mat& frame)
{
  if (!engine->runObjectDetection(frame))
    spdlog::error("ObjectDetection::runInference: Inference failed!");
  
}

void ObjectDetection::evaluateOutput(AbsEngine* engine)
{
  // TODO
  spdlog::info("ObjectDetection::validateOutput: test!");
}

void SemanticSegmentation::runInference(AbsEngine* engine, const cv::Mat& frame)
{
  if (!engine->runSemanticDetection(frame))
    spdlog::error("SemanticSegmentation::runInference: Inference failed!");
  

}

void SemanticSegmentation::evaluateOutput(AbsEngine* engine)
{
  // TODO
  spdlog::info("SemanticSegmentation::validateOutput: test!");
}


