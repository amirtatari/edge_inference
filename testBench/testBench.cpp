#include "testBench.h"

#include <spdlog/spdlog.h>

bool TestBenchFactory::start(const std::string& path)
{
  if (!m_config.parseConfigFile(path))
  {
    spdlog::error("TestBenchFactory::start: could not parse config file: {}", path);
    return false;
  }

  std::unique_ptr<AbsTestBench> testBench {getTestBench(m_config.m_benchType)};
  if (testBench == nullptr)
  {
    spdlog::error("TestBenchFactory::start: could not create test bench instance!");
    return false;
  }
    
  return testBench->runModelBenchmark(m_config.m_engineType, 
                                      m_config.m_datasetDir,
                                      m_config.m_engineConfigPath);
}

std::unique_ptr<AbsTestBench> TestBenchFactory::getTestBench(TestBenchType type)
{
  switch (type)
  {
    case TestBenchType::OBJECT_DETECTION:
      return std::make_unique<ObjectDetectionBench>();
    case TestBenchType::SEMANTIC_SEGMENTATION:
      return std::make_unique<SemanticSegmentationBench>();
    default:
      spdlog::error("TestBenchFactory::getTestBench: Unknown test bench type!");
      return nullptr;
  }
}

bool AbsTestBench::runModelBenchmark(EngineType type, 
                                    const std::string& datasetPath,
                                    const std::string& engineConfigPath)
{
  std::unique_ptr<AbsEngine> engine {getEngine(type)};
  if (engine == nullptr) 
  {
    spdlog::error("AbsTestBench::runModelBenchmark: could not create engine instance!");
    return false;
  }
  const std::vector<cv::Mat>& dataset {loadDataset(datasetPath)};
  if (dataset.empty())
  {
    spdlog::error("AbsTestBench::runModelBenchmark: could not load dataset from path: {}", 
                   datasetPath);
    return false;
  }

  if (!engine->init(engineConfigPath))  // initialize the engine parameters
  {
    spdlog::error("start: Engine initialization failed!");
    return false;
  }

  for (const auto& frame : dataset)
    runInference(engine.get(), frame);
  
  evaluateOutput(engine.get());

  return true;
}

std::unique_ptr<AbsEngine> AbsTestBench::getEngine(EngineType type)
{
  switch (type)
  {
    case EngineType::TFLITE:
      return std::make_unique<EngineLite>();
    case EngineType::OPENVINO:
      return std::make_unique<EngineVino>();
    case EngineType::TENSORRT:
      return std::make_unique<EngineRt>();
    default:
      spdlog::error("TestBench::getEngine: Unknown engine type!");
      return nullptr;
  }
}

std::vector<cv::Mat> AbsTestBench::loadDataset(const std::string& path)
{
  // TODO
  return std::vector<cv::Mat>{};
}

void ObjectDetectionBench::runInference(AbsEngine* engine, const cv::Mat& frame)
{
  if (!engine->runObjectDetection(frame))
    spdlog::error("ObjectDetectionBench::runInference: Inference failed!");
  
}

void ObjectDetectionBench::evaluateOutput(AbsEngine* engine)
{
  // TODO
  spdlog::info("ObjectDetectionBench::validateOutput: test!");
}

void SemanticSegmentationBench::runInference(AbsEngine* engine, const cv::Mat& frame)
{
  if (!engine->runSemanticDetection(frame))
    spdlog::error("SemanticSegmentationBench::runInference: Inference failed!");
  

}

void SemanticSegmentationBench::evaluateOutput(AbsEngine* engine)
{
  // TODO
  spdlog::info("SemanticSegmentationBench::validateOutput: test!");
}


