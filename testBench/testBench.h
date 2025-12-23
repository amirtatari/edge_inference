#pragma once

#include "../engine/tfLite.h"
#include "../engine/openVino.h"
#include "../engine/tensorRt.h"

class AbsTestBench
{
protected:
  /**
   * @brief evaluates the inference output with the expected results 
   * @param engine pointer to the inference engine   
   */
  virtual void evaluateOutput(AbsEngine* engine) = 0;

  /**
   * @brief runs inference on the given frame using the provided engine
   * @param engine pointer to the inference engine
   * @param frame input frame for inference
   */
  virtual void runInference(AbsEngine* engine, const cv::Mat& frame) = 0;

  /**
   * @brief creates and returns an inference engine based on the specified type
   * @param type type of the inference engine
   */
  std::unique_ptr<AbsEngine> getEngine(EngineType type);

  /**
   * @brief loads the test dataset from the specified directory
   * @param path path to the dataset directory
   * @return vector of cv::Mat containing the loaded dataset frames
   */
  std::vector<cv::Mat> loadDataset(const std::string& path);
public:
  /**
   * @brief runs the benchmark for the given engine type and dataset
   * @param type type of the inference engine
   * @param datasetPath path to the dataset directory
   * @param engineConfigPath path to the engine configuration file
   * @return true if successful, false otherwise
   */
  bool runModelBenchmark(EngineType type, 
                         const std::string& datasetPath,
                         const std::string& engineConfigPath);

  ~AbsTestBench() = default;
};

class ObjectDetectionBench : public AbsTestBench
{
  void evaluateOutput(AbsEngine* engine);
  void runInference(AbsEngine* engine, const cv::Mat& frame);
};

class SemanticSegmentationBench : public AbsTestBench
{
  void evaluateOutput(AbsEngine* engine);
  void runInference(AbsEngine* engine, const cv::Mat& frame);
};


class TestBenchFactory
{
  TestBenchConfig m_config;                      /// \var test bench configuration

  /**
   * @brief creates and returns a test bench instance based on the specified type
   * @param type type of the test bench
   */
  std::unique_ptr<AbsTestBench> getTestBench(TestBenchType type);
public:
  /**
   * @brief starts the test bench with the given configuration file
   * @param path path to the test bench configuration file
   * @return true if successful, false otherwise
   */
  bool start(const std::string& path);
};