#pragma once

#include "../engine/tfLite.hpp"
#include "../utils/config/config.hpp"

class AbsTask
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
   * @brief loads the test dataset from the specified directory
   * @param path path to the dataset directory
   * @return vector of cv::Mat containing the loaded dataset frames
   */
  std::vector<cv::Mat> loadDataset(const std::string& path);
public:
  /**
   * @brief runs the benchmark for the given engine type and dataset
   * @param config ptr to testbench config
   * @return true if successful, false otherwise
   */
  bool runModelBenchmark(Config* config);

  ~AbsTask() = default;
};

class ObjectDetection : public AbsTask
{
  void evaluateOutput(AbsEngine* engine);
  void runInference(AbsEngine* engine, const cv::Mat& frame);
};

class SemanticSegmentation : public AbsTask
{
  void evaluateOutput(AbsEngine* engine);
  void runInference(AbsEngine* engine, const cv::Mat& frame);
};

class TaskFactory
{
  Config m_config;                      /// \var test bench configuration

  /**
   * @brief creates and returns a task instance based on the specified type
   * @param type type of the test bench
   */
  std::optional<std::unique_ptr<AbsTask>> getTask(TaskType type) const;

  /**
   * @brief starts the test bench with the given configuration file
   * @param path path to the test bench configuration file
   */
  void start(const std::string& path);

public:
  explicit TaskFactory(const std::string& configPath);

  TaskFactory(const TaskFactory&) = delete;
  TaskFactory(TaskFactory&&) = delete;
  TaskFactory& operator=(const TaskFactory&) = delete;
  TaskFactory& operator=(TaskFactory&&) = delete;
};