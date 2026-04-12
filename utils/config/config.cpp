#include "config.hpp"

#include <spdlog/spdlog.h>
#include <algorithm> 
#include <cctype> 
#include <optional>

// helper function to convert string to TestBenchType enum
inline std::optional<TaskType> stringToTaskType(std::string& type_str) 
{
    if (type_str.empty()) {
        spdlog::error("stringToTestBenchType: empty type string!");
        return std::nullopt;
    }

    std::transform(type_str.begin(), type_str.end(), type_str.begin(), ::tolower);

    if (type_str == "object_detection") return TaskType::OBJECT_DETECTION;
    if (type_str == "semantic_segmentation") return TaskType::SEMANTIC_SEGMENTATION;

    return std::nullopt;
}

void Config::parseEngineNode(const pugi::xml_node& engineNode)
{
  if (!engineNode)
    throw std::runtime_error("parseEngineNode: Missing <engine> node!");

  const pugi::xml_node modelPathNode {engineNode.child("modelPath")};
  if (!modelPathNode)
    throw std::runtime_error("parseEngineNode: Missing <modelPath> node!");
  m_modelPath = modelPathNode.attribute("value").as_string();

  const pugi::xml_node classesPathNode {engineNode.child("classesPath")};
  if (!classesPathNode)
    throw std::runtime_error("Config::parseEngineNode: Missing <classesPath> node!");
  m_classNamesPath = classesPathNode.attribute("value").as_string();

  const pugi::xml_node iouNode = engineNode.child("iou");
  if (!iouNode)
    throw std::runtime_error("Config::parseEngineNode: Missing <iou> node!");
  m_iouThreshold = iouNode.attribute("value").as_float();

  const pugi::xml_node confidenceNode = engineNode.child("confidence");
  if (!confidenceNode)
    throw std::runtime_error("Config::parseEngineNode: Missing <confidence> node!");
  m_confidenceThreshold = confidenceNode.attribute("value").as_float();
}

void Config::parseRootNode(const pugi::xml_node& rootNode)
{
  if (!rootNode)
    throw std::runtime_error("Config::parseRootNode: Missing <Configs> root node!");

  const pugi::xml_node typeNode {rootNode.child("taskType")};
  if (!typeNode)
    throw std::runtime_error("Config::parseRootNode: Missing <type> node in {}");

  std::string taskTypeStr {typeNode.attribute("value").as_string()};
  const auto res = stringToTaskType(taskTypeStr);
  if (res.has_value())
    m_taskType = res.value();
  else
  {
      throw std::runtime_error("Config::parseRootNode: Unknown test bench type: " +
        std::string(typeNode.attribute("value").as_string()));
  }

  const pugi::xml_node datasetDirNode = rootNode.child("datasetDir");
  if (!datasetDirNode)
    throw std::runtime_error("Config::parseRootNode: Missing <datasetDir> node!");
  m_datasetDir = datasetDirNode.attribute("value").as_string();
}

void Config::parseFromFile(const std::string& path)
{
  try
  {
    pugi::xml_document doc;
    pugi::xml_parse_result result {doc.load_file(path.c_str())};

    if (!result)
      throw std::runtime_error(result.description());

    pugi::xml_node root {doc.child("config")};
  
    // parse the test bench configs node
    parseRootNode(root);
    parseEngineNode(root.child("engine"));
  } catch(const std::exception& e)
  {
    spdlog::error("Config: Exception while parsing config file: {}", 
      e.what());
  }
}

