#include "config.h"

#include <spdlog/spdlog.h>
#include <algorithm> 
#include <cctype> 

// helper function to convert string to EngineType enum
static EngineType stringToEngineType(const std::string& type_str) {
    std::string lower_type_str {type_str};
    std::transform(lower_type_str.begin(), lower_type_str.end(), lower_type_str.begin(), 
      ::tolower);

    if (lower_type_str == "tflite") return EngineType::TFLITE;
    if (lower_type_str == "openvino") return EngineType::OPENVINO;
    if (lower_type_str == "tensorrt") return EngineType::TENSORRT;
    return EngineType::UNKNOWN;
}

// helper function to convert string to TestBenchType enum
static TestBenchType stringToTestBenchType(const std::string& type_str) {
    std::string lower_type_str = type_str;
    std::transform(lower_type_str.begin(), lower_type_str.end(), lower_type_str.begin(), 
      ::tolower);

    if (lower_type_str == "object_detection") return TestBenchType::OBJECT_DETECTION;
    if (lower_type_str == "semantic_segmentation") return TestBenchType::SEMANTIC_SEGMENTATION;
    return TestBenchType::UNKNOWN;
}

bool TestBenchConfig::parseEngineNode(const pugi::xml_node& engineNode)
{
  if (!engineNode)
  {
    spdlog::error("TestBenchConfig::parseEngineNode: Missing <engine> node!");
    return false;
  }

  pugi::xml_node modelPathNode = engineNode.child("modelPath");
  if (!modelPathNode)
  {
    spdlog::error("TestBenchConfig::parseEngineNode: Missing <modelPath> node!");
    return false;
  }
  m_modelPath = modelPathNode.attribute("value").as_string();

  pugi::xml_node classesPathNode = engineNode.child("classesPath");
  if (!classesPathNode)
  {
    spdlog::error("TestBenchConfig::parseEngineNode: Missing <classesPath> node!");
    return false;
  }
  m_classNamesPath = classesPathNode.attribute("value").as_string();

  pugi::xml_node iouNode = engineNode.child("iou");
  if (!iouNode)
  {
    spdlog::error("TestBenchConfig::parseEngineNode: Missing <iou> node!");
    return false;
  }
  m_iouThreshold = iouNode.attribute("value").as_float();

  pugi::xml_node confidenceNode = engineNode.child("confidence");
  if (!confidenceNode)
  {
    spdlog::error("TestBenchConfig::parseEngineNode: Missing <confidence> node!");
    return false;
  }
  m_confidenceThreshold = confidenceNode.attribute("value").as_float();
  return true;
}

bool TestBenchConfig::parseTestBenchConfigsNode(const pugi::xml_node root)
{
  if (!root)
  {
    spdlog::error("TestBenchConfig::parseTestBenchConfigsNode: Missing <testBenchConfigs> root node!");
    return false;
  }

  pugi::xml_node typeNode {root.child("type")};
  if (!typeNode)
  {
    spdlog::error("TestBenchConfig::parseTestBenchConfigsNode: Missing <type> node in {}");
    return false;
  }
  m_benchType = stringToTestBenchType(typeNode.attribute("value").as_string());
  if (m_benchType == TestBenchType::UNKNOWN) {
      spdlog::error("TestBenchConfig::parseTestBenchConfigsNode: Unknown test bench type: {}", 
        typeNode.attribute("value").as_string());
      return false;
  }

  pugi::xml_node engineTypeNode {root.child("engineType")};
  if (!engineTypeNode)
  {
    spdlog::error("TestBenchConfig::parseTestBenchConfigsNode: Missing <engineType> node!");
    return false;
  }
  m_engineType = stringToEngineType(engineTypeNode.attribute("value").as_string());
  if (m_engineType == EngineType::UNKNOWN) {
      spdlog::error("TestBenchConfig::parseTestBenchConfigsNode: Unknown engine type: {}", 
        engineTypeNode.attribute("value").as_string());
      return false;
  }

  pugi::xml_node datasetDirNode = root.child("datasetDir");
  if (!datasetDirNode)
  {
    spdlog::warn("TestBenchConfig::parseTestBenchConfigsNode: Missing <datasetDir> node!");
    return false;  
  }
  m_datasetDir = datasetDirNode.attribute("value").as_string();
  return true;
}


bool TestBenchConfig::parseConfigFile(const std::string& path)
{
  pugi::xml_document doc;
  pugi::xml_parse_result result {doc.load_file(path.c_str())};
  if (!result)
  {
    spdlog::error("TestBenchConfig::parseConfigFile: XML [{}] parsed with errors, "
                  "description: {}, offset: {}", path, result.description(), result.offset);
    return false;
  }
  pugi::xml_node root {doc.child("testBenchConfigs")};
  
  // parse the test bench configs node
  if (!parseTestBenchConfigsNode(root))
    return false;
  
  // parse engine node
  return parseEngineNode(root.child("engine"));
}

