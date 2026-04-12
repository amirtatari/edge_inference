#include "../utils/config/config.hpp"

#include <gtest/gtest.h>
#include <fstream>

class ConfigTest : public ::testing::Test {
  protected:
    Config config;
    const std::string m_validConfigPath {"/workspace/configs/config.xml"};
    const std::string m_invalidConfigPath {"/workspace/configs/invalid_config.xml"};

    void SetUp() override {
        // Setup code if needed
    }

    void TearDown() override {
       std::remove(m_invalidConfigPath.c_str());
    }

    void createGarbageConfigFile(const std::string& content) {
        std::ofstream outfile(m_invalidConfigPath);
        outfile << "content";
        outfile.close();
    }
};

TEST_F(ConfigTest, ParseVaildConfigFile)
{
  EXPECT_NO_THROW(config.parseFromFile(m_validConfigPath));
}

TEST_F(ConfigTest, MissingFile)
{
  try
  {
    config.parseFromFile("non_existent_config.xml");
  } 
  catch(const std::exception& e)
  {
    EXPECT_STREQ(e.what(), "File was not found");
  }
}

TEST_F(ConfigTest, ParseInvalidConfigFile)
{
  // arange
  createGarbageConfigFile("This is not a valid XML config file");
  
  // act & assert
  EXPECT_THROW(config.parseFromFile(m_invalidConfigPath), std::runtime_error);
}

TEST_F(ConfigTest, MissingNodeEngine)
{
  // arange
  createGarbageConfigFile(
    "<?xml version='1.0' encoding='UTF-8' ?> \
      <config> \
        <taskType value='object_detection' /> \
        <datasetDir value='/path/to/dataset' /> \
      </config>");
  
  // act & assert
  EXPECT_THROW(config.parseFromFile(m_invalidConfigPath), std::runtime_error);
}

