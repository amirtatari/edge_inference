#include "../engine/tfLite.h"
#include "gtest/gtest.h"
#include <fstream>
#include <cstdio>
#include <string>

class EngineLiteTest : public ::testing::Test {
protected:
    EngineLite engine;
    std::string invalidModelPath = "invalid_model.tflite";

    void SetUp() override {
        // Setup code if needed
    }

    void TearDown() override {
        // Clean up garbage file if it exists
        std::remove(invalidModelPath.c_str());
    }

    void CreateGarbageFile(const std::string& path) {
        std::ofstream outfile(path);
        outfile << "This is not a tflite model";
        outfile.close();
    }
};

TEST_F(EngineLiteTest, LoadInvalidModel) 
{
  CreateGarbageFile(invalidModelPath);
  EXPECT_FALSE(engine.loadModel(invalidModelPath));
}


TEST_F(EngineLiteTest, LoadValidModel) 
{
  const std::string validModelPath {"/home/dev/repos/edge_inference/modelZoo/yolov5.tflite"};
  EXPECT_TRUE(engine.loadModel(validModelPath));
}