#include "base.h"

#include <spdlog/spdlog.h>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

bool AbsEngine::loadClassNames(const std::string& path)
{
  std::ifstream file(path);
  if (!file.is_open())
  {
    spdlog::error("AbsEngine::loadClassNames: could not open file: {}", path);
    return false;
  }

  m_classNames.clear();
  std::string line;
  while (std::getline(file, line))
  {
    // remove any trailing whitespace
    line.erase(line.find_last_not_of(" \n\r\t") + 1);
    if (!line.empty())
      m_classNames.push_back(line);
  }
  file.close();

  spdlog::info("AbsEngine::loadClassNames: loaded {} classes from {}", 
    m_classNames.size(), path);
  
  return true;
}

/* ------------------------------- Pre Processing ------------------------------------ */

void AbsEngine::resizeAndNormalize(const cv::Mat& frame)
{
  cv::resize(frame, m_resizedFrame, cv::Size(m_width, m_height));
  m_resizedFrame.convertTo(m_normalizedFrame, CV_32FC3, 1.0f / 255.0f);
}

/* ------------------------------- Post Processing ------------------------------------ */
/* 
  yolo v5 output shape: [1, num_boxes, 5 + num_classes]
  box-format: [x_center, y_center, width, height, objectness, class_probs...]
*/
 bool AbsEngine::yoloFivePostProc(void* data, int frameWidth, int frameHeight)
{
  const float* outputTensorData {static_cast<const float*>(data)};
  const int num_classes {static_cast<int>(m_classNames.size())};

  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  std::vector<int> class_ids;

  for (int i {0}; i < m_numBoxes; ++i)
  {
    const float objectness_score {outputTensorData[i * (num_classes + 5) + 4]};
    if (objectness_score > m_config.m_confidenceThreshold)
    {
      const float* class_probabilities {&outputTensorData[i * (num_classes + 5) + 5]};
      int best_class_id {-1};
      float best_class_score {0.0f};
      for (int j {0}; j < num_classes; ++j)
      {
        if (class_probabilities[j] > best_class_score)
        {
          best_class_score = class_probabilities[j];
          best_class_id = j;
        }
      }

      const float combined_score {objectness_score * best_class_score};
      if (combined_score > m_config.m_confidenceThreshold)
      {
        const float x_center {outputTensorData[i * (num_classes + 5) + 0]};
        const float y_center {outputTensorData[i * (num_classes + 5) + 1]};
        const float width {outputTensorData[i * (num_classes + 5) + 2]};
        const float height {outputTensorData[i * (num_classes + 5) + 3]};

        const int x1 {static_cast<int>((x_center - width / 2.0f) * frameWidth)};
        const int y1 {static_cast<int>((y_center - height / 2.0f) * frameHeight)};
        const int w {static_cast<int>(width * frameWidth)};
        const int h {static_cast<int>(height * frameHeight)};

        boxes.emplace_back(x1, y1, w, h);
        scores.push_back(combined_score);
        class_ids.push_back(best_class_id);
      }
    }
  }

  applyNms(boxes, scores, class_ids);
  return true;
}

/*
  yolo v8 output shape: [1, 4 + num_classes, num_boxes] (transposed)
  box-format: [x_center, y_center, width, height, class_probs...]
*/
bool AbsEngine::yoloEightPostProc(void* data, int frameWidth, int frameHeight)
{
  const float* outputTensorData {static_cast<const float*>(data)};
  const int num_classes {static_cast<int>(m_classNames.size())};

  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  std::vector<int> class_ids;

  // yolov8 output is transposed: [1, 4 + num_classes, num_boxes]
  for (int i {0}; i < m_numBoxes; ++i)
  {
    int best_class_id {-1};
    float best_class_score {0.0f};

    for (int j {0}; j < num_classes; ++j)
    {
      // access transposed class probabilities
      const float score {outputTensorData[(4 + j) * m_numBoxes + i]};
      if (score > best_class_score)
      {
        best_class_score = score;
        best_class_id = j;
      }
    }

    if (best_class_score > m_config.m_confidenceThreshold)
    {
      const float x_center {outputTensorData[0 * m_numBoxes + i]};
      const float y_center {outputTensorData[1 * m_numBoxes + i]};
      const float width {outputTensorData[2 * m_numBoxes + i]};
      const float height {outputTensorData[3 * m_numBoxes + i]};

      const int x1 {static_cast<int>((x_center - width / 2.0f) * frameWidth)};
      const int y1 {static_cast<int>((y_center - height / 2.0f) * frameHeight)};
      const int w {static_cast<int>(width * frameWidth)};
      const int h {static_cast<int>(height * frameHeight)};

      boxes.emplace_back(x1, y1, w, h);
      scores.push_back(best_class_score);
      class_ids.push_back(best_class_id);
    }
  }

  applyNms(boxes, scores, class_ids);
  return true;
}

/*
  yolo v10 is nms-free, output shape: [1, num_boxes, 6] 
  box-format: [xmin, ymin, xmax, ymax, score, class_id]
*/
bool AbsEngine::yoloTenPostProc(void* data, int frameWidth, int frameHeight)
{
  const float* outputTensorData {static_cast<const float*>(data)};
  // yolov10 is nms-free, typically outputs [1, 300, 6] 
  // format: [xmin, ymin, xmax, ymax, score, class_id]

  m_odOutput.m_classProbabilities.clear();
  m_odOutput.m_firstPoints.clear();
  m_odOutput.m_secondPoints.clear();
  m_odOutput.m_classNameIdxs.clear();

  for (int i {0}; i < m_numBoxes; ++i)
  {
    const float score {outputTensorData[i * 6 + 4]};
    if (score > m_config.m_confidenceThreshold)
    {
      const float x1 {outputTensorData[i * 6 + 0]};
      const float y1 {outputTensorData[i * 6 + 1]};
      const float x2 {outputTensorData[i * 6 + 2]};
      const float y2 {outputTensorData[i * 6 + 3]};
      const int class_id {static_cast<int>(outputTensorData[i * 6 + 5])};

      m_odOutput.m_classProbabilities.push_back(score);
      m_odOutput.m_firstPoints.emplace_back(static_cast<int>(x1 * frameWidth), 
                                            static_cast<int>(y1 * frameHeight));
      m_odOutput.m_secondPoints.emplace_back(static_cast<int>(x2 * frameWidth), 
                                             static_cast<int>(y2 * frameHeight));
      m_odOutput.m_classNameIdxs.push_back(static_cast<std::size_t>(class_id));
    }
  }

  return true;
}

float AbsEngine::calculateIoU(const cv::Rect& box1, const cv::Rect& box2)
{
  const int x1 {std::max(box1.x, box2.x)};
  const int y1 {std::max(box1.y, box2.y)};
  const int x2 {std::min(box1.x + box1.width, box2.x + box2.width)};
  const int y2 {std::min(box1.y + box1.height, box2.y + box2.height)};
        
  const int intersection_area {std::max(0, x2 - x1) * std::max(0, y2 - y1)};
  const int box1_area {box1.width * box1.height};
  const int box2_area {box2.width * box2.height};
  const float union_area {static_cast<float>(
    box1_area + box2_area - intersection_area)
  };

  return (union_area == 0) ? 0.0f : static_cast<float>(intersection_area) / union_area;
}

void AbsEngine::applyNms(const std::vector<cv::Rect>& boxes, 
                        const std::vector<float>& scores,
                        const std::vector<int>& classIds)
{
  std::vector<int> nms_indices;
  if (!scores.empty())
  {
    std::vector<int> indices(scores.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(),
              [&](int a, int b) { return scores[a] > scores[b]; });

    while (!indices.empty())
    {
      int current_idx {indices[0]};
      nms_indices.push_back(current_idx);

      std::vector<int> remaining_indices;
      for (size_t i {1}; i < indices.size(); ++i)
      {
        int other_idx {indices[i]};

        const cv::Rect& box1 {boxes[current_idx]};
        const cv::Rect& box2 {boxes[other_idx]};

        if (calculateIoU(box1, box2) < m_config.m_iouThreshold)
          remaining_indices.push_back(other_idx);
      }
      indices = std::move(remaining_indices);
    }
  }

  m_odOutput.m_classProbabilities.clear();
  m_odOutput.m_firstPoints.clear();
  m_odOutput.m_secondPoints.clear();
  m_odOutput.m_classNameIdxs.clear();

  for (int idx : nms_indices)
  {
    const cv::Rect& box {boxes[idx]};
    m_odOutput.m_classProbabilities.push_back(scores[idx]);
    m_odOutput.m_firstPoints.emplace_back(box.x, box.y);
    m_odOutput.m_secondPoints.emplace_back(box.x + box.width, box.y + box.height);
    m_odOutput.m_classNameIdxs.push_back(static_cast<std::size_t>(classIds[idx]));
  }
}


bool AbsEngine::ssdPostProc(void* data, int frameWidth, int frameHeight)
{
  // TODO
  return true;
}

bool AbsEngine::init(const std::string& configPath)
{
  if (!m_config.parseConfigFile(configPath))
  {
    spdlog::error("AbsEngine::init: could not parse config file: {}", configPath);
    return false;
  }

  if (!loadModel(m_config.m_modelPath))
  {
    spdlog::error("AbsEngine::init: could not load model from path: {}", 
      m_config.m_modelPath);
    return false;
  }

  if (!loadClassNames(m_config.m_classNamesPath))
  {
    spdlog::error("AbsEngine::init: could not load class names from path: {}", 
      m_config.m_classNamesPath);
    return false;
  }

  return true;
}

void AbsEngine::semanticPostProc(void* data, int outW, int outH, int numClasses,
                                  int frameWidth, int frameHeight)
{
  const float* outputData {static_cast<const float*>(data)};
  
  // reserve memory for performance
  m_semantics.m_pixels.reserve(outH * outW);
  m_semantics.m_classNameIdxs.reserve(outH * outW);

  for (int y {0}; y < outH; ++y)
  {
    for (int x {0}; x < outW; ++x)
    {
      // iterate thorugh all classes to find the max probability
      float maxProb {-1.0f};
      int maxIdx {-1};

      for (int c {0}; c < numClasses; ++c)
      {
        float prob {outputData[(y * outW + x) * numClasses + c]};
        if (prob > maxProb)
        {
          maxProb = prob;
          maxIdx = c;
        }
      }

      // scale points back to original frame size
      const int origX {static_cast<int>(static_cast<float>(x) / outW * frameWidth)};
      const int origY {static_cast<int>(static_cast<float>(y) / outH * frameHeight)};

      m_semantics.m_pixels.emplace_back(origX, origY);
      m_semantics.m_classNameIdxs.push_back(static_cast<std::size_t>(maxIdx));
    }
  }
}