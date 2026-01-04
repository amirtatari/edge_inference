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
/**
  * This method processes the output tensor of a YOLOv5 model.
  *
  * It assumes the output tensor is structured as follows:
  * Shape: [1, num_boxes, 85] (assuming 80 classes)
  * Each detection (row) contains 85 float values:
  * [x, y, w, h, objectness_score, class_prob_0, class_prob_1, ..., class_prob_79]
  *
  * - `x, y, w, h`: Bounding box center coordinates (x, y) and dimensions (width, height).
  *   These are normalized values (0.0 to 1.0) and need to be scaled back to original frame dimensions.
  * - `objectness_score`: A confidence score indicating the probability that an object exists in this box.
  * - `class_prob_0` to `class_prob_79`: Probabilities for each possible class.
  *
  * Post-processing typically involves:
  * - Filtering detections by `objectness_score` and `class_probabilities` thresholds.
  * - Applying Non-Maximum Suppression (NMS) to remove redundant overlapping boxes.
  * - Scaling bounding box coordinates to the original frame size.
  */
bool AbsEngine::yoloFivePostProc(void* data, int frameWidth, int frameHeight)
{
  // TODO

  
  /*
  //const float* outputTensorData {m_models.m_outputTensorPtrs[modelIdx]->data.f};
  //const auto* outputDims = m_models.m_outputTensorPtrs[modelIdx]->dims;
  const float* outputTensorData = nullptr;
  //const auto* outputDims = nullptr;
  const float* outputDims = nullptr;
  const int num_boxes = 0; //outputDims->data[1];
  const int num_classes_plus_box = 0; // outputDims->data[2];
  const int num_classes = num_classes_plus_box - 5;

  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  std::vector<int> class_ids;

  const float confidence_threshold{0.0};// = m_models.m_confidences[modelIdx];

  for (int i = 0; i < num_boxes; ++i)
  {
    const float objectness_score = outputTensorData[i * num_classes_plus_box + 4];
    if (objectness_score > confidence_threshold)
    {
      const float* class_probabilities = &outputTensorData[i * num_classes_plus_box + 5];
      int best_class_id = -1;
      float best_class_score = 0.0f;
      for (int j = 0; j < num_classes; ++j)
      {
        if (class_probabilities[j] > best_class_score)
        {
          best_class_score = class_probabilities[j];
          best_class_id = j;
        }
      }

      if (best_class_score > confidence_threshold)
      {
        const float x_center = outputTensorData[i * num_classes_plus_box + 0];
        const float y_center = outputTensorData[i * num_classes_plus_box + 1];
        const float width = outputTensorData[i * num_classes_plus_box + 2];
        const float height = outputTensorData[i * num_classes_plus_box + 3];

        const int x1 = static_cast<int>((x_center - width / 2.0f) * frameWidth);
        const int y1 = static_cast<int>((y_center - height / 2.0f) * frameHeight);
        const int w = static_cast<int>(width * frameWidth);
        const int h = static_cast<int>(height * frameHeight);

        boxes.emplace_back(x1, y1, w, h);
        scores.push_back(objectness_score * best_class_score); // Use combined score
        class_ids.push_back(best_class_id);
      }
    }
  }

  // Custom Non-Maximum Suppression
  const float iou_threshold {0.0};// = m_models.m_IoUs[modelIdx];
  std::vector<int> nms_indices;

  if (!scores.empty()) {
    std::vector<int> indices(scores.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
      return scores[a] > scores[b];
    });

    while (!indices.empty()) {
      int current_idx = indices[0];
      nms_indices.push_back(current_idx);

      std::vector<int> remaining_indices;
      for (size_t i = 1; i < indices.size(); ++i) {
        int other_idx = indices[i];
        
        // Calculate IoU
        const cv::Rect& box1 = boxes[current_idx];
        const cv::Rect& box2 = boxes[other_idx];
        const int x1 = std::max(box1.x, box2.x);
        const int y1 = std::max(box1.y, box2.y);
        const int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
        const int y2 = std::min(box1.y + box1.height, box2.y + box2.height);
        const int intersection_area = std::max(0, x2 - x1) * std::max(0, y2 - y1);
        const int box1_area = box1.width * box1.height;
        const int box2_area = box2.width * box2.height;
        const float union_area = box1_area + box2_area - intersection_area;
        const float iou = (union_area == 0) ? 0.0f : static_cast<float>(intersection_area) / union_area;

        if (iou < iou_threshold) {
          remaining_indices.push_back(other_idx);
        }
      }
      indices = remaining_indices;
    }
  }

  for (int idx : nms_indices)
  {
    const cv::Rect& box = boxes[idx];
    m_objsDetected.m_classProbabilities.push_back(scores[idx]);
    m_objsDetected.m_xOnes.push_back(box.x);
    m_objsDetected.m_yOnes.push_back(box.y);
    m_objsDetected.m_xTwos.push_back(box.x + box.width);
    m_objsDetected.m_yTwos.push_back(box.y + box.height);
    m_objsDetected.m_classNameIdxs.push_back(class_ids[idx]);
  }
  */

  return true;
}

/**
  *
  * It assumes the output tensor is structured as follows:
  * Shape: [1, num_detections, 7]
  * Each detection (row) contains 7 float values:
  * [batch_id, ymin, xmin, ymax, xmax, class_id, score]
  *
  * - `batch_id`: Index of the image in the batch (typically 0 for single image inference).
  * - `ymin, xmin, ymax, xmax`: Normalized bounding box coordinates (0.0 to 1.0)
  *   representing the top-left (ymin, xmin) and bottom-right (ymax, xmax) corners.
  *   These are scaled back to original frame dimensions for `m_objsDetected`.
  * - `class_id`: The integer ID of the detected object's class.
  * - `score`: The confidence score (probability) of the detection.
  *
  * Data is accessed using `outputTensorData[i * 7 + index]`, where `i` is the detection index
  * and `index` corresponds to the position within the 7-value detection array.
  */
bool AbsEngine::ssdPostProc(void* data, int frameWidth, int frameHeight)
{
  // TODO

  /*
  const float* outputTensorData {m_models.m_outputTensorPtrs[modelIdx]->data.f};
  const int numDetections {m_models.m_outputTensorPtrs[modelIdx]->dims->data[1]};

  for (int i = 0; i < numDetections; ++i)
  {
    const float score {outputTensorData[i * 7 + 6]};
    if (score > m_models.m_confidences[modelIdx])
    {
      const float ymin {outputTensorData[i * 7 + 1]};
      const float xmin {outputTensorData[i * 7 + 2]};
      const float ymax {outputTensorData[i * 7 + 3]};
      const float xmax {outputTensorData[i * 7 + 4]};
      const size_t classId {static_cast<std::size_t>(outputTensorData[i * 7 + 5])};

      m_objsDetected.m_classProbabilities.push_back(score);
      m_objsDetected.m_xOnes.push_back(static_cast<int>(xmin * originalFrameWidth));
      m_objsDetected.m_yOnes.push_back(static_cast<int>(ymin * originalFrameHeight));
      m_objsDetected.m_xTwos.push_back(static_cast<int>(xmax * originalFrameWidth));
      m_objsDetected.m_yTwos.push_back(static_cast<int>(ymax * originalFrameHeight));
      m_objsDetected.m_classNameIdxs.push_back(classId);
    }
  }
  */
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