#pragma once

#include "liteRt.hpp"

// stl
#include <span>

class PostProcess
{
  static float m_confidenceThreshold {0.5f};
   /**
   * @brief applies non-maximum suppression to filter overlapping boxes
   * @param boxes vector of bounding boxes
   * @param scores vector of confidence scores
   * @param classIds vector of class indices
   */
  static void applyNms(std::span<const cv::Rect> boxes, 
                      const std::vector<float>& scores,
                      const std::vector<int>& classIds);

public:
  /**
   * @brief run post proccessing algorithm on the output tensor of a YOLOv5 model
   * @param data pointer to the output tensor data
   * @param frameWidth original frame width
   * @param frameHeight original frame height
   * @return true if successful, false otherwise
   */
  static std::optional<modelIo::DetectedObjects> 
  yoloFivePostProc(void* data, int frameWidth, int frameHeight, int numBoxes) const;

  /**
   * @brief run post proccessing algorithm on the output tensor of a YOLOv8 model
   * @param data pointer to the output tensor data
   * @param frameWidth original frame width
   * @param frameHeight original frame height
   * @return true if successful, false otherwise
   */
  static std::optional<modelIo::DetectedObjects> 
  yoloEightPostProc(void* data, int frameWidth, int frameHeight, int numBoxes) const;

  /**
   * @brief run post proccessing algorithm on the output tensor of a YOLOv10 model
   * @param data pointer to the output tensor data
   * @param frameWidth original frame width
   * @param frameHeight original frame height
   * @return true if successful, false otherwise
   */
  static std::optional<modelIo::DetectedObjects> 
  yoloTenPostProc(void* data, int frameWidth, int frameHeight, int numBoxes) const;

  /**
   * @brief run post proccessing algorithm on the output tensor of an SSD model
   * @param data pointer to the output tensor data
   * @param frameWidth original frame width
   * @param frameHeight original frame height
   * @return true if successful, false otherwise
   */
  static std::optional<modelIo::DetectedObjects> 
  ssdPostProc(void* data, int frameWidth, int frameHeight, int numBoxes) const;

  /**
   * @brief run post proccessing algorithm for semantic segmentation model
   * @param data pointer to the output tensor data
   * @param outW output tensor width
   * @param outH output tensor height
   * @param numClasses number of classes
   * @param frameWidth original frame width
   * @param frameHeight original frame height
   */
  static std::optional<modelIo::DetectedSemantics>  
  semanticPostProc(void* data, int outW, int outH, int numClasses, int frameWidth, int frameHeight) const;

  /**
   * @brief calculates Intersection over Union (IoU) between two boxes
   * @param box1 first bounding box
   * @param box2 second bounding box
   * @return IoU value
   */
  static float calculateIoU(const cv::Rect& box1, const cv::Rect& box2) const;
};