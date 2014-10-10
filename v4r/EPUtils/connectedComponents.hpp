#ifndef EPCONNECTEDCOMPONENTS_H
#define EPCONNECTEDCOMPONENTS_H

#include "headers.hpp"

namespace EPUtils
{
  
struct ConnectedComponent {
  std::vector<cv::Point> points;
  std::vector<float> saliency_values;
  float average_saliency;
  ConnectedComponent();
};
/**
 * extracts connected components from the map using given threshold
 * map is considered to be in the range 0..1 with type CV_32F
 * */
void extractConnectedComponents(cv::Mat map, std::vector<ConnectedComponent> &connected_components, float th = 0.1);
void extractConnectedComponents(cv::Mat map, std::vector<ConnectedComponent> &connected_components, cv::Point attention_point, float th = 0.1);

/**
 * extracts connected components from the map using given threshold
 * map is considered to be in the range 0..1 with type CV_32F
 * */
//void extractConnectedComponents2(cv::Mat map, std::vector<ConnectedComponent> &connected_components, float th = 0.1);

/**
 * draws single connected component over the image
 */
void drawConnectedComponent(ConnectedComponent component, cv::Mat &image, cv::Scalar color);

/**
 * draws connected components over the image
 */
void drawConnectedComponents(std::vector<ConnectedComponent> components, cv::Mat &image, cv::Scalar color);

} //namespace EPUtils

#endif // EPCONNECTEDCOMPONENTS_H