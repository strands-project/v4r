#ifndef EPDRAWUTILS_H
#define EPDRAWUTILS_H

#include "headers.hpp"
#include "connectedComponents.hpp"

namespace EPUtils
{

/**
 * draw one segmentation masks
 * */
void drawSegmentationMask(cv::Mat &image, cv::Mat mask, cv::Scalar color, int line_width = 2);
  
/**
 * draw a banch of segmentation masks
 * */
void drawSegmentationMasks(cv::Mat &image, std::vector<cv::Mat> &masks, int line_width = 2);

/**
 * draw a segmentation masks and attetnion points
 * */
void drawSegmentationResults(cv::Mat &image, std::vector<cv::Point> &attentionPoints,
                             std::vector<std::vector<cv::Point> > &contours, bool drawAttentionPoints = true,
                             bool drawSegmentationResults = true, bool drawLines = false, unsigned int num = -1);

/**
 * draws segmentation and attention results
 * */
void drawSegmentationResults(cv::Mat &image, std::vector<cv::Point> &attentionPoints,
                             std::vector<cv::Mat> &binMasks, std::vector<std::vector<cv::Point> > &contours,
                             bool drawAttentionPoints, bool drawSegmentationResults);

/**
 * draws attention points
 * */
void drawAttentionPoints(cv::Mat &image, std::vector<cv::Point> &attentionPoints,
                         unsigned int maxNumber = 0, bool connect_points = false);

/**
 * draws path through the map
 */
void drawPath(cv::Mat &image, std::vector<cv::Point> &path, cv::Mat &mapx, cv::Mat &mapy);

/**
 * draws line
 */
void drawLine(cv::Mat &image, std::vector<cv::Point> points, cv::Scalar color = cv::Scalar(0));

} //namespace EPUtils

#endif //EPDRAWUTILS_H