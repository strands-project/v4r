#ifndef EPUTILS_H
#define EPUTILS_H

#include "headers.hpp"

namespace EPUtils
{

/**
 * reads files from a given directory
 * */
int  readFiles(const std::string &directoryName, std::vector<std::string> &names);
/**
 * reads coordinated of the polygons from the given file
 * */
void readPolygons(std::vector<std::vector<cv::Point> > &polygons, std::string &str);

/**
 * write coordinated of the polygons from the given file
 * */
void writePolygons(std::vector<std::vector<cv::Point> > &polygons, std::string &str);

/**
 * reads rectangles form fole
 * */
void readRectangles(std::vector<cv::Rect> &rectangles, std::string &str);
/**
 * reads attention points from file
 * */
void readAttentionPoints(std::vector<std::vector<cv::Point> > &attentionPoints, std::string &str);

void readAttentionPoints(std::vector<cv::Point> &attentionPoints, std::string &str);
/**
 * writes attention points to file
 * */
void writeAttentionPoints(std::vector<cv::Point> attentionPoints, std::string &str);
/**
 * reads attention points and contours from file
 * */
void readAttentionPointsAndContours(std::vector<cv::Point> &attentionPoints,
                                    std::vector<std::vector<cv::Point> > &contours, std::string &str);
/**
 * reads image annotations from file
 * */
void readAnnotationsFromFile(std::vector<std::vector<cv::Point> > &polygons, std::string &annotationName);
/**
 * saturates image
 * */
void saturation(cv::Mat &map, float max_value = 1.0);
/**
 * calculates centers of the polygons
 * */
void calculatePolygonsCenters(std::vector<std::vector<cv::Point> > &polygons,
                              std::vector<cv::Point> &attentionPoints,
                              std::vector<int> &attentionPointsBelongTo);

double PolygonArea(std::vector<cv::Point> &polygon);
void Centroid(std::vector<cv::Point> &polygon, cv::Point &center);
void makeGaborFilter(cv::Mat &filter0, cv::Mat &filter90, float angle, float stddev = 2.33, float elongation = 1, int filterSize = 9,
                     int filterPeriod = 7);

void readCenters(std::vector<cv::Point> &centers, std::string &str);
void makeGaborKernel2D(cv::Mat &kernel, float &max_sum, float theta = 0, float bandwidth = 4, float lambda = 10, float sigma = 0, float phi = 0, 
                       float gamma = 0.5);
void saveDistribution(std::vector<float> dist, std::string filename);
void readDistribution(std::vector<float> &dist, std::string filename);

} //namespace EPUtils

#endif // EPUTILS_H
