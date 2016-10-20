/**
 *  Copyright (C) 2012  
 *    Ekaterina Potapova
 *    Automation and Control Institute
 *    Vienna University of Technology
 *    Gusshausstraße 25-29
 *    1040 Vienna, Austria
 *    potapova(at)acin.tuwien.ac.at
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see http://www.gnu.org/licenses/
 */


#ifndef EPEVALUATION_SEGMENT_HPP
#define EPEVALUATION_SEGMENT_HPP

#include <v4r/core/macros.h>
#include "v4r/attention_segmentation/headers.h"

namespace EPEvaluation {

void printSegmentationEvaluation(std::string output_filename, std::string base_name, 
                                 std::vector<long int> &tp, std::vector<long int> &fp, std::vector<long int> &fn, std::vector<bool> &used, std::vector<int> &objNumber);
V4R_EXPORTS void evaluate(pcl::PointCloud<pcl::PointXYZRGBL>::Ptr pcl_cloud_l, cv::Mat &mask, std::string base_name, std::string output_filename);
V4R_EXPORTS void evaluate(pcl::PointCloud<pcl::PointXYZRGBL>::Ptr pcl_cloud_l, cv::Mat &mask, cv::Point attention_point, std::string base_name, std::string output_filename);
V4R_EXPORTS void evaluate(const cv::Mat &ground_truth_image, cv::Mat &mask, cv::Point attention_point, std::string base_name, std::string output_filename);
void evaluateSegmentation(pcl::PointCloud<pcl::PointXYZRGBL>::Ptr pcl_cloud_l, cv::Mat &mask, 
                          std::vector<long int> &tp, std::vector<long int> &fp, std::vector<long int> &fn, std::vector<bool> &used, std::vector<int> &objNumber);
void evaluateSegmentation(pcl::PointCloud<pcl::PointXYZRGBL>::Ptr pcl_cloud_l, cv::Mat &mask, cv::Point attention_point,
                          std::vector<long int> &tp, std::vector<long int> &fp, std::vector<long int> &fn, std::vector<bool> &used, std::vector<int> &objNumber);
void evaluateSegmentation(const cv::Mat &ground_truth_image, cv::Mat &mask, cv::Point attention_point,
                          std::vector<long int> &tp, std::vector<long int> &fp, std::vector<long int> &fn, std::vector<bool> &used, std::vector<int> &objNumber);

} //namespace EPEvaluation

#endif //EPEVALUATION_SEGMENT_HPP
