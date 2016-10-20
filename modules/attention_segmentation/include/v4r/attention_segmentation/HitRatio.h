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


#ifndef HITRATIO_HPP
#define HITRATIO_HPP

#include "headers.h"

namespace EPEvaluation {

struct ObjectEvaluation{
  float distance;
  int attentionPointIdx;
};

struct PointEvaluation{
  float distance;
  int ObjectIdx;
};

void labeling2Mask(cv::Mat &mask, cv::Mat labeling, int maskNum);
float hitRatio(std::vector<cv::Point> attentionPoints, cv::Mat labeling, std::vector<bool> &usedPoints);
void calculateAccumulatedHR(std::vector<bool> usedPoints, std::vector<int> &accumulatedHR);
void distance2Center(std::vector<cv::Point> attentionPoints, cv::Mat labeling, std::vector<cv::Point> centers, 
                     std::vector<ObjectEvaluation> &firstDistance2Objects, std::vector<ObjectEvaluation> &bestDistance2Objects, 
                     std::vector<bool> &usedObjects);
void distance2Center(std::vector<cv::Point> attentionPoints, cv::Mat labeling, std::vector<cv::Point> centers, 
                     std::vector<PointEvaluation> &distances, std::vector<bool> &usedAttentionPoints);
  
} //namespace EPEvaluation

#endif //HITRATIO_HPP