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


#include "v4r/attention_segmentation/HitRatio.h"
#include <list>

namespace EPEvaluation {

void labeling2Mask(cv::Mat &mask, cv::Mat labeling, int maskNum)
{
  assert(labeling.type() == CV_8UC1);
  
  mask = cv::Mat_<uchar>::zeros(labeling.rows,labeling.cols);
  
  for(int i = 0; i < labeling.rows; ++i)
  {
    for(int j = 0; j < labeling.cols; ++j)
    {
      if(labeling.at<uchar>(i,j) == maskNum)
	mask.at<uchar>(i,j) = 1;
    }
  }
}

float hitRatio(std::vector<cv::Point> attentionPoints, cv::Mat labeling, std::vector<bool> &usedPoints)
{
  assert(labeling.type() == CV_8UC1);
  
  if(attentionPoints.size() <= 0)
  {
    return(0.0f);
  }
  
  usedPoints.clear();
  usedPoints.resize(attentionPoints.size(), false);
  
  double maxVal=0;
  cv::minMaxLoc(labeling,0,&maxVal,0,0);
  
  std::vector<bool> usedObjects(maxVal,false);
  
  float visitedObjects = 0;
  
  for(unsigned int i = 0; i < attentionPoints.size(); ++i)
  {
    cv::Point p = attentionPoints.at(i);
    uchar objNum = labeling.at<uchar>(p.y,p.x);
    if(objNum == 0)
      continue;
    if(!(usedObjects.at(objNum-1)))
    {
      usedPoints.at(i) = true;
      usedObjects.at(objNum-1) = true;
      visitedObjects = visitedObjects + 1;
    }
  }
  
  float HR = visitedObjects/attentionPoints.size();
  
  return(HR);
}

void calculateAccumulatedHR(std::vector<bool> usedPoints, std::vector<int> &accumulatedHR)
{
  accumulatedHR.resize(usedPoints.size(),0);
  
  for(unsigned int i = 0; i < usedPoints.size(); ++i)
  {
    int before = 0;
    if(i > 0)
      before = accumulatedHR.at(i-1);
    
    if(usedPoints.at(i))
      before += 1;
    
    accumulatedHR.at(i) = before;
  }
}

void distance2Center(std::vector<cv::Point> attentionPoints, cv::Mat labeling, std::vector<cv::Point> centers, 
                     std::vector<PointEvaluation> &distances, std::vector<bool> &usedAttentionPoints)
{
  assert(labeling.type() == CV_8UC1);
  assert(attentionPoints.size() > 0);
  
  double maxVal=0;
  cv::minMaxLoc(labeling,0,&maxVal,0,0);
  
  assert(centers.size() == (unsigned int)maxVal);
  
  distances.resize(attentionPoints.size());
  usedAttentionPoints.resize(attentionPoints.size(),false);
  
  for(unsigned int i = 0; i < attentionPoints.size(); ++i)
  {
    cv::Point p = attentionPoints.at(i);
    uchar objNum = labeling.at<uchar>(p.y,p.x);
    if(objNum == 0)
    {
      usedAttentionPoints.at(i) = true;
      distances.at(i).ObjectIdx = -1;
      distances.at(i).distance = -1;
      continue;
    }
    
    //calculate distance to the center
    float distance = v4r::calculateDistance(centers.at(objNum-1),p);
    
    usedAttentionPoints.at(i) = true;
    distances.at(i).ObjectIdx = objNum;
    distances.at(i).distance = distance;
  }
  
  return;
}

void distance2Center(std::vector<cv::Point> attentionPoints, cv::Mat labeling, std::vector<cv::Point> centers, 
                     std::vector<ObjectEvaluation> &firstDistance2Objects, std::vector<ObjectEvaluation> &bestDistance2Objects, 
                     std::vector<bool> &usedObjects)
{
  assert(labeling.type() == CV_8UC1);
  assert(attentionPoints.size() > 0);
  
  double maxVal=0;
  cv::minMaxLoc(labeling,0,&maxVal,0,0);
  
  assert(centers.size() == (unsigned int)maxVal);
  
  firstDistance2Objects.resize(maxVal);
  bestDistance2Objects.resize(maxVal);
  usedObjects.resize(maxVal,false);
  
  for(unsigned int i = 0; i < attentionPoints.size(); ++i)
  {
    cv::Point p = attentionPoints.at(i);
    uchar objNum = labeling.at<uchar>(p.y,p.x);
    if(objNum == 0)
      continue;
    
    //calculate distance to the center
    float distance = v4r::calculateDistance(centers.at(objNum-1),p);
    
    // object is used the first time
    if(!(usedObjects.at(objNum-1)))
    {
      usedObjects.at(objNum-1) = true;
      firstDistance2Objects.at(objNum-1).attentionPointIdx = i;
      firstDistance2Objects.at(objNum-1).distance = distance;
      
      bestDistance2Objects.at(objNum-1).attentionPointIdx = i;
      bestDistance2Objects.at(objNum-1).distance = distance;
    }
    else
    {
      if(distance < bestDistance2Objects.at(objNum-1).distance)
      {
        bestDistance2Objects.at(objNum-1).attentionPointIdx = i;
        bestDistance2Objects.at(objNum-1).distance = distance;
      }
    }
  }
  
  return;
  
}


}//namespace EPEvaluation
