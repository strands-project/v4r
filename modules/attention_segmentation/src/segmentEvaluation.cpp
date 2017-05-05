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


#include "v4r/attention_segmentation/segmentEvaluation.h"

namespace EPEvaluation {

void printSegmentationEvaluation(std::string output_filename, std::string base_name, 
                                 std::vector<long int> &tp, std::vector<long int> &fp, std::vector<long int> &fn, std::vector<bool> &used, std::vector<int> &objNumber)
{
  assert(tp.size() == fp.size());
  assert(tp.size() == fn.size());
  //assert(tp.size() == used.size());
  assert(tp.size() == objNumber.size());
  
  FILE *f;
  // image mask# object# tp# fp# fn#
  
  //check if file exists
  if ( !boost::filesystem::exists(output_filename) )
  {
    f = fopen(output_filename.c_str(), "a");
    fprintf(f,"%12s %7s %7s %8s %8s %8s %6s %6s %6s\n","image_name","mask#","object#","tp#","fp#","fn#","p","r","fscore");
  }
  else
  {
    f = fopen(output_filename.c_str(), "a");
  }
    
  for(size_t i = 1; i < objNumber.size(); ++i)
  {
    boost::filesystem::path path(base_name);
    
    float precision = 0;
    if((tp.at(i) + fn.at(i)) > 0)
      precision = ((float)tp.at(i))/(((float)tp.at(i)) + ((float)fn.at(i)));
      
    float recall = 0;
    if((tp.at(i) + fp.at(i)) > 0)
      recall = ((float)tp.at(i))/(((float)tp.at(i)) + ((float)fp.at(i)));
      
    float fscore = 0;
    if((precision + recall) > 0)
      fscore = (2*precision*recall)/(precision + recall);
    
    fprintf(f,"%12s %7d %7d %8ld %8ld %8ld %5.4f %5.4f %5.4f\n",path.filename().c_str(),(int)i,objNumber.at(i),tp.at(i),fp.at(i),fn.at(i),precision,recall,fscore);  
  }
    
  fclose(f);
}

void evaluate(pcl::PointCloud<pcl::PointXYZRGBL>::Ptr pcl_cloud_l, cv::Mat &mask, std::string base_name, std::string output_filename)
{
  std::vector<long int> tp;
  std::vector<long int> fp;
  std::vector<long int> fn;
  std::vector<bool> used;
  std::vector<int> objNumber;
  
  evaluateSegmentation(pcl_cloud_l,mask,tp,fp,fn,used,objNumber);

  printSegmentationEvaluation(output_filename,base_name,tp,fp,fn,used,objNumber);
}

void evaluateSegmentation(pcl::PointCloud<pcl::PointXYZRGBL>::Ptr pcl_cloud_l, cv::Mat &mask, 
                          std::vector<long int> &tp, std::vector<long int> &fp, std::vector<long int> &fn, std::vector<bool> &used, std::vector<int> &objNumber)
{
  // finding maximum number of objects
  int numberOfObjects = 0;
  for(size_t i = 0; i < pcl_cloud_l->points.size(); ++i)
  {
    if(pcl_cloud_l->points.at(i).label > ((unsigned int)numberOfObjects))
      numberOfObjects = pcl_cloud_l->points.at(i).label;
  }
    
  // finding mask maximum label number
  int maxLabel = 0;
  for(int i = 0; i < mask.rows; i++)
  {
    for(int j = 0; j < mask.cols; j++)
    {
      if(mask.at<uchar>(i,j) > maxLabel)
        maxLabel = mask.at<uchar>(i,j);
    }
  }
    
  if(maxLabel <= 0)
  {
    printf("It seems like no objects were segmented!\n");
    return;
  }
    
  tp.resize(maxLabel+1, 0);
  fp.resize(maxLabel+1, 0);
  fn.resize(maxLabel+1, 0);
  objNumber.resize(maxLabel+1,0);
  used.resize(maxLabel+1,false);
    
  //find to which object the mask belongs
  for(int l = 1; l <= maxLabel; ++l)
  {
    used.at(l) = false;
    std::vector<int> pointsOnTheObject(numberOfObjects+1,0);
    
    for(size_t i = 0; i < pcl_cloud_l->height; i++)
    {
      for(size_t j = 0; j < pcl_cloud_l->width; j++)
      {
        //if current point belong to the mask
	if(mask.at<uchar>(i,j) == l)
        {
          int idx = i*pcl_cloud_l->width + j;
          if(pcl_cloud_l->points.at(idx).label > 0)
          {
            pointsOnTheObject.at(pcl_cloud_l->points.at(idx).label) += 1;
          }
          else
          {
            pointsOnTheObject.at(0) += 1;
          }
        }
      }
    }
      
    // detect object index
    int objectIdx = 0;
    for(int j = 1; j <= numberOfObjects; ++j)
    {
      if(pointsOnTheObject.at(j) > pointsOnTheObject.at(objectIdx))
        objectIdx = j;
    }
      
    if(objectIdx == 0)
      continue;

    used.at(l) = true;
    objNumber.at(l) = objectIdx;
      
    for(size_t i = 0; i < pcl_cloud_l->height; i++)
    {
      for(size_t j = 0; j < pcl_cloud_l->width; j++)
      {
        int idx = i*pcl_cloud_l->width + j;
          
        if(pcl_cloud_l->points.at(idx).label == (unsigned int)objectIdx)
        {
          if(mask.at<uchar>(i,j) == l)
            tp.at(l) += 1;
          else
            fn.at(l) += 1;
        }
        else
        {
          if(mask.at<uchar>(i,j) == l)
            fp.at(l) += 1;
        }
      }
    }
  }
}

void evaluateSegmentation(pcl::PointCloud<pcl::PointXYZRGBL>::Ptr pcl_cloud_l, cv::Mat &mask, cv::Point attention_point,
                          std::vector<long int> &tp, std::vector<long int> &fp, std::vector<long int> &fn, std::vector<bool> &used, std::vector<int> &objNumber)
{
  // finding maximum number of objects
  int numberOfObjects = 0;
  for(size_t i = 0; i < pcl_cloud_l->points.size(); ++i)
  {
    if(pcl_cloud_l->points.at(i).label > ((unsigned int)numberOfObjects))
      numberOfObjects = pcl_cloud_l->points.at(i).label;
  }
  
  tp.resize(2,0);
  fp.resize(2,0);
  fn.resize(2,0);
  used.resize(2,false);
  objNumber.resize(2,0);
  
  int objectIdx = pcl_cloud_l->points.at(attention_point.y * mask.cols + attention_point.x).label;
  
  if(objectIdx == 0)
    return;
    
  objNumber.at(1) = objectIdx;
  used.at(1) = true;
  
  for(size_t i = 0; i < pcl_cloud_l->height; i++)
  {
    for(size_t j = 0; j < pcl_cloud_l->width; j++)
    {
      int idx = i*pcl_cloud_l->width + j;
          
      if(pcl_cloud_l->points.at(idx).label == (unsigned int)objectIdx)
      {
        if(mask.at<uchar>(i,j) > 0)
          tp.at(1) += 1;
        else
          fn.at(1) += 1;
      }
      else
      {
        if(mask.at<uchar>(i,j) > 0)
          fp.at(1) += 1;
      }
    }
  }
}

void evaluate(pcl::PointCloud<pcl::PointXYZRGBL>::Ptr pcl_cloud_l, cv::Mat &mask, cv::Point attention_point, std::string base_name, std::string output_filename)
{
  std::vector<long int> tp;
  std::vector<long int> fp;
  std::vector<long int> fn;
  std::vector<bool> used;
  std::vector<int> objNumber;
  
  evaluateSegmentation(pcl_cloud_l,mask,attention_point,tp,fp,fn,used,objNumber);
  
  printSegmentationEvaluation(output_filename,base_name,tp,fp,fn,used,objNumber);
}

void evaluateSegmentation(const cv::Mat &ground_truth_image, cv::Mat &mask, cv::Point attention_point,
                          std::vector<long int> &tp, std::vector<long int> &fp, std::vector<long int> &fn, std::vector<bool> &used, std::vector<int> &objNumber)
{
  // finding maximum number of objects
  int numberOfObjects = 0;
  for(int i = 0; i < ground_truth_image.rows; ++i)
  {
    for(int j = 0; j < ground_truth_image.cols; ++j)
    {
      if(ground_truth_image.at<uchar>(i,j) > ((unsigned int)numberOfObjects))
        numberOfObjects = ground_truth_image.at<uchar>(i,j);
    }
  }
  
  tp.resize(2,0);
  fp.resize(2,0);
  fn.resize(2,0);
  used.resize(2,false);
  objNumber.resize(2,0);
  
  int objectIdx = ground_truth_image.at<uchar>(attention_point.y,attention_point.x);
  
  if(objectIdx == 0)
    return;
    
  objNumber.at(1) = objectIdx;
  used.at(1) = true;
  
  for(int i = 0; i < ground_truth_image.rows; i++)
  {
    for(int j = 0; j < ground_truth_image.cols; j++)
    {    
      if((ground_truth_image.at<uchar>(i,j)) == (unsigned int)objectIdx)
      {
        if(mask.at<uchar>(i,j) > 0)
          tp.at(1) += 1;
        else
          fn.at(1) += 1;
      }
      else
      {
        if(mask.at<uchar>(i,j) > 0)
          fp.at(1) += 1;
      }
    }
  }
}

void evaluate(const cv::Mat &ground_truth_image, cv::Mat &mask, cv::Point attention_point, std::string base_name, std::string output_filename)
{
  std::vector<long int> tp;
  std::vector<long int> fp;
  std::vector<long int> fn;
  std::vector<bool> used;
  std::vector<int> objNumber;
  
  evaluateSegmentation(ground_truth_image,mask,attention_point,tp,fp,fn,used,objNumber);
  
  printSegmentationEvaluation(output_filename,base_name,tp,fp,fn,used,objNumber);
}

} //namespace EPEvaluation
