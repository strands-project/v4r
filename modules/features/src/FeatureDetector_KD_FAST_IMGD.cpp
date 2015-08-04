/**
 * $Id$
 * 
 * Software License Agreement (GNU General Public License)
 *
 *  Copyright (C) 2015:
 *
 *    Johann Prankl, prankl@acin.tuwien.ac.at
 *    Aitor Aldoma, aldoma@acin.tuwien.ac.at
 *
 *      Automation and Control Institute
 *      Vienna University of Technology
 *      Gusshausstraße 25-29
 *      1170 Vienn, Austria
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
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * @author Johann Prankl, Aitor Aldoma
 *
 */

#include <v4r/features/FeatureDetector_KD_FAST_IMGD.h>

namespace v4r
{

using namespace std;

/************************************************************************************
 * Constructor/Destructor
 */
FeatureDetector_KD_FAST_IMGD::FeatureDetector_KD_FAST_IMGD(const Parameter &_p)
 : FeatureDetector(KD_FAST_IMGD), param(_p)
{ 
  //orb = new cv::ORB(10000, 1.2, 6, 13, 0, 2, cv::ORB::HARRIS_SCORE, 13); //31
  //orb = new cv::ORB(1000, 1.44, 2, 17, 0, 2, cv::ORB::HARRIS_SCORE, 17);
  orb = new cv::ORB(param.nfeatures, param.scaleFactor, param.nlevels, param.patchSize, 0, 2, cv::ORB::HARRIS_SCORE, param.patchSize);

  imGDesc.reset(new ComputeImGradientDescriptors(param.gdParam));

  fs.reset( new FeatureSelection(FeatureSelection::Parameter(2.,0.5)) );
}

FeatureDetector_KD_FAST_IMGD::~FeatureDetector_KD_FAST_IMGD()
{
}

/***************************************************************************************/

/**
 * detect
 */
void FeatureDetector_KD_FAST_IMGD::detect(const cv::Mat &image, std::vector<cv::KeyPoint> &keys, cv::Mat &descriptors)
{
  if( image.type() != CV_8U ) cv::cvtColor( image, im_gray, CV_RGB2GRAY );
  else im_gray = image;  

  (*orb)(im_gray, cv::Mat(), keys);

  imGDesc->compute(im_gray, keys, descriptors);
}

/**
 * detect
 */
void FeatureDetector_KD_FAST_IMGD::detect(const cv::Mat &image, std::vector<cv::KeyPoint> &keys)
{
  keys.clear();

  if( image.type() != CV_8U ) cv::cvtColor( image, im_gray, CV_RGB2GRAY );
  else im_gray = image;  

  if (param.tiles>1)
  {
    cv::Rect rect;
    cv::Point2f pt_offs;
    std::vector<cv::KeyPoint> tmp_keys;

    tile_size_w = image.cols/param.tiles;
    tile_size_h = image.rows/param.tiles;

    for (int v=0; v<param.tiles; v++)
    { 
      for (int u=0; u<param.tiles; u++)
      {
        getExpandedRect(u,v, image.rows, image.cols, rect);
        (*orb)(im_gray(rect), cv::Mat(), tmp_keys);
    
        pt_offs = cv::Point2f(rect.x, rect.y);

        for (unsigned i=0; i<tmp_keys.size(); i++)
          tmp_keys[i].pt += pt_offs;

        keys.insert(keys.end(), tmp_keys.begin(), tmp_keys.end());
        //cout<<"tile "<<v*param.tiles+u<<": "<<tmp_keys.size()<<" features"<<endl;  //DEBUG!!!!
      } 
    }
  } else (*orb)(im_gray, cv::Mat(), keys);
}

/**
 * detect
 */
void FeatureDetector_KD_FAST_IMGD::extract(const cv::Mat &image, std::vector<cv::KeyPoint> &keys, cv::Mat &descriptors)
{
  if( image.type() != CV_8U ) cv::cvtColor( image, im_gray, CV_RGB2GRAY );
  else im_gray = image;  

  imGDesc->compute(im_gray, keys, descriptors);

  if (param.do_feature_selection)
  {
    fs->dbg = image; 
    //cout<<"[FeatureDetector_KD_FAST_IMGD::extract] num detected: "<<keys.size()<<", "<<descriptors.rows<<endl;
    fs->compute(keys, descriptors); 
    //cout<<"[FeatureDetector_KD_FAST_IMGD::extract] num selected: "<<keys.size()<<", "<<descriptors.rows<<endl;
  }
}
}
