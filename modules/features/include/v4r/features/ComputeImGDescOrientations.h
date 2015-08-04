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

#ifndef V4R_COMPUTE_GDESC_ORIENTATIONS_HH
#define V4R_COMPUTE_GDESC_ORIENTATIONS_HH

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <v4r/core/macros.h>

#include <v4r/features/ImGDescOrientation.h>


namespace v4r 
{

class V4R_EXPORTS ComputeImGDescOrientations
{
public:
  class Parameter
  {
  public:
    int win_size;
    ImGDescOrientation::Parameter goParam;
    Parameter(int _win_size=34, 
      const ImGDescOrientation::Parameter &_goParam=ImGDescOrientation::Parameter())
    : win_size(_win_size), goParam(_goParam) {}
  };

private:
  Parameter param;

  int h_win;

public:
 

  ComputeImGDescOrientations(const Parameter &p=Parameter());
  ~ComputeImGDescOrientations();

  void compute(const cv::Mat_<unsigned char> &image, const std::vector<cv::Point2f> &pts, 
        std::vector<cv::KeyPoint> &keys);
  void compute(const cv::Mat_<unsigned char> &image, std::vector<cv::KeyPoint> &keys);
  //void compute(const cv::Mat_<unsigned char> &image, std::vector<AffKeypoint> &keys);


  typedef SmartPtr< ::v4r::ComputeImGDescOrientations> Ptr;
  typedef SmartPtr< ::v4r::ComputeImGDescOrientations const> ConstPtr;

};



/*************************** INLINE METHODES **************************/



} //--END--

#endif

