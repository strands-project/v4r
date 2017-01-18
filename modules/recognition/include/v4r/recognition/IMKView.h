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

#ifndef KP_IMK_VIEW_HH
#define KP_IMK_VIEW_HH

#include <stdio.h>
#include <string>
#include <stdexcept>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Dense>
#include <v4r/common/impl/DataMatrix2D.hpp>


namespace v4r
{


class V4R_EXPORTS IMKView
{
public:
  unsigned object_id;
  std::vector< Eigen::Vector3f > points;
  std::vector< cv::KeyPoint > keys;

  DataMatrix2D<Eigen::Vector3f> cloud;
  cv::Mat_<float> weight_mask;
  std::vector<float> conf_desc;

  cv::Mat_<unsigned char> im_gray;
  IMKView(){}
  IMKView(const unsigned &_object_id) : object_id(_object_id) {}
};


} //--END--

#endif

