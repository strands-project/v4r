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

#ifndef KEYPOINT_OBJECT_RECOGNIZER_BOOST_SERIALIZATION
#define KEYPOINT_OBJECT_RECOGNIZER_BOOST_SERIALIZATION

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/base_object.hpp>
#include <v4r/keypoints/impl/pair_serialization.hpp>
#include <v4r/keypoints/impl/eigen_boost_serialization.hpp>
#include <v4r/keypoints/impl/opencv_serialization.hpp>
#include <v4r/keypoints/impl/triple_serialization.hpp>
#include <v4r/recognition/IMKView.h>


 

/** View serialization **/
namespace boost{namespace serialization{

  template<class Archive>
  void serialize(Archive & ar, v4r::IMKView &view, const unsigned int version)
  {
    ar & view.object_id;
    ar & view.points;
    ar & view.keys;
    ar & view.cloud.type;
    ar & view.cloud.rows;
    ar & view.cloud.cols;
    ar & view.cloud.data;
    ar & view.weight_mask;
    ar & view.conf_desc;
  }

}}


#endif
