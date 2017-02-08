/******************************************************************************
 * Copyright (c) 2016 Johann Prankl
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 ******************************************************************************/


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
      (void)version;
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
