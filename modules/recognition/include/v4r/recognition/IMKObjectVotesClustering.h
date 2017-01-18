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

#ifndef KP_IMK_OBJECT_VOTES_CLUSTERING_HH
#define KP_IMK_OBJECT_VOTES_CLUSTERING_HH

#include <stdio.h>
#include <string>
#include <stdexcept>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <boost/shared_ptr.hpp>
#include <Eigen/Dense>
#include <v4r/recognition/IMKView.h>
#include <v4r/common/ClusteringRNN.h>
#include <v4r/keypoints/impl/triple.hpp>


namespace v4r
{


/**
 * IMKObjectVotesClustering
 */
class V4R_EXPORTS IMKObjectVotesClustering
{
public:
  class Parameter
  {
  public:
    double cluster_dist;
    Parameter()
    : cluster_dist(50) {}
  };

private:
  Parameter param;

  std::vector<std::vector<int> > indices;
  DataMatrix2Df votes;
  std::vector< cv::DMatch > voting_matches;
  std::vector< double > weights;

  ClusteringRNN rnn;

  static const float two_pi;

  void createVotes(unsigned id, const std::vector<IMKView> &views, const std::vector<cv::KeyPoint> &keys, const std::vector< std::vector< cv::DMatch > > &matches, DataMatrix2Df &votes, std::vector< cv::DMatch > &voting_matches, std::vector<double> &weights);
  inline float diffAngle_0_2pi(float b, float a);
  inline float scaleAngle_0_2pi(float a);

public:
  cv::Mat dbg;

  IMKObjectVotesClustering(const Parameter &p = Parameter());
  ~IMKObjectVotesClustering();

  void operate(const std::vector<std::string> &object_names, const std::vector<IMKView> &views, const std::vector<cv::KeyPoint> &keys, const std::vector< std::vector< cv::DMatch > > &matches,  std::vector< boost::shared_ptr<v4r::triple<unsigned, double, std::vector< cv::DMatch > > > > &clusters);

  void setParameter(const Parameter &_param) {param = _param;}

  typedef boost::shared_ptr< ::v4r::IMKObjectVotesClustering> Ptr;
  typedef boost::shared_ptr< ::v4r::IMKObjectVotesClustering const> ConstPtr;
};


/***************************** inline methods *******************************/

inline float IMKObjectVotesClustering::diffAngle_0_2pi(float b, float a)
{
  return IMKObjectVotesClustering::scaleAngle_0_2pi(b - a);
}

inline float IMKObjectVotesClustering::scaleAngle_0_2pi(float a)
{
  while(a >= two_pi) a -= two_pi;
  while(a < 0.) a += two_pi;
  return a;
}

} //--END--

#endif

