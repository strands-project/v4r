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
    : cluster_dist(70) {}
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

