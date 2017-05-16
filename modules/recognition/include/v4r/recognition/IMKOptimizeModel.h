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


#ifndef KP_IMK_OPTIMIZE_MODEL_HH
#define KP_IMK_OPTIMIZE_MODEL_HH

#include <stdio.h>
#include <string>
#include <stdexcept>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <boost/shared_ptr.hpp>
#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <v4r/features/FeatureDetector.h>
#include "RansacSolvePnPdepth.h"



namespace v4r
{


/**
 * IMKOptimizeModel
 */
class V4R_EXPORTS IMKOptimizeModel
{
public:
  class Parameter
  {
  public:
    float eq_pose_dist;
    float nnr;
    RansacSolvePnPdepth::Parameter pnp_param;
    Parameter()
    : eq_pose_dist(0.01), nnr(0.8),
      pnp_param(RansacSolvePnPdepth::Parameter(1.5, 0.01, 5000, INT_MIN, 4, 0.015)){}
  };
  class Link
  {
  public:
    int idx_part;
    int idx_view;
    double conf;
    Eigen::Matrix4f pose;
    std::vector<cv::DMatch> matches;
    Link() {}
    Link(const int &_idx_part, const int &_idx_view, const double &_conf, const Eigen::Matrix4f &_pose)
      : idx_part(_idx_part), idx_view(_idx_view), conf(_conf), pose(_pose) {}
  };
  class View
  {
  public:
    int idx_part;
    int idx_view;
    std::string name;
    cv::Mat descs;
    std::vector<cv::KeyPoint> keys;
    std::vector<Eigen::Vector3f> points3d;
    Eigen::Matrix4f pose;
    std::vector<Link> links;
    View() {}
    ~View() {}
  };


private:
  Parameter param;

  cv::Mat_<double> dist_coeffs;
  cv::Mat_<double> intrinsic;
  
  cv::Mat image;
  cv::Mat_<cv::Vec3b> im_lab;
  std::vector< cv::Mat_<unsigned char> > im_channels;
  std::vector<cv::Point3f> cv_points_3d;
  std::vector< cv::Point2f > im_points;
  std::vector<float> depth_values;
  std::vector< int > inliers;

  std::vector<cv::KeyPoint> keys;
  std::vector<std::vector< cv::DMatch > > matches01, matches10;

  std::string base_dir;
  std::vector<std::string> object_names;
  std::vector< std::vector< boost::shared_ptr<View> > > object_views;

  v4r::FeatureDetector::Ptr detector;
  v4r::FeatureDetector::Ptr descEstimator;
  v4r::RansacSolvePnPdepth pnp;
  cv::Ptr<cv::DescriptorMatcher> matcher;

  void loadObject(const unsigned &idx);
  void addPoints3d(const std::vector<cv::KeyPoint> &keys, const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const cv::Mat_<unsigned char> &mask, View &view);
  bool loadObjectIndices(const std::string &_filename, cv::Mat_<unsigned char> &_mask, const cv::Size &_size);
  void convertImage(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, cv::Mat &_image);
  bool estimatePose(const View &view0, const View &view1, const std::vector<std::vector< cv::DMatch > > &matches, Eigen::Matrix4f &pose, std::vector<int> &_inliers);
  bool validatePoseAndAddLinks(View &view0, View &view1, const std::vector<std::vector< cv::DMatch > > &_matches01, const std::vector<std::vector< cv::DMatch > > &_matches10, const Eigen::Matrix4f &pose01, const Eigen::Matrix4f &pose10, const std::vector<int> &_inliers01, const std::vector<int> &_inliers10);
  void matchViews();

public:
  cv::Mat dbg;

  IMKOptimizeModel(const Parameter &p,
                           const v4r::FeatureDetector::Ptr &_detector,
                           const v4r::FeatureDetector::Ptr &_descEstimator);
  ~IMKOptimizeModel();

  void clear();
  void setDataDirectory(const std::string &_base_dir);
  void addObject(const std::string &_object_name);
  void loadAllObjectViews();
  void optimize();

  void setCameraParameter(const cv::Mat &_intrinsic, const cv::Mat &_dist_coeffs);

  typedef boost::shared_ptr< ::v4r::IMKOptimizeModel> Ptr;
  typedef boost::shared_ptr< ::v4r::IMKOptimizeModel const> ConstPtr;
};


/***************************** inline methods *******************************/



} //--END--

#endif

