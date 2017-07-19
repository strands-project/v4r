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


#ifndef KP_OBJECT_RECOGNIZER_HH
#define KP_OBJECT_RECOGNIZER_HH

#include <stdio.h>
#include <string>
#include <stdexcept>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <boost/shared_ptr.hpp>
#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <v4r/keypoints/CodebookMatcher.h>
#include <v4r/features/FeatureDetector.h>
#include <v4r/features/ImGradientDescriptor.h>
#include <v4r/keypoints/impl/triple.hpp>
#include "IMKView.h"
#include "IMKObjectVotesClustering.h"
#include "RansacSolvePnPdepth.h"



namespace v4r
{


/**
 * IMKRecognizer
 */
class V4R_EXPORTS IMKRecognizer
{
public:
  class Parameter
  {
  public:
    int use_n_clusters;
    double min_cluster_size;
    int image_size_conf_desc;
    CodebookMatcher::Parameter cb_param;
    IMKObjectVotesClustering::Parameter vc_param;
    RansacSolvePnPdepth::Parameter pnp_param;
    Parameter(int _use_n_clusters=10,
      const CodebookMatcher::Parameter &_cb_param=CodebookMatcher::Parameter(0.25, .98, 1.),
      const RansacSolvePnPdepth::Parameter &_pnp_param=RansacSolvePnPdepth::Parameter())
    : use_n_clusters(_use_n_clusters), min_cluster_size(5), image_size_conf_desc(66),
      cb_param(_cb_param), pnp_param(_pnp_param){}
  };


private:
  Parameter param;

  cv::Mat_<double> dist_coeffs;
  cv::Mat_<double> intrinsic;
  
  cv::Mat image;
  cv::Mat_<cv::Vec3b> im_lab;
  std::vector< cv::Mat_<unsigned char> > im_channels;
  cv::Mat_<unsigned char> im_warped_scaled;
  std::vector<cv::Mat_<unsigned char> > ims_warped;
  std::vector< cv::Point2f > im_points;
  std::vector< int > inliers;
  std::vector<int> cnt_view_matches;
  std::vector<float> desc;

  cv::Mat descs;
  std::vector<cv::KeyPoint> keys;
  std::vector< std::vector< cv::DMatch > > matches;
  std::vector< boost::shared_ptr<v4r::triple<unsigned, double, std::vector< cv::DMatch > > > > clusters; // <object_id, clustered matches>
  std::vector< cv::Point2f > query_pts;
  std::vector< cv::Point3f > model_pts;

  std::string base_dir;
  std::string codebookFilename;
  std::vector<std::string> object_names;
  std::vector<IMKView> object_models;

  CodebookMatcher::Ptr cbMatcher;
  v4r::FeatureDetector::Ptr detector;
  v4r::FeatureDetector::Ptr descEstimator;

  ImGradientDescriptor cp;

  v4r::IMKObjectVotesClustering votesClustering;
  v4r::RansacSolvePnPdepth pnp;

  void createObjectModel(const unsigned &idx);
  bool loadObjectIndices(const std::string &_filename, cv::Mat_<unsigned char> &_mask, const cv::Size &_size);
  void convertImage(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, cv::Mat &_image);
  void addView(const unsigned &idx, const std::vector<cv::KeyPoint> &keys, const cv::Mat &descs, const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const cv::Mat_<unsigned char> &mask, const Eigen::Matrix4f &pose, Eigen::Vector3d &centroid, unsigned &cnt);
  void poseEstimation(const std::vector< cv::Mat_<unsigned char> > &_im_channels, const std::vector<std::string> &object_names, const std::vector<IMKView> &views, const std::vector<cv::KeyPoint> &keys, const cv::Mat &descs,
                      const std::vector< std::vector< cv::DMatch > > &matches,
                      const std::vector< boost::shared_ptr<v4r::triple<unsigned, double, std::vector< cv::DMatch > > > > &clusters,
                      std::vector<v4r::triple<std::string, double, Eigen::Matrix4f> > &objects, const pcl::PointCloud<pcl::PointXYZRGB> &_cloud);
  int getMaxViewIndex(const std::vector<IMKView> &views, const std::vector<cv::DMatch> &matches, const std::vector<int> &inliers);
  void getNearestNeighbours(const Eigen::Vector2f &pt, const std::vector<cv::KeyPoint> &keys, const float &sqr_inl_radius_conf, std::vector<int> &nn_indices);
  float getMinDescDist32F(const cv::Mat &desc, const cv::Mat &descs, const std::vector<int> &indices);
  void setViewDescriptor(const std::vector< cv::Mat_<unsigned char> > &_im_channels, const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const cv::Mat_<unsigned char> &mask, const Eigen::Matrix4f &pose, IMKView &view);
  double computeGradientHistogramConf(const std::vector< cv::Mat_<unsigned char> > &_im_channels, const IMKView &view, const Eigen::Matrix4f &pose);




public:
  cv::Mat dbg;

  IMKRecognizer(const Parameter &p,
                           const v4r::FeatureDetector::Ptr &_detector,
                           const v4r::FeatureDetector::Ptr &_descEstimator);
  ~IMKRecognizer();

  void recognize(const cv::Mat &_image, std::vector<v4r::triple<std::string, double, Eigen::Matrix4f> > &objects);
  void recognize(const pcl::PointCloud<pcl::PointXYZRGB> &_cloud, std::vector<v4r::triple<std::string, double, Eigen::Matrix4f> > &objects);

  void clear();
  void setDataDirectory(const std::string &_base_dir);
  void setCodebookFilename(const std::string &_codebookFilename);
  void addObject(const std::string &_object_name);
  void initModels();

  void setCameraParameter(const cv::Mat &_intrinsic, const cv::Mat &_dist_coeffs);

  typedef boost::shared_ptr< ::v4r::IMKRecognizer> Ptr;
  typedef boost::shared_ptr< ::v4r::IMKRecognizer const> ConstPtr;
};


/***************************** inline methods *******************************/



} //--END--

#endif

