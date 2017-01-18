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
#include "RansacSolvePnP.h"



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
    RansacSolvePnP::Parameter pnp_param;
    Parameter(int _use_n_clusters=10,
      const CodebookMatcher::Parameter &_cb_param=CodebookMatcher::Parameter(0.25, .98, 1.),
      const RansacSolvePnP::Parameter &_pnp_param=RansacSolvePnP::Parameter())
    : use_n_clusters(_use_n_clusters), min_cluster_size(5), image_size_conf_desc(66),
      cb_param(_cb_param), pnp_param(_pnp_param){}
  };


private:
  Parameter param;

  cv::Mat_<double> dist_coeffs;
  cv::Mat_<double> intrinsic;
  
  cv::Mat_<unsigned char> im_gray, im_warped, im_warped_scaled;
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
  std::vector<std::string> object_names;
  std::vector<IMKView> object_models;

  CodebookMatcher::Ptr cbMatcher;
  v4r::FeatureDetector::Ptr detector;
  v4r::FeatureDetector::Ptr descEstimator;

  ImGradientDescriptor cp;

  v4r::IMKObjectVotesClustering votesClustering;
  v4r::RansacSolvePnP pnp;

  void createObjectModel(const unsigned &idx);
  void convertImage(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, cv::Mat &image);
  void addView(const unsigned &idx, const std::vector<cv::KeyPoint> &keys, const cv::Mat &descs, const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const cv::Mat_<unsigned char> &mask, const Eigen::Matrix4f &pose, Eigen::Vector3d &centroid, unsigned &cnt);
  void poseEstimation(const cv::Mat_<unsigned char> &im_gray, const std::vector<std::string> &object_names, const std::vector<IMKView> &views, const std::vector<cv::KeyPoint> &keys, const cv::Mat &descs,
                      const std::vector< std::vector< cv::DMatch > > &matches,
                      const std::vector< boost::shared_ptr<v4r::triple<unsigned, double, std::vector< cv::DMatch > > > > &clusters,
                      std::vector<v4r::triple<std::string, double, Eigen::Matrix4f> > &objects);
  int getMaxViewIndex(const std::vector<IMKView> &views, const std::vector<cv::DMatch> &matches, const std::vector<int> &inliers);
  void getNearestNeighbours(const Eigen::Vector2f &pt, const std::vector<cv::KeyPoint> &keys, const float &sqr_inl_radius_conf, std::vector<int> &nn_indices);
  float getMinDescDist32F(const cv::Mat &desc, const cv::Mat &descs, const std::vector<int> &indices);
  void setViewDescriptor(const cv::Mat_<unsigned char> &im_gray, const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const cv::Mat_<unsigned char> &mask, const Eigen::Matrix4f &pose, IMKView &view);
  double computeGradientHistogramConf(const cv::Mat_<unsigned char> &im_gray, const IMKView &view, const Eigen::Matrix4f &pose);




public:
  cv::Mat dbg;

  IMKRecognizer(const Parameter &p,
                           const v4r::FeatureDetector::Ptr &_detector,
                           const v4r::FeatureDetector::Ptr &_descEstimator);
  ~IMKRecognizer();

  void recognize(const cv::Mat &image, std::vector<v4r::triple<std::string, double, Eigen::Matrix4f> > &objects);

  void clear();
  void setDataDirectory(const std::string &_base_dir);
  void addObject(const std::string &_object_name);
  void initModels();

  void setCameraParameter(const cv::Mat &_intrinsic, const cv::Mat &_dist_coeffs);

  typedef boost::shared_ptr< ::v4r::IMKRecognizer> Ptr;
  typedef boost::shared_ptr< ::v4r::IMKRecognizer const> ConstPtr;
};


/***************************** inline methods *******************************/



} //--END--

#endif

