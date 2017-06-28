/**
 * $Id$
 *
 * @author Johann Prankl
 *
 */

#ifndef KP_TSF_OPTIMIZE_CLOUD_H
#define KP_TSF_OPTIMIZE_CLOUD_H

#include <iostream>
#include <fstream>
#include <float.h>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <boost/thread/mutex.hpp>
#include <boost/thread.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/io.h>
#include <boost/shared_ptr.hpp>
#include <v4r/common/impl/DataMatrix2D.hpp>
#include <v4r/camera_tracking_and_mapping/Surfel.hh>
#include <list>
#include <v4r/keypoints/impl/triple.hpp>


#include <v4r/core/macros.h>


namespace v4r
{



/**
 * TSFOptimizeClouds
 */
class V4R_EXPORTS TSFOptimizeClouds
{
public:

  /**
   * Parameter
   */
  class Parameter
  {
  public:
    int batch_size_clouds;
    double cam_distance_select_frame; //[m]
    double angle_select_frame;        //[deg]
    float inv_depth_cut_off;
    Parameter()
      : batch_size_clouds(11), cam_distance_select_frame(0.001), angle_select_frame(0.1), inv_depth_cut_off(0.01) {}
  };

 

private:
  Parameter param;

  double sqr_cam_distance_select_frame, cos_delta_angle_select_frame;
  int width, height;

  static std::vector<cv::Vec4i> npat;

  cv::Mat_<double> intrinsic, dist_coeffs;
  cv::Mat_<double> tgt_intrinsic;

  boost::mutex mtx_shm;
  double sf_timestamp;
  Eigen::Matrix4f sf_pose;
  v4r::DataMatrix2D<v4r::Surfel> sf_cloud;
  std::list< v4r::triple< pcl::PointCloud<pcl::PointXYZRGB>::Ptr, Eigen::Matrix4f, double > > frames;

  pcl::PointCloud<pcl::PointXYZRGB> tmp_cloud;
  cv::Mat_<float> depth;
  cv::Mat_<float> depth_weight;
  cv::Mat_<cv::Vec3f> im_bgr;

  bool run, have_thread;

  boost::thread th_obectmanagement;
  boost::thread th_init;

  void operate();

  bool selectFrame(const Eigen::Matrix4f &pose0, const Eigen::Matrix4f &pose1);
  void integrateData(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const Eigen::Matrix4f &pose);
  void integrateDataRGB(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const Eigen::Matrix4f &pose);
  void project3D(v4r::DataMatrix2D<v4r::Surfel> &sf_cloud, const float &px_offs);
  void initKeyframe(const pcl::PointCloud<pcl::PointXYZRGB> &cloud0);

  inline float sqr(const float &d) {return d*d;}


public:
  cv::Mat dbg;

  TSFOptimizeClouds(const Parameter &p=Parameter());
  ~TSFOptimizeClouds();

  void start();
  void stop();

  inline bool isStarted() {return have_thread;}

  void reset();

  static void computeRadius(v4r::DataMatrix2D<Surfel> &sf_cloud, const cv::Mat_<double> &intrinsic);
  static void computeNormals(v4r::DataMatrix2D<Surfel> &sf_cloud, int nb_dist=1);

  void addCloud(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const Eigen::Matrix4f &pose, const double &timestamp, bool have_track);

  void getSurfelCloud(v4r::DataMatrix2D<Surfel> &cloud, Eigen::Matrix4f &pose, double &timestamp);
  void getFilteredCloudNormals(pcl::PointCloud<pcl::PointXYZRGBNormal> &cloud, Eigen::Matrix4f &pose, double &timestamp);

  void setCameraParameter(const cv::Mat &_intrinsic, const cv::Mat &_dist_coeffs=cv::Mat());
  void setCameraParameterTgt(const cv::Mat &_intrinsic, int _width, int _height);
  void setParameter(const Parameter &p);

  typedef boost::shared_ptr< ::v4r::TSFOptimizeClouds> Ptr;
  typedef boost::shared_ptr< ::v4r::TSFOptimizeClouds const> ConstPtr;
};



/*************************** INLINE METHODES **************************/



} //--END--

#endif

