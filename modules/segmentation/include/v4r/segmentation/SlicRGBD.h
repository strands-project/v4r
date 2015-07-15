#ifndef V4R_SLIC_RGBD_H
#define V4R_SLIC_RGBD_H


#include <vector>
#include <string>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>


namespace v4r
{

/**
 * Data structure
 */
class SlicRGBDPoint
{
public:
  double x, y;
  double l, a, b;
  Eigen::Vector3d pt;
  Eigen::Vector3d n;
  SlicRGBDPoint() : x(0), y(0), l(0), a(0), b(0), pt(0,0,0), n(0,0,0) {}
  SlicRGBDPoint(const int &_x, const int &_y, const cv::Vec3b &_lab, const Eigen::Vector3d &_pt, const Eigen::Vector3d &_n)
  : x(_x), y(_y), pt(_pt), n(_n) {
    l = _lab[0], a = _lab[1], b = _lab[2];
  }
};

/**
 * SlicRGBD
 */
class SlicRGBD  
{
public:
  class Parameter
  {
  public:
    int superpixelsize;
    double compactness_image;
    double compactness_xyz;
    double weight_diff_normal_angle;
    double normals_max_depth_change_factor;//0.02f);
    double normals_smoothing_size;//20.0f);
    bool normals_depth_dependent_smoothing;
    Parameter(int _superpixelsize=100, double _compactness_image=10, double _compactness_xyz=2000,
              double _weight_diff_normal_angle=1500,
              double _normals_max_depth_change_factor=0.02, double _normals_smoothing_size=15.,
              bool _normals_depth_dependent_smoothing=true)
      : superpixelsize(_superpixelsize), compactness_image(_compactness_image), 
        compactness_xyz(_compactness_xyz), weight_diff_normal_angle(_weight_diff_normal_angle),
        normals_max_depth_change_factor(_normals_max_depth_change_factor), normals_smoothing_size(_normals_smoothing_size),
        normals_depth_dependent_smoothing(_normals_depth_dependent_smoothing){}
  };

private:
  Parameter param;
  int num_superpixel;

  cv::Mat_<cv::Vec3d> im_lab;
  cv::Mat grad_x, grad_y;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
  pcl::PointCloud<pcl::Normal>::Ptr normals;
  std::vector<double> dists;
  std::vector<SlicRGBDPoint> seeds;
  std::vector<SlicRGBDPoint> sigma;
  std::vector<double> clustersize;
  

  void performSlicRGBD(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const pcl::PointCloud<pcl::Normal> &normals, const cv::Mat_<cv::Vec3d> &im_lab,
                       const std::vector<bool> &valid, std::vector<SlicRGBDPoint> &seeds, cv::Mat_<int> &labels, const int &step);
  void getSeeds(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const pcl::PointCloud<pcl::Normal> &normals, const cv::Mat_<cv::Vec3d> &im_lab,
                const std::vector<bool> &valid, std::vector<SlicRGBDPoint> &seeds, const int &step);
  void getSeeds2(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const pcl::PointCloud<pcl::Normal> &normals, const cv::Mat_<cv::Vec3d> &im_lab,
                const std::vector<bool> &valid, std::vector<SlicRGBDPoint> &seeds, const int &step);
  void enforceLabelConnectivity(cv::Mat_<int> &labels, cv::Mat_<int> &out_labels, int& numlabels, const int& K);

  static void convertRGBtoLAB(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, cv::Mat_<cv::Vec3d> &im_lab);
  inline bool isnan(const Eigen::Vector3f &pt);
  inline double sqr(const double &v);

public:

  std::vector<bool> valid;
  SlicRGBD(const Parameter &p=Parameter());
	~SlicRGBD();

  /** segment superpixel **/
  void segmentSuperpixel(cv::Mat_<int> &labels, int& numlabels);

  /** set the desired number of superpixel (overrides param.superpixelsize) **/
  void setNumberOfSuperpixel(int N);

  /** set the superpixel size **/
  void setSuperpixeSize(int size);

  /** set input data **/
  void setCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &_cloud, const pcl::PointCloud<pcl::Normal>::Ptr &_normals = pcl::PointCloud<pcl::Normal>::Ptr());

  /** returns the CIE Lab image (segmentXX needs to be called before) **/
  cv::Mat_<cv::Vec3d> &getImageLAB() {return im_lab;}

  /** draw the contours **/
  void drawContours(cv::Mat_<cv::Vec3b> &im_rgb, const cv::Mat_<int> &labels, int r=-1, int g=-1, int b=-1);


};

inline bool SlicRGBD::isnan(const Eigen::Vector3f &pt)
{
  if (std::isnan(pt[0]) || std::isnan(pt[1]) || std::isnan(pt[2]))
    return true;
  return false;
}

inline double SlicRGBD::sqr(const double &v)
{
  return v*v;
}

}

#endif // -- THE END --
