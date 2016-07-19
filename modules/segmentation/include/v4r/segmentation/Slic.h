#ifndef V4R_SLIC_H
#define V4R_SLIC_H


#include <vector>
#include <string>
#include <algorithm>
#include <opencv2/core/core.hpp>

#include <v4r/core/macros.h>

namespace v4r
{

/**
 * Data structure
 */
class V4R_EXPORTS SlicPoint
{
public:
  double x, y;
  double l, a, b;
  SlicPoint() : x(0), y(0), l(0), a(0), b(0) {};
  SlicPoint(const int &_x, const int &_y, const cv::Vec3b &_lab) : x(_x), y(_y) {
    l = _lab[0], a = _lab[1], b = _lab[2];
  }
};

/**
 * Slic
 */
class V4R_EXPORTS Slic
{
private:

  cv::Mat_<cv::Vec3d> im_lab;
  std::vector<double> dists;
  std::vector<SlicPoint> seeds;
  std::vector<SlicPoint> sigma;
  std::vector<double> clustersize;
  

  void performSlic(const cv::Mat_<cv::Vec3d> &im_lab, std::vector<SlicPoint> &seeds, cv::Mat_<int> &labels, 
        const int &step, const double &m);
  void getSeeds(const cv::Mat_<cv::Vec3d> &im_lab, std::vector<SlicPoint> &_seeds, const int &step);
  void enforceLabelConnectivity(cv::Mat_<int> &labels, cv::Mat_<int> &out_labels, int& numlabels, const int& K);

public:
	Slic();
	~Slic();

  /** segment superpixel given a desired size **/
  void segmentSuperpixelSize(const cv::Mat_<cv::Vec3b> &im_rgb,
        cv::Mat_<int> &labels, int &numlabels, const int &superpixelsize, const double& compactness);

  /** segment superpixel given a desired number of superpixel **/
  void segmentSuperpixelNumber(const cv::Mat_<cv::Vec3b> &im_rgb,
        cv::Mat_<int> &labels, int& numlabels, const int& K, const double& compactness);

  /** returns the CIE Lab image (segmentXX needs to be called before) **/
  cv::Mat_<cv::Vec3d> &getImageLAB() {return im_lab;}

  /** draw the contours **/
  void drawContours(cv::Mat_<cv::Vec3b> &im_rgb, const cv::Mat_<int> &labels, int r=-1, int g=-1, int b=-1);


  static void convertRGBtoLAB(const cv::Mat_<cv::Vec3b> &im_rgb, cv::Mat_<cv::Vec3d> &im_lab);
  static void convertRGBtoLAB(const double r, const double g, const double b, double &labL, double &labA, double &labB);

};

}

#endif // -- THE END --
