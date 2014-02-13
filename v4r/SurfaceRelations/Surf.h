/**
 * @file Surf.h
 * @author Richtsfeld
 * @date December 2011
 * @version 0.1
 * @brief Calculate surf features to compare surface texture.
 */

#ifndef SURFACE_SURF_H
#define SURFACE_SURF_H

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/legacy/compat.hpp>

//#include <utility>
// #include "v4r/PCLAddOns/PCLCommonHeaders.h"

namespace surface
{

class Surf
{
public:
// EIGEN_MAKE_ALIGNED_OPERATOR_NEW     /// for 32-bit systems for pcl mandatory
  
protected:

private:

  bool have_input_image;
  int width;                              // image width
  int height;                             // image height
  IplImage *gray_image;                   // gray-level image

  CvSURFParams params;                    // SURF parameters
  
  CvMemStorage* storage_compute;          // Storage memory for surf calculations
  CvSeq *keypoints, *descriptors;         // Keypoints and descriptors for compute()
  CvMemStorage* storage_compare_1;        // Storage memory for surf calculations
  CvMemStorage* storage_compare_2;        // Storage memory for surf calculations
  CvSeq *keypoints_1, *keypoints_2;       // Keypoints of surf features
  CvSeq *descriptors_1, *descriptors_2;   // Descriptors of surf features

  std::vector< std::map<double, unsigned> > matches;  
  std::vector<unsigned> first;            // First id
  std::vector<unsigned> second;           // second id
  std::vector<double> surfDistance;       // distances between first and second
  
  double compareSURFDescriptors(const float* descriptor_1, 
                                const float* descriptor_2, 
                                int descriptorsCount,
                                float lastMinSquaredDistance);
  
  int findNaiveNearestNeighbor(const float* descriptor_1, 
                               const CvSURFPoint* keypoint_1, 
                               CvSeq* descriptors_2, 
                               CvSeq* keypoints_2,
                               double &distance);
  
public:
  Surf();
  ~Surf();
  
  /** Set input image **/
  void setInputImage(IplImage *_image);

  /** Compute the surf features **/
  void compute();
  
  /** Compare surfaces **/
  double compare(std::vector<int> indices_0, std::vector<int> indices_1);
  
  /** Draw surf features into image and draw it **/
  void drawFeatures(cv::Mat_<cv::Vec3b> &_feature_image);

};

/*************************** INLINE METHODES **************************/

} //--END--

#endif

