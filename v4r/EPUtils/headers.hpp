#ifndef EPUTILS_MODULE_HEADERS_HPP
#define EPUTILS_MODULE_HEADERS_HPP

#include <string>
#include <iostream>
#include <stdio.h>
#include <stdarg.h>
#include <time.h>

#include <boost/filesystem/fstream.hpp>
#include <boost/filesystem.hpp>

#include <opencv2/opencv.hpp>

#include <eigen3/Eigen/Eigen>

// #ifndef NOT_USE_PCL
#include <pcl/ModelCoefficients.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include "PCLPreprocessingXYZRC.hpp"
// #endif

namespace EPUtils
{
  
static const int dy8[8] = {-1,-1,-1,0,1,1,1,0};
static const int dx8[8] = {-1,0,1,1,1,0,-1,-1};

static const int dx4[4] = {-1,1,0,0};
static const int dy4[4] = {0,0,-1,1};

} //namespace EPUtils

#endif //EPUTILS_MODULE_HEADERS_HPP
