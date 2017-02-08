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
 * @author Johann Prankl
 *
 */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <queue>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/time.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>

#include "v4r/common/convertImage.h"
#include "v4r/keypoints/temporal_smoothing_filter.h"




using namespace std;




//------------------------------ helper methods -----------------------------------
void setup(int argc, char **argv);

//----------------------------- data containers -----------------------------------
cv::Mat_<cv::Vec3b> image;
cv::Mat_<cv::Vec3b> im_draw;
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);



cv::Mat_<double> distCoeffs;// = cv::Mat::zeros(4, 1, CV_64F);
cv::Mat_<double> intrinsic = cv::Mat_<double>::eye(3,3);
Eigen::Matrix4f pose=Eigen::Matrix4f::Identity();

int start=0, idx_end=174;
cv::Point track_win[2];
std::string cam_file;
std::string path_to_vgn_file, path_to_crf_file;
string filenames;
unsigned global_map_size = 0;







/******************************************************************
 * MAIN
 */
int main(int argc, char *argv[] )
{
  int sleep = 0;
  char filename[PATH_MAX];
  bool loop=false;
  Eigen::Matrix4f _pose;//, inv_pose;
//  int id_cloud = -1;
//  double time, mean_time=0;
//  int cnt_time=0;
  int vis_mode = 0;

  setup(argc,argv);

  intrinsic(0,0)=intrinsic(1,1)=525;
  intrinsic(0,2)=320, intrinsic(1,2)=240;

  if (cam_file.size()>0)
  {
    cv::FileStorage fs( cam_file, cv::FileStorage::READ );
    fs["camera_matrix"] >> intrinsic;
    fs["distortion_coefficients"] >> distCoeffs;
  }

  cv::namedWindow( "image", CV_WINDOW_AUTOSIZE );

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
  pcl::PointCloud<pcl::PointXYZRGBNormal> filt_cloud;
//  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr global_cloud;

  v4r::TemporalSmoothingFilter filter;
  filter.setCameraParameter(intrinsic);

  v4r::TemporalSmoothingFilter::Parameter param;
  param.global_map_size = global_map_size;
  filter.setParameter(param);

  for (int i=start; i<=idx_end || loop; i++)
  {
    cout<<"---------------- FRAME #"<<i<<" -----------------------"<<endl;
    snprintf(filename,PATH_MAX, filenames.c_str(), i);
    cout<<filename<<endl;

    if(pcl::io::loadPCDFile(filename, *_cloud)==-1)
      continue;

    v4r::convertImage(*_cloud, image);
    image.copyTo(im_draw);

    // track
    filter.filter(*_cloud, filt_cloud, _pose);


    // debug out draw
    cv::imshow("image",im_draw);

    int key = cv::waitKey(sleep);
    if (((char)key)==27) break;
    if (((char)key)=='r') sleep=1;
    if (((char)key)=='s') sleep=0;
    if (((char)key)=='v') vis_mode = (vis_mode==1?0:1);
  }

  cv::waitKey(0);

  return 0;
}



/******************************** SOME HELPER METHODS **********************************/





/**
 * setup
 */
void setup(int argc, char **argv)
{
  cv::FileStorage fs;
  int c;
  while(1)
  {
    c = getopt(argc, argv, "f:s:e:a:g:c:v:h");
    if(c == -1)
      break;
    switch(c)
    {
      case 'f':
        filenames = optarg;
        break;
      case 's':
        start = std::atoi(optarg);
        break;
      case 'e':
        idx_end = std::atoi(optarg);
        break;
      case 'a':
        cam_file = optarg;
        break;
      case 'g':
        global_map_size = atoi(optarg);
        break;
 
      case 'h':
        printf("%s [-f filenames] [-s start_idx] [-e end_idx] [-a cam_file.yml] [-g global_map_size] [-h]\n"
        "   -f cloud filename (printf-style)\n"
        "   -a camera calibration files (opencv format)\n"
        "   -g log global map\n"
        "   -h help\n",  argv[0]);
        exit(1);

        break;
    }
  }
}













