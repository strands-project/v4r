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

#define AO_DBG_IMAGES



#include <map>
#include <iostream>
#include <fstream>
#include <float.h>
#include <stdexcept>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <v4r/tracking/ObjectTrackerMono.h>
#include <v4r/keypoints/impl/convertImage.hpp>
#include <v4r/keypoints/ArticulatedObject.h>
#include <v4r/keypoints/impl/convertPose.hpp>
#include <v4r/keypoints/impl/invPose.hpp>
#include <v4r/keypoints/impl/toString.hpp>
#include <v4r/common/impl/SmartPtr.hpp>
#include <v4r/reconstruction/impl/projectPointToImage.hpp>
#include <v4r/keypoints/io.h>
#include <v4r/common/impl/ScopeTime.hpp>




using namespace std;
using namespace v4r;


static std::string log_dir("log/");

int live = -1;
int dbg = 0;


// helper methods
static void onMouse( int event, int x, int y, int flags, void* );
void setup(int argc, char **argv);
void drawConfidenceBar(cv::Mat &im, const double &conf);
void drawCoordinateSystem(cv::Mat &im, const Eigen::Matrix4f &pose, const cv::Mat_<double> &intrinsic, const cv::Mat_<double> &dist_coeffs, double size, int thickness);


// data containers
int ao_sleep=1;
int start=0, last_frame=100;
cv::Mat_<cv::Vec3b> image;
cv::Mat_<cv::Vec3b> im_draw;


cv::Mat_<double> dist_coeffs;// = cv::Mat::zeros(4, 1, CV_64F);
cv::Mat_<double> intrinsic = cv::Mat_<double>::eye(3,3);
cv::Mat_<double> src_dist_coeffs;
cv::Mat_<double> src_intrinsic;
Eigen::Matrix4f pose;
ArticulatedObject::Ptr model(new ArticulatedObject());

string model_file, filenames, cam_file, cam_file_model;
std::vector<Eigen::Vector3f> cam_trajectory;
bool log_cams=false;
bool log_files=false;



// libs
v4r::ObjectTrackerMono::Ptr tracker;





/******************************************************************
 * MAIN
 */
int main(int argc, char *argv[] )
{
  char filename[PATH_MAX];
  bool loop=false;

  setup(argc, argv);

  v4r::ObjectTrackerMono::Parameter param;
  param.kt_param.plk_param.use_ncc=true;
  param.kt_param.plk_param.ncc_residual=.5;
  tracker.reset(new v4r::ObjectTrackerMono(param));


  cv::VideoCapture cap;
  if (live!=-1) {
    cap.open(live);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
    loop = true;
    if( !cap.isOpened() ) {
      cout << "Could not initialize capturing...\n";
      return 0;
    }
  }
  

  cv::namedWindow( "image", CV_WINDOW_AUTOSIZE );

  intrinsic(0,0)=intrinsic(1,1)=525;
  intrinsic(0,2)=320, intrinsic(1,2)=240;

  if (cam_file.size()>0)
  {
    cv::FileStorage fs( cam_file, cv::FileStorage::READ );
    fs["camera_matrix"] >> intrinsic;
    fs["distortion_coefficients"] >> dist_coeffs;
  }
  if (cam_file_model.size()>0)
  {
    cv::FileStorage fs( cam_file_model, cv::FileStorage::READ );
    fs["camera_matrix"] >> src_intrinsic;
    fs["distortion_coefficients"] >> src_dist_coeffs;
  }

  if(!src_intrinsic.empty()) tracker->setObjectCameraParameter(src_intrinsic, src_dist_coeffs);
  tracker->setCameraParameter(intrinsic, dist_coeffs);




  // -------------------- load model ---------------------------
  cout<<"Load: "<<model_file<<endl;

  if(v4r::io::read(model_file, model))
  {
    tracker->setObjectModel(model);
  }
  else
  {
    cout<<"file not found!"<<endl;
    exit(1);
  }


  // ---------------------- track model --------------------------- 
  double time, conf;
  double mean_time=0;
  int cnt4time=0;

  for (int i=start; i<last_frame || loop; i++)
  {
    cout<<"---------------- FRAME #"<<i<<" -----------------------"<<endl;

    if (live!=-1){
      cap >> image;
    }else {
      snprintf(filename,PATH_MAX,filenames.c_str(),i);
      cout<<filename<<endl;
      image = cv::imread(filename, 1);
    }

    if (image.empty()) { cv::waitKey(200); continue; }

    image.copyTo(im_draw);

    bool is_ok = false;
    if (dbg) tracker->dbg = im_draw;

    { v4r::ScopeTime t("overall time");

      is_ok = tracker->track(image, pose, conf);

      time = t.getTime();
    }

    cnt4time++;
    mean_time += time;
    cout<<"tracker conf="<<conf<<endl;
    cout<<"mean_time="<<mean_time/double(cnt4time)<<" ("<<cnt4time<<")"<<endl;

    if (is_ok)cout<<"Status: stable tracking"<<endl;
    else if (conf>0.05) cout<<"Status: unstable tracking"<<endl;
    else cout<<"Status: object lost!"<<endl;

    if (conf>0.05 && i>start)
    {
      if (log_cams) cam_trajectory.push_back(pose.block<3, 1>(0,3));
      drawCoordinateSystem(im_draw, pose, intrinsic, dist_coeffs, 0.1, 4);
    } 

    drawConfidenceBar(im_draw, conf);

    cv::imshow("image", im_draw);


    int key = cv::waitKey(ao_sleep);
    if (((char)key)==27) break;
    if (((char)key)=='r') ao_sleep=1;
    if (((char)key)=='s') ao_sleep=0;
    if (((char)key)=='c') cam_trajectory.clear();
    if (((char)key)=='l') { log_cams = (!log_cams); cv::waitKey(100); }
    if (((char)key)=='r') { tracker->reset(); }
  }
	
	return 0;
}



/******************************** SOME HELPER METHODS **********************************/

/**
 * setup
 */
void setup(int argc, char **argv)
{
  int c;
  while(1)
  {
    c = getopt(argc, argv, "a:b:m:f:s:e:l:h");
    if(c == -1)
      break;
    switch(c)
    {
      case 'a':
        cam_file = optarg;
        break;
      case 'b':
        cam_file_model = optarg;
        break;
      case 'm':
        model_file = optarg;
        break;
      case 'f':
        filenames = optarg;
        break;
      case 's':
        start = std::atoi(optarg);
        break;
      case 'e':
        last_frame = std::atoi(optarg);
        break;
      case 'l':
        live = std::atoi(optarg);
        break;
      case 'h':
        printf("%s [-m model_file] [-f filenames] [-s start_idx] [-e end_idx] [-l camera_id] [-a query_camera_file.yml] [-b model_camera_file.yml] [-h]\n"
        "   -m model_file (pointcloud or stored model with .bin)\n"
        "   -f image filename (printf-style) or filename\n"
        "   -a camera calibration files for the tracking sequence (opencv format)\n"
        "   -b camera calibration files for the object model (optional)\n"
        "   -l use live stream of camera_id\n"
        "   -h help\n",  argv[0]);
        exit(1);

        break;
    }
  }
}

/**
 * drawConfidenceBar
 */
void drawConfidenceBar(cv::Mat &im, const double &conf)
{
  int bar_start = 50, bar_end = 200;
  int diff = bar_end-bar_start;
  int draw_end = diff*conf;
  double col_scale = 255./(double)diff;
  cv::Point2f pt1(0,30);
  cv::Point2f pt2(0,30);
  cv::Vec3b col(0,0,0);

  if (draw_end<=0) draw_end = 1;

  for (int i=0; i<draw_end; i++)
  {
    col = cv::Vec3b(255-(i*col_scale), i*col_scale, 0);
    pt1.x = bar_start+i;
    pt2.x = bar_start+i+1;
    cv::line(im, pt1, pt2, CV_RGB(col[0],col[1],col[2]), 8);
  }
}

/**
 * @brief drawCoordinateSystem
 * @param im
 * @param pose
 * @param intrinsic
 * @param dist_coeffs
 */
void drawCoordinateSystem(cv::Mat &im, const Eigen::Matrix4f &pose, const cv::Mat_<double> &intrinsic, const cv::Mat_<double> &dist_coeffs, double size, int thickness)
{
  Eigen::Matrix3f R = pose.topLeftCorner<3,3>();
  Eigen::Vector3f t = pose.block<3, 1>(0,3);

  Eigen::Vector3f pt0 = R * Eigen::Vector3f(0,0,0) + t;
  Eigen::Vector3f pt_x = R * Eigen::Vector3f(size,0,0) + t;
  Eigen::Vector3f pt_y = R * Eigen::Vector3f(0,size,0) + t;
  Eigen::Vector3f pt_z = R * Eigen::Vector3f(0,0,size) +t ;

  cv::Point2f im_pt0, im_pt_x, im_pt_y, im_pt_z;

  if (!dist_coeffs.empty())
  {
    v4r::projectPointToImage(&pt0[0], &intrinsic(0), &dist_coeffs(0), &im_pt0.x);
    v4r::projectPointToImage(&pt_x[0], &intrinsic(0), &dist_coeffs(0), &im_pt_x.x);
    v4r::projectPointToImage(&pt_y[0], &intrinsic(0), &dist_coeffs(0), &im_pt_y.x);
    v4r::projectPointToImage(&pt_z[0], &intrinsic(0), &dist_coeffs(0), &im_pt_z.x);
  }
  else
  {
    v4r::projectPointToImage(&pt0[0], &intrinsic(0), &im_pt0.x);
    v4r::projectPointToImage(&pt_x[0], &intrinsic(0), &im_pt_x.x);
    v4r::projectPointToImage(&pt_y[0], &intrinsic(0), &im_pt_y.x);
    v4r::projectPointToImage(&pt_z[0], &intrinsic(0), &im_pt_z.x);
  }

  cv::line(im, im_pt0, im_pt_x, CV_RGB(255,0,0), thickness);
  cv::line(im, im_pt0, im_pt_y, CV_RGB(0,255,0), thickness);
  cv::line(im, im_pt0, im_pt_z, CV_RGB(0,0,255), thickness);
}












