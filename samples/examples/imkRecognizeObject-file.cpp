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





#include <map>
#include <iostream>
#include <fstream>
#include <float.h>
#include <stdexcept>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/program_options.hpp>
#include <boost/format.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/time.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/centroid.h>
#include "pcl/common/transforms.h"
#include <v4r/recognition/IMKRecognizer.h>
#include <v4r/keypoints/impl/invPose.hpp>
#include <v4r/keypoints/impl/toString.hpp>
#include <v4r/keypoints/impl/PoseIO.hpp>
#include <v4r/reconstruction/impl/projectPointToImage.hpp>
#include <v4r_config.h>
#include <v4r/io/filesystem.h>
//#include <v4r/features/FeatureDetector_KD_FAST_IMGD.h>
#ifdef HAVE_SIFTGPU
#define USE_SIFT_GPU
#include <v4r/features/FeatureDetector_KD_SIFTGPU.h>
#else
#include <v4r/features/FeatureDetector_KD_CVSIFT.h>
#endif

#include <pcl/common/time.h>


using namespace std;
namespace po = boost::program_options;


void drawConfidenceBar(cv::Mat &im, const double &conf);
cv::Point2f drawCoordinateSystem(cv::Mat &im, const Eigen::Matrix4f &pose, const cv::Mat_<double> &intrinsic, const cv::Mat_<double> &dist_coeffs, double size, int thickness);
void convertImage(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, cv::Mat &image);

//--------------------------- default configuration -------------------------------


void InitParameter();

void InitParameter()
{
}

//------------------------------ helper methods -----------------------------------
static void onMouse( int event, int x, int y, int flags, void* );
void setup(int argc, char **argv);

//----------------------------- data containers -----------------------------------
cv::Mat_<cv::Vec3b> image;
cv::Mat_<cv::Vec3b> im_draw;
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

cv::Mat_<double> dist_coeffs;// = cv::Mat::zeros(4, 1, CV_64F);
cv::Mat_<double> intrinsic = cv::Mat_<double>::eye(3,3);
Eigen::Matrix4f pose=Eigen::Matrix4f::Identity();

int ul_lr=0;
int start=0, end_idx=10;
cv::Point track_win[2];
std::string cam_file;
string filenames, base_dir, codebook_filename;
std::vector<std::string> object_names;
std::vector<v4r::triple<std::string, double, Eigen::Matrix4f> > objects;
double thr_conf=0;

int live = -1;
bool loop = false;





/******************************************************************
 * MAIN
 */
int main(int argc, char *argv[] )
{
  int sleep = 0;
  char filename[PATH_MAX];
  track_win[0] = cv::Point(0, 0);
  track_win[1] = cv::Point(640, 480);

  // config
  InitParameter();

  setup(argc,argv);

  intrinsic(0,0)=intrinsic(1,1)=525;
  intrinsic(0,2)=320, intrinsic(1,2)=240;
  if (cam_file.size()>0)
  {
    cv::FileStorage fs( cam_file, cv::FileStorage::READ );
    fs["camera_matrix"] >> intrinsic;
    fs["distortion_coefficients"] >> dist_coeffs;
  }

  cv::namedWindow( "image", CV_WINDOW_AUTOSIZE );

  // init recognizer
  v4r::IMKRecognizer::Parameter param;
  param.pnp_param.eta_ransac = 0.01;
  param.pnp_param.max_rand_trials = 10000;
  param.pnp_param.inl_dist_px = 2;
  param.pnp_param.inl_dist_z = 0.02;
  param.vc_param.cluster_dist = 40;

  #ifdef USE_SIFT_GPU
  param.cb_param.nnr = 1.000001;
  param.cb_param.thr_desc_rnn = 0.25;
  param.cb_param.max_dist = FLT_MAX;
  v4r::FeatureDetector::Ptr detector(new v4r::FeatureDetector_KD_SIFTGPU());
  #else
  v4r::KeypointObjectRecognizer::Parameter param;
  param.cb_param.nnr = 1.000001;
  param.cb_param.thr_desc_rnn = 250.;
  param.cb_param.max_dist = FLT_MAX;
  v4r::FeatureDetector::Ptr detector(new v4r::FeatureDetector_KD_CVSIFT());
  #endif

//  // -- test imgd --
//  param.cb_param.nnr = 1.000001;
//  param.cb_param.thr_desc_rnn = 0.25;
//  param.cb_param.max_dist = FLT_MAX;
//  v4r::FeatureDetector_KD_FAST_IMGD::Parameter imgd_param(1000, 1.3, 4, 15);
//  v4r::FeatureDetector::Ptr detector(new v4r::FeatureDetector_KD_FAST_IMGD(imgd_param));
//  // -- end --

  v4r::IMKRecognizer recognizer(param, detector, detector);

  recognizer.setCameraParameter(intrinsic, dist_coeffs);
  recognizer.setDataDirectory(base_dir);
  if (!codebook_filename.empty())
        recognizer.setCodebookFilename(codebook_filename);

  if (object_names.size() == 0) { //take all direcotry names from the base_dir
      	object_names = v4r::io::getFoldersInDirectory(base_dir);
  }

  for (unsigned i=0; i<object_names.size(); i++)
    recognizer.addObject(object_names[i]);

  std::cout << "Number of models: " << object_names.size() << std::endl;
  recognizer.initModels();

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



  // ---------------------- recognize object ---------------------------

  for (int i=start; i<=end_idx || loop; i++)
  {
    cout<<"---------------- FRAME #"<<i<<" -----------------------"<<endl;
    if (live!=-1){
      cap >> image;
    } else {
      snprintf(filename,PATH_MAX, filenames.c_str(), i);
      cout<<filename<<endl;
      image = cv::Mat_<cv::Vec3b>();

      if (filenames.compare(filenames.size()-3,3,"pcd")==0)
      {
        if(pcl::io::loadPCDFile(filename, *cloud)==-1)
          continue;
        convertImage(*cloud,image);
      }
      else
      {
        image = cv::imread(filename, 1);
      }
    }

    image.copyTo(im_draw);
    recognizer.dbg = im_draw;

//    cloud->clear();

    // track
    { pcl::ScopeTime t("overall time");

    if (cloud->width!=(unsigned)image.cols || cloud->height!=(unsigned)image.rows)
    {
      recognizer.recognize(image, objects);
      cout<<"Use image only!"<<endl;
    }
    else
    {
      recognizer.recognize(*cloud, objects);
      cout<<"Use image and cloud!"<<endl;
    }

    } //-- overall time --

    // debug out draw
    //cv::addText( image, P::toString(1000./time)+std::string(" fps"), cv::Point(25,35), font);

    cout<<"Confidence value threshold for visualization: "<<thr_conf<<endl;
    cout<<"Found objects:"<<endl;
    for (unsigned j=0; j<objects.size(); j++)
    {
      cout<<j<<": name= "<<objects[j].first<<", conf= "<<objects[j].second<<endl;
      if (objects[j].second>=thr_conf)
      {
        cv::Point2f origin = drawCoordinateSystem(im_draw, objects[j].third, intrinsic, dist_coeffs, 0.1, 4);
        //cv::addText( im_draw, objects[i].first+std::string(" (")+v4r::toString(objects[i].second)<<std::string(")"), origin+cv::Point(0,-10), font);
        std::string txt = v4r::toString(j)+std::string(": ")+objects[j].first+std::string(" (")+v4r::toString(objects[j].second)+std::string(")");
        cv::putText(im_draw, txt, origin+cv::Point2f(0,10), cv::FONT_HERSHEY_PLAIN, 1.5,  CV_RGB(255,255,255), 2, CV_AA);
      }
    }

    cv::imshow("image",im_draw);


    int key = cv::waitKey(sleep);
    if (((char)key)==27) break;
    if (((char)key)=='r') sleep=1;
    if (((char)key)=='s') sleep=0;
  }


  cv::waitKey(0);


  
  return 0;
}



/******************************** SOME HELPER METHODS **********************************/




static void onMouse( int event, int x, int y, int flags, void* )
{
    if(x < 0 || x >= im_draw.cols || y < 0 || y >= im_draw.rows )
        return;
    if( event == CV_EVENT_LBUTTONUP && (flags & CV_EVENT_FLAG_LBUTTON) )
    {
      track_win[ul_lr%2] = cv::Point(x,y);
      if (ul_lr%2==0) cout<<"upper left corner: "<<track_win[ul_lr%2]<<endl;
      else cout<<"lower right corner: "<<track_win[ul_lr%2]<<endl;
      ul_lr++;
    }
}

/**
 * setup
 */
void setup(int argc, char **argv)
{
  po::options_description general("General options");
  general.add_options()
      ("help,h", "show help message")
      ("filenames,f", po::value<std::string>(&filenames)->default_value(filenames), "Input filename for recognition (printf-style)")
      ("base_dir,d", po::value<std::string>(&base_dir)->default_value(base_dir), "Object model directory")
      ("codebook_filename,c", po::value<std::string>(&codebook_filename), "Optional filename for codebook")
      ("object_names,n", po::value< std::vector<std::string> >(&object_names)->multitoken(), "Object names")
      ("start,s", po::value<int>(&start)->default_value(start), "start index")
      ("end,e", po::value<int>(&end_idx)->default_value(end_idx), "end index")
      ("cam_file,a", po::value<std::string>(&cam_file)->default_value(cam_file), "camera calibration files (opencv format)")
      ("live,l", po::value<int>(&live)->default_value(live), "use live camera (OpenCV)")
      ("thr_conf,t", po::value<double>(&thr_conf)->default_value(thr_conf), "Confidence value threshold (visualization)")
      ;

  po::options_description all("");
  all.add(general);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).
            options(all).run(), vm);
  po::notify(vm);
  std::string usage = "";

  if(vm.count("help"))
  {
      std::cout << usage << std::endl;
      std::cout << all;
      return;
  }

  return;
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



cv::Point2f drawCoordinateSystem(cv::Mat &im, const Eigen::Matrix4f &_pose, const cv::Mat_<double> &_intrinsic, const cv::Mat_<double> &_dist_coeffs, double size, int thickness)
{
  Eigen::Matrix3f R = _pose.topLeftCorner<3,3>();
  Eigen::Vector3f t = _pose.block<3, 1>(0,3);

  Eigen::Vector3f pt0 = R * Eigen::Vector3f(0,0,0) + t;
  Eigen::Vector3f pt_x = R * Eigen::Vector3f(size,0,0) + t;
  Eigen::Vector3f pt_y = R * Eigen::Vector3f(0,size,0) + t;
  Eigen::Vector3f pt_z = R * Eigen::Vector3f(0,0,size) +t ;

  cv::Point2f im_pt0, im_pt_x, im_pt_y, im_pt_z;

  if (!_dist_coeffs.empty())
  {
    v4r::projectPointToImage(&pt0[0], &_intrinsic(0), &_dist_coeffs(0), &im_pt0.x);
    v4r::projectPointToImage(&pt_x[0], &_intrinsic(0), &_dist_coeffs(0), &im_pt_x.x);
    v4r::projectPointToImage(&pt_y[0], &_intrinsic(0), &_dist_coeffs(0), &im_pt_y.x);
    v4r::projectPointToImage(&pt_z[0], &_intrinsic(0), &_dist_coeffs(0), &im_pt_z.x);
  }
  else
  {
    v4r::projectPointToImage(&pt0[0], &_intrinsic(0), &im_pt0.x);
    v4r::projectPointToImage(&pt_x[0], &_intrinsic(0), &im_pt_x.x);
    v4r::projectPointToImage(&pt_y[0], &_intrinsic(0), &im_pt_y.x);
    v4r::projectPointToImage(&pt_z[0], &_intrinsic(0), &im_pt_z.x);
  }

  cv::line(im, im_pt0, im_pt_x, CV_RGB(255,0,0), thickness);
  cv::line(im, im_pt0, im_pt_y, CV_RGB(0,255,0), thickness);
  cv::line(im, im_pt0, im_pt_z, CV_RGB(0,0,255), thickness);

  return im_pt0;
}




void convertImage(const pcl::PointCloud<pcl::PointXYZRGB> &_cloud, cv::Mat &_image)
{
  _image = cv::Mat_<cv::Vec3b>(_cloud.height, _cloud.width);

  for (unsigned v = 0; v < _cloud.height; v++)
  {
    for (unsigned u = 0; u < _cloud.width; u++)
    {
      cv::Vec3b &cv_pt = _image.at<cv::Vec3b> (v, u);
      const pcl::PointXYZRGB &pt = _cloud(u,v);

      cv_pt[2] = pt.r;
      cv_pt[1] = pt.g;
      cv_pt[0] = pt.b;
    }
  }
}




