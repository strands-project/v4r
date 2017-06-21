#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <queue>
#include <unistd.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/openni_grabber.h>
#include <pcl/common/time.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/io/ply_io.h>
#include <pcl/conversions.h>

#include "pcl/common/transforms.h"

#include "v4r/keypoints/impl/PoseIO.hpp"
#include "v4r/keypoints/impl/invPose.hpp"
#include "v4r/camera_tracking_and_mapping/TSFVisualSLAM.h"
#include "v4r/camera_tracking_and_mapping/TSFData.h"
#include "v4r/reconstruction/impl/projectPointToImage.hpp"
#include "v4r/features/FeatureDetector_KD_FAST_IMGD.h"
#include "v4r/camera_tracking_and_mapping/TSFGlobalCloudFilteringSimple.h"
#include "v4r/io/filesystem.h"






using namespace std;





//------------------------------ helper methods -----------------------------------
void setup(int argc, char **argv);

void convertImage(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, cv::Mat &image);
void drawCoordinateSystem(cv::Mat &im, const Eigen::Matrix4f &pose, const cv::Mat_<double> &intrinsic, const cv::Mat_<double> &dist_coeffs, double size, int thickness);
void drawConfidenceBar(cv::Mat &im, const double &conf, int x_start=50, int x_end=200, int y=30);



//----------------------------- data containers -----------------------------------
cv::Mat_<cv::Vec3b> image;
cv::Mat_<cv::Vec3b> im_draw;
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

std::string in_dir;
std::string cam_file, filenames, file_mesh, file_cloud;
std::string pattern =  std::string(".*.")+std::string("pcd");

cv::Mat_<double> distCoeffs;// = cv::Mat::zeros(4, 1, CV_64F);
cv::Mat_<double> intrinsic = cv::Mat_<double>::eye(3,3);
cv::Mat_<double> dist_coeffs_opti = cv::Mat::zeros(4, 1, CV_64F);
cv::Mat_<double> intrinsic_opti = cv::Mat_<double>::eye(3,3);

Eigen::Matrix4f pose;
float voxel_size = 0.003;
double thr_weight = 10;      //e.g. 10    // surfel threshold for the final model
double thr_delta_angle = 75; // e.g. 80
int display = true;

cv::Point track_win[2];



/******************************************************************
 * MAIN
 */
int main(int argc, char *argv[] )
{
  int sleep = 0;
  //char filename[PATH_MAX];
  bool loop=false;
  Eigen::Matrix4f inv_pose;
  double time, mean_time=0;
  int cnt_time=0;

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

  std::vector< std::pair<Eigen::Matrix4f, int> > all_poses; //<pose, kf_index>

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
  pcl::PointCloud<pcl::PointXYZRGBNormal> filt_cloud;
  Eigen::Matrix4f filt_pose;
  uint64_t timestamp;
  bool have_pose;

  // configure camera tracking, temporal smothing and mapping
  v4r::TSFVisualSLAM tsf;
  tsf.setCameraParameter(intrinsic);

  v4r::TSFVisualSLAM::Parameter param;
  param.map_param.refine_plk = true;
  param.map_param.detect_loops = true;
  param.map_param.ba.optimize_delta_cloud_rgb_pose_global = false;
  param.map_param.ba.optimize_delta_cloud_rgb_pose = false;
  tsf.setParameter(param);

  v4r::FeatureDetector::Ptr detector(new v4r::FeatureDetector_KD_FAST_IMGD());
  tsf.setDetectors(detector, detector);

  std::vector<std::string> cloud_files;
  cloud_files = v4r::io::getFilesInDirectory(in_dir, pattern, false);

  std::sort(cloud_files.begin(), cloud_files.end());

  double conf_ransac_iter = 1;
  double conf_tracked_points = 1;


  // start camera tracking 
  for (int i=0; i<(int)cloud_files.size() || loop; i++)
  {
    cout<<"---------------- FRAME #"<<i<<" -----------------------"<<endl;
    cout<<in_dir+std::string("/")+cloud_files[i]<<endl;

//    pcl::PCLPointCloud2::Ptr cloud2;
//    cloud2.reset (new pcl::PCLPointCloud2);
//    if (pcd.read (filename, *cloud2, origin, orientation, version) < 0)
//      continue;
//    pcl::fromPCLPointCloud2 (*cloud2, *cloud);
//    cout<<"cloud: "<<cloud->width<<"x"<<cloud->height<<endl;

    if(pcl::io::loadPCDFile(in_dir+std::string(cloud_files[i]), *cloud)==-1)
      continue;

    convertImage(*cloud, image);
    image.copyTo(im_draw);

    tsf.setDebugImage(im_draw);

    // track
    { pcl::ScopeTime t("overall time");
      have_pose = tsf.track(*cloud, i, pose, conf_ransac_iter, conf_tracked_points);
      time = t.getTime();
    } //-- overall time --

    cout<<"conf (ransac, tracked points): "<<conf_ransac_iter<<", "<<conf_tracked_points<<endl;
    if (!have_pose) cout<<"Lost pose!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<endl;

    all_poses.push_back(std::make_pair(pose,-1));
    v4r::invPose(pose,inv_pose);

    // get filtered frame
    tsf.getFilteredCloudNormals(filt_cloud, filt_pose, timestamp);

    mean_time += time;
    cnt_time++;
    cout<<"mean="<<mean_time/double(cnt_time)<<"ms ("<<1000./(mean_time/double(cnt_time))<<"fps)"<<endl;
    cout<<"timestamp (c/f): "<<i<<"/"<<timestamp<<endl;

    // debug out draw
    int key=0;
    if (display)
    {
      drawConfidenceBar(im_draw, conf_ransac_iter, 50, 200, 30);
      drawConfidenceBar(im_draw, conf_tracked_points, 50, 200, 50);
      cv::imshow("image",im_draw);
      key = cv::waitKey(sleep);
      if (conf_ransac_iter<0.2) cv::waitKey(0);
    }
    else usleep(50000);

    if (((char)key)==27) break;
    if (((char)key)=='r') sleep=50;
    if (((char)key)=='s') sleep=0;
  }

  // optimize map
  tsf.stop();
  tsf.optimizeMap();

  //tsf.getMap();
  tsf.getCameraParameter(intrinsic_opti, dist_coeffs_opti);

  // create model in global coordinates
  cout<<"Create pointcloud model..."<<endl;
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr glob_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
  pcl::PolygonMesh mesh;
  v4r::TSFGlobalCloudFilteringSimple gfilt;
  v4r::TSFGlobalCloudFilteringSimple::Parameter filt_param;
  filt_param.filter_largest_cluster = true;
  filt_param.voxel_size = voxel_size;
  gfilt.setParameter(filt_param);

  gfilt.setCameraParameter(intrinsic_opti, dist_coeffs_opti);
  gfilt.getGlobalCloudFiltered(tsf.getMap(), *glob_cloud);
  cout<<"Creat mesh..."<<endl;
  gfilt.getMesh(glob_cloud, mesh);

  // store resulting files
  if (file_mesh.size()>0) pcl::io::savePLYFile(file_mesh, mesh);
  if (file_cloud.size()>0) pcl::io::savePCDFileBinary(file_cloud, *glob_cloud);

  cout<<"Finished!"<<endl;


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
    c = getopt(argc, argv, "i:p:a:o:m:c:d:h");
    if(c == -1)
      break;
    switch(c)
    {
    case 'i':
      in_dir = optarg;
      break;
    case 'p':
      pattern = optarg;
      break;
    case 'a':
      cam_file = optarg;
      break;
    case 'm':
      file_mesh = optarg;
      break;
    case 'c':
      file_cloud = optarg;
      break;
    case 'd':
      display = std::atoi(optarg);
      break;

    case 'h':
      printf("%s [-f filenames] [-s start_idx] [-e end_idx] [-a cam_file.yml] [-h]\n"
             "   -i input directory\n"
             "   -p input file pattern (\".*.pcd\")\n"
             "   -a camera calibration files (opencv format)\n"
             "   -m mesh file name\n"
             "   -c point cloud file name\n"
             "   -d display results [0/1]\n"
             "   -h help\n",  argv[0]);
      exit(1);

      break;
    }
  }
}

void convertImage(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, cv::Mat &image)
{
  image = cv::Mat_<cv::Vec3b>(cloud.height, cloud.width);

  for (unsigned v = 0; v < cloud.height; v++)
  {
    for (unsigned u = 0; u < cloud.width; u++)
    {
      cv::Vec3b &cv_pt = image.at<cv::Vec3b> (v, u);
      const pcl::PointXYZRGB &pt = cloud(u,v);

      cv_pt[2] = pt.r;
      cv_pt[1] = pt.g;
      cv_pt[0] = pt.b;
    }
  }
}



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

/**
 * drawConfidenceBar
 */
void drawConfidenceBar(cv::Mat &im, const double &conf, int x_start, int x_end, int y)
{
  int bar_start = x_start, bar_end = x_end;
  int diff = bar_end-bar_start;
  int draw_end = diff*conf;
  double col_scale = (diff>0?255./(double)diff:255.);
  cv::Point2f pt1(0,y);
  cv::Point2f pt2(0,y);
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

















