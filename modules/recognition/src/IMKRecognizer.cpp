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


#include <v4r/recognition/IMKRecognizer.h>
#include <v4r/recognition/IMKRecognizerIO.h>
#include <v4r/reconstruction/impl/projectPointToImage.hpp>
#include <v4r/io/filesystem.h>
#include <v4r/keypoints/impl/PoseIO.hpp>
#include <v4r/keypoints/impl/invPose.hpp>
//#include "v4r/KeypointTools/ScopeTime.hpp"
#include <v4r/common/impl/Vector.hpp>
#include <v4r/keypoints/impl/warpPatchHomography.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/common/time.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <algorithm>

//#define DEBUG_AR_GUI

#ifdef DEBUG_AR_GUI
#include "v4r/TomGine/tgTomGineThread.h"
#include "opencv2/highgui/highgui.hpp"
#include <numeric>
#endif


namespace v4r
{

using namespace std;


#ifdef DEBUG_AR_GUI
boost::shared_ptr<TomGine::tgTomGineThread> win(new TomGine::tgTomGineThread(1024,768, "TomGine"));
#endif


inline bool cmpObjectsDec(const v4r::triple<std::string, double, Eigen::Matrix4f> &a, const v4r::triple<std::string, double, Eigen::Matrix4f> &b)
{
  return (a.second>b.second);
}


/************************************************************************************
 * Constructor/Destructor
 */
IMKRecognizer::IMKRecognizer(const Parameter &p,
                                                       const v4r::FeatureDetector::Ptr &_detector,
                                                       const v4r::FeatureDetector::Ptr &_descEstimator)
 : param(p), detector(_detector), descEstimator(_descEstimator)
{
#ifdef DEBUG_AR_GUI
cv::Mat_<double> cam = (cv::Mat_<double>(3,3) << 525, 0, 320, 0, 525, 240, 0, 0, 1);
cv::Mat_<float> cR = cv::Mat_<float>::eye(3,3);
cv::Mat_<float> ct = cv::Mat_<float>::zeros(3,1);
win->SetCamera(cam);
win->SetCamera(cR,ct);
#endif

  if (detector.get()==0) detector = descEstimator;
  cbMatcher.reset(new CodebookMatcher(param.cb_param));
  votesClustering.setParameter(p.vc_param);
  pnp.setParameter(p.pnp_param);
}

IMKRecognizer::~IMKRecognizer()
{
}

/**
 * @brief IMKRecognizer::convertImage
 * @param cloud
 * @param image
 */
void IMKRecognizer::convertImage(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, cv::Mat &image)
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

/**
 * @brief IMKRecognizer::addView
 * @param idx
 * @param keys
 * @param descs
 * @param cloud
 * @param mask
 * @param pose
 */
void IMKRecognizer::addView(const unsigned &idx, const std::vector<cv::KeyPoint> &keys, const cv::Mat &descs, const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const cv::Mat_<unsigned char> &mask, const Eigen::Matrix4f &pose, Eigen::Vector3d &centroid, unsigned &cnt)
{
  object_models.push_back(IMKView(idx));
  IMKView &view = object_models.back();
  view.keys.reserve(keys.size());
  view.points.reserve(keys.size());
  cv::Mat tmp_descs;

  Eigen::Matrix3f R = pose.topLeftCorner<3,3>();
  Eigen::Vector3f t = pose.block<3,1>(0,3);

  for (unsigned i=0; i<keys.size(); i++)
  {
    const cv::KeyPoint &key = keys[i];
    if ((int)key.pt.x>=0 && (int)key.pt.x<(int)cloud.width && (int)key.pt.y>=0 && (int)key.pt.y<(int)cloud.height)
    {
      if (mask.rows == (int)cloud.height && mask.cols == (int)cloud.width && mask((int)key.pt.y, (int)key.pt.x)<128)
        continue;

      const pcl::PointXYZRGB &pt = cloud((int)key.pt.x, (int)key.pt.y);

      if (!isnan(pt.x) && !isnan(pt.y) && !isnan(pt.z))
      {
        view.keys.push_back(key);
        view.points.push_back(R*pt.getVector3fMap()+t);
        tmp_descs.push_back(descs.row(i));
        //view.descs.push_back(descs.row(i));
        centroid += view.points.back().cast<double>();
        cnt++;

#ifdef DEBUG_AR_GUI
//  cout<<view.points.back().transpose()<<endl;
//  win->AddPoint3D(view.points.back()[0],view.points.back()[1],view.points.back()[2], 0,0,255, 1);
#endif
      }
    }
  }

  cbMatcher->addView(tmp_descs,object_models.size()-1);
  //cbMatcher->addView(view.descs,object_models.size()-1);
}

/**
 * @brief IMKRecognizer::setViewDescriptor
 * @param image
 * @param cloud
 * @param mask
 * @param view
 */
void IMKRecognizer::setViewDescriptor(const cv::Mat_<unsigned char> &im_gray, const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const cv::Mat_<unsigned char> &mask, const Eigen::Matrix4f &pose, IMKView &view)
{
  //kp::ScopeTime tc("IMKRecognizer::setViewDescriptor");

  double min_val=DBL_MAX, max_val=-DBL_MAX;
  int min_x = INT_MAX;
  int min_y = INT_MAX;
  int max_x = 0;
  int max_y = 0;
  //get ROI
  for (int v=0; v<(int)mask.rows; v++)
  {
    for (int u=0; u<(int)mask.cols; u++)
    {
      if (mask(v,u)>128)
      {
        if (u<min_x) min_x=u;
        if (u>max_x) max_x=u;
        if (v<min_y) min_y=v;
        if (v>max_y) max_y=v;
        if (cloud(v,u).z>max_val) max_val=cloud(v,u).z;
        if (cloud(v,u).z<min_val) min_val=cloud(v,u).z;
      }
    }
  }
  if (min_x>0) min_x--;
  if (min_y>0) min_y--;
  if (max_x<mask.cols-1) max_x++;
  if (max_x<mask.rows-1) max_y++;
  // get depth map and image roi
  cv::Mat_<float> depth(max_y-min_y+1, max_x-min_x+1);
  view.cloud.resize(max_y-min_y+1, max_x-min_x+1);
  cv::Mat_<unsigned char> im_roi(max_y-min_y+1, max_x-min_x+1);
  cv::Mat_<unsigned char> mask_roi(max_y-min_y+1, max_x-min_x+1);
  Eigen::Matrix3f R = pose.topLeftCorner<3,3>();
  Eigen::Vector3f pt_tmp, t = pose.block<3,1>(0,3);
  for (int v=0; v<depth.rows; v++)
  {
    for (int u=0; u<depth.cols; u++)
    {
      const pcl::PointXYZRGB &pt = cloud(u+min_x, v+min_y);
      depth(v,u) = (isnan(pt.z)?10000:pt.z);
      view.cloud(v,u) = R*pt.getVector3fMap()+t;
      im_roi(v,u) = im_gray(v+min_y, u+min_x);
      mask_roi(v,u) = mask(v+min_y, u+min_x);
    }
  }
  // inpainting
  cv::Mat depth8U, inp8U;
  depth.convertTo(depth8U, CV_8UC1, 255.0/(max_val-min_val), -min_val * 255.0/(max_val-min_val));
  cv::inpaint(depth8U,(depth==10000),inp8U, 5.0, cv::INPAINT_TELEA);
  // scale and copy back
  float scale = (max_val-min_val)/255.;
  float offs = min_val * 255.0/(max_val-min_val);
  for (int v=0; v<depth.rows; v++)
  {
    for (int u=0; u<depth.cols; u++)
    {
      if (depth(v,u)==10000)
        depth(v,u) = ( ((float)inp8U.at<unsigned char>(v,u))+offs) * scale;
    }
  }
  //set cloud to view in global coordinates
  #ifdef DEBUG_AR_GUI
  //win->Clear();
  #endif
  double invC0 = 1./intrinsic(0,0);
  double invC4 = 1./intrinsic(1,1);
  for (int v=0; v<depth.rows; v++)
  {
    for (int u=0; u<depth.cols; u++)
    {
      Eigen::Vector3f &pt = view.cloud(v,u);
      if (isnan(pt[0]))
      {
        pt_tmp[2] = depth(v,u);
        pt_tmp[0] = pt_tmp[2]*((u+min_x-intrinsic(0,2))*invC0);
        pt_tmp[1] = pt_tmp[2]*((v+min_y-intrinsic(1,2))*invC4);
        pt = R*pt_tmp + t;
        #ifdef DEBUG_AR_GUI
        //win->AddPoint3D(pt[0],pt[1],pt[2], 255,100,100, 1);
        #endif
      }
      #ifdef DEBUG_AR_GUI
      //else win->AddPoint3D(pt[0],pt[1],pt[2], 255,255,255, 1);
      #endif
    }
  }
  // compute descriptor
  cv::Mat_<unsigned char> im_roi_scaled, mask_roi_scaled;
  cv::resize(im_roi,im_roi_scaled, cv::Size(param.image_size_conf_desc,param.image_size_conf_desc));
  cv::resize(mask_roi,mask_roi_scaled, cv::Size(param.image_size_conf_desc,param.image_size_conf_desc));
  mask_roi_scaled.convertTo(view.weight_mask, CV_32F, 1./255);
  view.weight_mask = (view.weight_mask>0.5);
  cp.compute(im_roi_scaled, view.weight_mask, view.conf_desc);

  #ifdef DEBUG_AR_GUI
  //win->Update();
  im_roi.copyTo(view.im_gray);  // debug
  cv::imshow("depth8U",depth8U);
  cv::imshow("inp8U",inp8U);
  cv::imshow("im_roi",im_roi);
  cv::imshow("im_roi_scaled",im_roi_scaled);
  //cv::waitKey(0);
  #endif
}


/**
 * @brief IMKRecognizer::createObjectModel
 * @param _object_name
 */
void IMKRecognizer::createObjectModel(const unsigned &idx)
{
  if (detector.get()==0 || descEstimator.get()==0)
    throw std::runtime_error("[IMKRecognizer::createObjectModel] Kepoint detector or descriptor estimator not set!");
  if (intrinsic.empty())
    throw std::runtime_error("[IMKRecognizer::createObjectModel] Intrinsic camera parameter not set!");


  cv::Mat_<cv::Vec3b> image;
  cv::Mat_<unsigned char> im_gray;
  cv::Mat_<unsigned char> mask;
  std::vector<std::string> cloud_files;
  std::string pose_file, mask_file;
  std::string so_far = "";
  std::string pattern =  std::string("cloud_")+std::string(".*.")+std::string("pcd");
  const std::string &name = object_names[idx];

  v4r::io::getFilesInDirectory(base_dir+std::string("/models/")+name+std::string("/views/"),cloud_files,so_far,pattern,false);

  pcl::PCDReader pcd;
  Eigen::Vector4f origin;
  Eigen::Quaternionf orientation;
  int version;
  Eigen::Matrix4f pose, inv_pose;
  pcl::PCLPointCloud2::Ptr cloud2(new pcl::PCLPointCloud2);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  std::vector<Eigen::Matrix4f> poses;
  Eigen::Vector3d centroid(0.,0.,0.);
  unsigned key_cnt = 0;
  unsigned start_frame = object_models.size();

#ifdef DEBUG_AR_GUI
  win->Clear();
#endif

  for (unsigned i=0; i<cloud_files.size(); i++)
  {
    mask_file = pose_file = cloud_files[i];
    boost::replace_last (mask_file, "pcd", "png");
    boost::replace_last (mask_file, "cloud_", "mask_");
    boost::replace_last (pose_file, "pcd", "txt");
    boost::replace_last (pose_file, "cloud_", "pose_");

    if (pcd.read (base_dir+std::string("/models/")+name+std::string("/views/")+cloud_files[i], *cloud2, origin, orientation, version) < 0)
      continue;

    mask = cv::imread(base_dir+std::string("/models/")+name+std::string("/views/")+mask_file, CV_LOAD_IMAGE_GRAYSCALE);

    if (mask.empty())
      continue;

    pcl::fromPCLPointCloud2 (*cloud2, *cloud);
    convertImage(*cloud, image);

    if (!readPose(base_dir+std::string("/models/")+name+std::string("/views/")+pose_file, pose))
      continue;

    if( image.type() != CV_8U ) cv::cvtColor( image, im_gray, CV_RGB2GRAY );
    else im_gray = image;

    detector->detect(im_gray,keys);
    descEstimator->extract(im_gray, keys, descs);

    int tmp_nb = key_cnt;

    addView(idx, keys, descs, *cloud, mask, pose, centroid, key_cnt);
    setViewDescriptor(im_gray, *cloud, mask, pose, object_models.back());

    cout<<"Load "<<(name+std::string("/")+cloud_files[i])<<": detected "<<key_cnt-tmp_nb<<" keys"<<endl;

    poses.push_back(pose);
  }

  // center image points
  if (key_cnt>0)
  {
    centroid /= (double)key_cnt;
  }

#ifdef DEBUG_AR_GUI
  win->AddPoint3D(centroid[0],centroid[1],centroid[2], 255,0,0, 5);
  win->Update();
  cv::waitKey(0);
#endif

  cv::Point2f center;

  for(unsigned i=0, z=start_frame; i<poses.size(); i++, z++)
  {
    invPose(poses[i],inv_pose);

    Eigen::Matrix3f R = inv_pose.topLeftCorner<3,3>();
    Eigen::Vector3f t = inv_pose.block<3,1>(0,3);

    Eigen::Vector3f pt = R*centroid.cast<float>() + t;

    if (dist_coeffs.empty())
      projectPointToImage(&pt[0], &intrinsic(0,0), &center.x);
    else projectPointToImage(&pt[0], &intrinsic(0,0), &dist_coeffs(0,0), &center.x);

    IMKView &view = object_models[z];

    for (unsigned j=0; j<view.keys.size(); j++)
    {
      view.keys[j].pt = center-view.keys[j].pt;
    }
  }
}

/**
 * @brief IMKRecognizer::getMaxViewIndex
 * @param views
 * @param matches
 * @param inliers
 */
int IMKRecognizer::getMaxViewIndex(const std::vector<IMKView> &views, const std::vector<cv::DMatch> &matches, const std::vector<int> &inliers)
{
  cnt_view_matches.assign(views.size(),0);
  for (unsigned i=0; i<inliers.size(); i++)
  {
    cnt_view_matches[matches[inliers[i]].imgIdx] ++;
  }
  return std::distance(cnt_view_matches.begin(), std::max_element(cnt_view_matches.begin(), cnt_view_matches.end()));
}

/**
 * @brief IMKRecognizer::getNearestNeighbours
 */
void IMKRecognizer::getNearestNeighbours(const Eigen::Vector2f &pt, const std::vector<cv::KeyPoint> &keys, const float &sqr_inl_radius_conf, std::vector<int> &nn_indices)
{
  nn_indices.clear();
  for (unsigned i=0; i<keys.size(); i++)
  {
    if ((pt-Eigen::Map<const Eigen::Vector2f>(&keys[i].pt.x)).squaredNorm() < sqr_inl_radius_conf)
      nn_indices.push_back(i);
  }
}

/**
 * @brief IMKRecognizer::getMinDescDistU8
 * @param desc
 * @param descs
 * @param indices
 * @return
 */
float IMKRecognizer::getMinDescDist32F(const cv::Mat &desc, const cv::Mat &descs, const std::vector<int> &indices)
{
  float dist, min = FLT_MAX;
  for (unsigned i=0; i<indices.size(); i++)
  {
    dist = distanceL1(&desc.at<float>(0,0), &descs.at<float>(indices[i],0),desc.cols);
    if (dist<min)
      min = dist;
  }
  return min;
}

/**
 * @brief IMKRecognizer::computeGradientHistogramConf
 * @param pose
 * @param view
 * @return
 */
double IMKRecognizer::computeGradientHistogramConf(const cv::Mat_<unsigned char> &im_gray, const IMKView &view, const Eigen::Matrix4f &pose)
{
  //kp::ScopeTime tc("IMKRecognizer::computeGradientHistogramConf");
  if (view.conf_desc.size()==0 || view.cloud.rows<5 || view.cloud.cols<5 || view.weight_mask.rows!=param.image_size_conf_desc || view.weight_mask.cols!=param.image_size_conf_desc)
    throw std::runtime_error("[IMKRecognizer::computeGradientHistogramConf] The object model does not fit to the current configuration! Please rebuild the model (delete the auto-generated model folder)!");

  // warp image to model view
  Eigen::Vector2f im_pt;
  Eigen::Matrix3f R = pose.topLeftCorner<3,3>();
  Eigen::Vector3f pt, t = pose.block<3,1>(0,3);
  im_warped = cv::Mat_<unsigned char>(view.cloud.rows, view.cloud.cols);
  for (int v=0; v<im_warped.rows; v++)
  {
    for (int u=0; u<im_warped.cols; u++)
    {
      pt = R * view.cloud(v,u) + t;
      if (dist_coeffs.empty())
        projectPointToImage(&pt[0],&intrinsic(0,0),&im_pt[0]);
      else projectPointToImage(&pt[0],&intrinsic(0,0), &dist_coeffs(0,0), &im_pt[0]);
      if (im_pt[0]>=0 && im_pt[1]>=0 && im_pt[0]<im_gray.cols-1 && im_pt[1]<im_gray.rows-1)
        im_warped(v,u) = getInterpolated(im_gray, im_pt);
      else im_warped(v,u) = 128;
    }
  }
  // compute descriptor
  cv::resize(im_warped,im_warped_scaled, cv::Size(param.image_size_conf_desc,param.image_size_conf_desc));
  cp.compute(im_warped_scaled, view.weight_mask, desc);
  #ifdef DEBUG_AR_GUI
  if (!view.im_gray.empty()) cv::imshow("view.im_gray",view.im_gray);
  cv::imshow("im_warped", im_warped);
  cv::waitKey(0);
  #endif
  if (view.conf_desc.size()>0 && view.conf_desc.size()==desc.size())
    return 1.-sqrt(squaredDistance(&view.conf_desc[0],&desc[0],desc.size()));
  return 0.;
}

/**
 * @brief IMKRecognizer::poseEstimation
 * @param object_names
 * @param views
 * @param keys
 * @param matches
 * @param clusters
 */
void IMKRecognizer::poseEstimation(const cv::Mat_<unsigned char> &im_gray, const std::vector<std::string> &object_names, const std::vector<IMKView> &views, const std::vector<cv::KeyPoint> &keys, const cv::Mat &descs, const std::vector< std::vector< cv::DMatch > > &matches, const std::vector< boost::shared_ptr<v4r::triple<unsigned, double, std::vector< cv::DMatch > > > > &clusters, std::vector<v4r::triple<std::string, double, Eigen::Matrix4f> > &objects)
{
  std::vector<cv::Point3f> points;
  std::vector<cv::Point2f> im_points;
  std::vector<cv::DMatch> tmp_matches;
  Eigen::Matrix4f pose;
  std::vector<int> inliers;

  for (int i=0; i<(int)clusters.size() && i<param.use_n_clusters; i++)
  {

    int nb_ransac_trials;
    const v4r::triple<unsigned, double, std::vector< cv::DMatch > > &ms = *clusters[i];
    im_points.clear();
    points.clear();
    tmp_matches.clear();

    if (ms.second<param.min_cluster_size)
      continue;

    for (unsigned j=0; j<ms.third.size(); j++)
    {
      const cv::DMatch &m = ms.third[j];
      if (m.distance<=std::numeric_limits<float>::epsilon())
        continue;
      im_points.push_back(keys[m.queryIdx].pt);
      const Eigen::Vector3f &pt = views[m.imgIdx].points[m.trainIdx];
      points.push_back(cv::Point3f(pt[0],pt[1],pt[2]));
      tmp_matches.push_back(m);
    }

    nb_ransac_trials = pnp.ransacSolvePnP(points, im_points, pose, inliers);

    if (nb_ransac_trials<(int)param.pnp_param.max_rand_trials)
    {
      int view_idx = getMaxViewIndex(views, tmp_matches, inliers);
      //double conf = getConfidenceKeypointMatch(views, keys, descs, pose, getMaxViewIndex(views, tmp_matches, inliers) );
//      double conf = (view_idx>=0 && view_idx<(int)views.size()? (views[view_idx].keys.size()>0? ((double)inliers.size())/(double)views[view_idx].keys.size() : 0.) : 0.);
      double conf = computeGradientHistogramConf(im_gray, views[view_idx], pose);
      objects.push_back(v4r::triple<std::string, double, Eigen::Matrix4f>(object_names[ms.first],(conf>1?1.:conf), pose));
    }

#ifdef DEBUG_AR_GUI
//    cout<<i<<": object_name="<<object_names[ms.first]<<", nb_ransac_trials="<<nb_ransac_trials<<"/"<<param.pnp_param.max_rand_trials<<(nb_ransac_trials==(int)param.pnp_param.max_rand_trials?" failed":" converged!!")<<endl;
    if (!dbg.empty())
    {
    if (inliers.size()<param.min_cluster_size || nb_ransac_trials==(int)param.pnp_param.max_rand_trials) continue;
    cv::Vec3b col(rand()%255,rand()%255,rand()%255);
    for (unsigned j=0; j<inliers.size(); j++)
      cv::circle(dbg,im_points[inliers[j]], 3, CV_RGB(col[0],col[1],col[2]),2);
    cv::imshow("debug",dbg);
//    cv::waitKey(0);
    }
#endif
  }
}




/******************************* PUBLIC ***************************************/

/**
 * detect
 */
void IMKRecognizer::recognize(const cv::Mat &image, std::vector<v4r::triple<std::string, double, Eigen::Matrix4f> > &objects)
{
  objects.clear();

  if (intrinsic.empty())
    throw std::runtime_error("[IMKRecognizer::detect] Intrinsic camera parameter not set!");

  if( image.type() != CV_8U ) cv::cvtColor( image, im_gray, CV_RGB2GRAY );
  else im_gray = image;

  // get matches
  { //kp::ScopeTime t("IMKRecognizer::recognize - keypoint detection");
  detector->detect(im_gray, keys);
  descEstimator->extract(im_gray, keys, descs);
  }

#ifdef DEBUG_AR_GUI
//  image.copyTo(dbg);
  votesClustering.dbg = dbg;
#endif

  { //kp::ScopeTime t("IMKRecognizer::recognize - matching");
  cbMatcher->queryMatches(descs, matches, false);
  }

  { //kp::ScopeTime t("IMKRecognizer::recognize - clustering");
  votesClustering.operate(object_names, object_models, keys, matches, clusters);
  }

  { //kp::ScopeTime t("IMKRecognizer::recognize - pnp pose estimation");
  poseEstimation(im_gray, object_names, object_models, keys, descs, matches, clusters, objects);
  }

  std::sort(objects.begin(), objects.end(), cmpObjectsDec);
}



/**
 * @brief IMKRecognizer::clear
 */
void IMKRecognizer::clear()
{
  object_names.clear();
  object_models.clear();
  cbMatcher->clear();
}

/**
 * @brief IMKRecognizer::setDataDirectory
 * @param _base_dir
 */
void IMKRecognizer::setDataDirectory(const std::string &_base_dir)
{
  base_dir = _base_dir;
}

/**
 * @brief IMKRecognizer::addObject
 * @param _object_name
 */
void IMKRecognizer::addObject(const std::string &_object_name)
{
  object_names.push_back(_object_name);
}

/**
 * @brief IMKRecognizer::initModels
 */
void IMKRecognizer::initModels()
{
  IMKRecognizerIO io;
  if ( IMKRecognizerIO::read(base_dir+std::string("/models/"), object_names, object_models, *cbMatcher) )
  {
  }
  else
  {
    for (unsigned i=0; i<object_names.size(); i++)
    {
      createObjectModel(i);
    }
    cbMatcher->createCodebook();

    IMKRecognizerIO::write(base_dir+std::string("/models/"), object_names, object_models, *cbMatcher);
  }
}



/**
 * setSourceCameraParameter
 */
void IMKRecognizer::setCameraParameter(const cv::Mat &_intrinsic, const cv::Mat &_dist_coeffs)
{
  dist_coeffs = cv::Mat_<double>();
  if (_intrinsic.type() != CV_64F)
    _intrinsic.convertTo(intrinsic, CV_64F);
  else intrinsic = _intrinsic;
  if (!_dist_coeffs.empty())
  {
    dist_coeffs = cv::Mat_<double>::zeros(1,8);
    for (int i=0; i<_dist_coeffs.cols*_dist_coeffs.rows; i++)
      dist_coeffs(0,i) = _dist_coeffs.at<double>(0,i);
  }

  pnp.setCameraParameter(intrinsic, dist_coeffs);
}



}












