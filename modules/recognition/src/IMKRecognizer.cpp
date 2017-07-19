/******************************************************************************
 * Copyright (c) 2016 Johann Prankl
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 ******************************************************************************/



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
//#include "v4r/TomGine/tgTomGineThread.h"
#include "opencv2/highgui/highgui.hpp"
#include <numeric>
#endif


namespace v4r
{

using namespace std;



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
  setCodebookFilename("");
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
void IMKRecognizer::convertImage(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, cv::Mat &_image)
{
 _image = cv::Mat_<cv::Vec3b>(cloud.height, cloud.width);

  for (unsigned v = 0; v < cloud.height; v++)
  {
    for (unsigned u = 0; u < cloud.width; u++)
    {
      cv::Vec3b &cv_pt = _image.at<cv::Vec3b> (v, u);
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
void IMKRecognizer::addView(const unsigned &idx, const std::vector<cv::KeyPoint> &_keys, const cv::Mat &_descs, const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const cv::Mat_<unsigned char> &mask, const Eigen::Matrix4f &pose, Eigen::Vector3d &centroid, unsigned &cnt)
{
  object_models.push_back(IMKView(idx));
  IMKView &view = object_models.back();
  view.keys.reserve(_keys.size());
  view.points.reserve(_keys.size());
  cv::Mat tmp_descs;

  Eigen::Matrix3f R = pose.topLeftCorner<3,3>();
  Eigen::Vector3f t = pose.block<3,1>(0,3);

  for (unsigned i=0; i<_keys.size(); i++)
  {
    const cv::KeyPoint &key = _keys[i];
    if ((int)key.pt.x>=0 && (int)key.pt.x<(int)cloud.width && (int)key.pt.y>=0 && (int)key.pt.y<(int)cloud.height)
    {
      if (mask.rows == (int)cloud.height && mask.cols == (int)cloud.width && mask((int)key.pt.y, (int)key.pt.x)<128)
        continue;

      const pcl::PointXYZRGB &pt = cloud((int)key.pt.x, (int)key.pt.y);

      if (!isnan(pt.x) && !isnan(pt.y) && !isnan(pt.z))
      {
        view.keys.push_back(key);
        view.points.push_back(R*pt.getVector3fMap()+t);
        tmp_descs.push_back(_descs.row(i));
        //view.descs.push_back(descs.row(i));
        centroid += view.points.back().cast<double>();
        cnt++;
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
void IMKRecognizer::setViewDescriptor(const std::vector< cv::Mat_<unsigned char> > &_im_channels, const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const cv::Mat_<unsigned char> &mask, const Eigen::Matrix4f &pose, IMKView &view)
{
  //kp::ScopeTime tc("IMKRecognizer::setViewDescriptor");

  if (_im_channels.size()==0)
    return;

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
      }
    }
  }
  // compute descriptor
  cv::Mat_<unsigned char> im_roi_scaled, mask_roi_scaled;
  cv::resize(mask_roi,mask_roi_scaled, cv::Size(param.image_size_conf_desc,param.image_size_conf_desc));
  mask_roi_scaled.convertTo(view.weight_mask, CV_32F, 1./255);
  view.weight_mask = (view.weight_mask>0.5);
  view.conf_desc.clear();
  std::vector<float> _desc;
  for (unsigned i=0; i<_im_channels.size(); i++)
  {
    cv::resize(_im_channels[i]( cv::Rect(min_x,min_y,depth.cols,depth.rows) )  ,im_roi_scaled, cv::Size(param.image_size_conf_desc,param.image_size_conf_desc));
    cp.compute(im_roi_scaled, view.weight_mask, _desc);
    view.conf_desc.insert(view.conf_desc.begin(), _desc.begin(), _desc.end());
  }

  #ifdef DEBUG_AR_GUI
  _im_channels[0](cv::Rect(min_x,min_y,depth.cols,depth.rows)).copyTo(view.im_gray);  // debug
  cv::imshow("depth8U",depth8U);
  cv::imshow("inp8U",inp8U);
  cv::imshow("im_roi",_im_channels[0](cv::Rect(min_x,min_y,depth.cols,depth.rows)));
  cv::imshow("im_roi_scaled",im_roi_scaled);
  //cv::waitKey(0);
  #endif
}

/**
 * @brief IMKRecognizer::loadObjectIndices
 * @param _filename
 * @param _mask
 * @param _size
 * @return
 */
bool IMKRecognizer::loadObjectIndices(const std::string &_filename, cv::Mat_<unsigned char> &_mask, const cv::Size &_size)
{
  int idx;
  std::vector<int> indices;

  std::ifstream mi_f (  _filename );
  if (mi_f.is_open())
  {
    while ( mi_f >> idx )
      indices.push_back(idx);
    mi_f.close();

    _mask = cv::Mat_<unsigned char>::zeros(_size);
    int size = _mask.rows*_mask.cols;

    for (unsigned i=0; i<indices.size(); i++)
    {
      if (indices[i]<size)
        _mask(indices[i]/_mask.cols, indices[i]%_mask.cols) = 255;
    }
    return true;
  }

  return false;
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


  cv::Mat_<unsigned char> im_gray;
  cv::Mat_<unsigned char> mask;
  std::string pose_file, mask_file, object_indices_file;
  std::string pattern =  std::string("cloud_")+std::string(".*.")+std::string("pcd");
  const std::string &name = object_names[idx];
  std::vector<std::string> cloud_files = v4r::io::getFilesInDirectory(base_dir+std::string("/")+name+std::string("/views/"),pattern,false);

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

  for (unsigned i=0; i<cloud_files.size(); i++)
  {
    object_indices_file = mask_file = pose_file = cloud_files[i];
    boost::replace_last (mask_file, "pcd", "png");
    boost::replace_last (mask_file, "cloud_", "mask_");
    boost::replace_last (pose_file, "pcd", "txt");
    boost::replace_last (pose_file, "cloud_", "pose_");
    boost::replace_last (object_indices_file, "pcd", "txt");
    boost::replace_last (object_indices_file, "cloud_", "object_indices_");

    if (pcd.read (base_dir+std::string("/")+name+std::string("/views/")+cloud_files[i], *cloud2, origin, orientation, version) < 0)
      continue;

    mask = cv::imread(base_dir+std::string("/")+name+std::string("/views/")+mask_file, CV_LOAD_IMAGE_GRAYSCALE);

    if (mask.empty())
    {
      if (!loadObjectIndices(base_dir+std::string("/")+name+std::string("/views/")+object_indices_file, mask, cv::Size(cloud2->width,cloud2->height)))
        continue;
    }

    pcl::fromPCLPointCloud2 (*cloud2, *cloud);
    convertImage(*cloud, image);

    if (!readPose(base_dir+std::string("/")+name+std::string("/views/")+pose_file, pose))
      continue;

    cv::cvtColor(image, im_lab, CV_BGR2Lab);
    cv::split(im_lab, im_channels);

    detector->detect(im_channels[0],keys);
    descEstimator->extract(im_channels[0], keys, descs);

    int tmp_nb = key_cnt;

    addView(idx, keys, descs, *cloud, mask, pose, centroid, key_cnt);
    setViewDescriptor(im_channels, *cloud, mask, pose, object_models.back());

    cout<<"Load "<<(name+std::string("/")+cloud_files[i])<<": detected "<<key_cnt-tmp_nb<<" keys"<<endl;

    poses.push_back(pose);
  }

  // center image points
  if (key_cnt>0)
  {
    centroid /= (double)key_cnt;
  }

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
int IMKRecognizer::getMaxViewIndex(const std::vector<IMKView> &views, const std::vector<cv::DMatch> &_matches, const std::vector<int> &_inliers)
{
  cnt_view_matches.assign(views.size(),0);
  for (unsigned i=0; i<_inliers.size(); i++)
  {
    cnt_view_matches[_matches[_inliers[i]].imgIdx] ++;
  }
  return std::distance(cnt_view_matches.begin(), std::max_element(cnt_view_matches.begin(), cnt_view_matches.end()));
}

/**
 * @brief IMKRecognizer::getNearestNeighbours
 */
void IMKRecognizer::getNearestNeighbours(const Eigen::Vector2f &pt, const std::vector<cv::KeyPoint> &_keys, const float &sqr_inl_radius_conf, std::vector<int> &nn_indices)
{
  nn_indices.clear();
  for (unsigned i=0; i<_keys.size(); i++)
  {
    if ((pt-Eigen::Map<const Eigen::Vector2f>(&_keys[i].pt.x)).squaredNorm() < sqr_inl_radius_conf)
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
float IMKRecognizer::getMinDescDist32F(const cv::Mat &_desc, const cv::Mat &_descs, const std::vector<int> &indices)
{
  float dist, min = FLT_MAX;
  for (unsigned i=0; i<indices.size(); i++)
  {
    dist = distanceL1(&_desc.at<float>(0,0), &_descs.at<float>(indices[i],0),_desc.cols);
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
double IMKRecognizer::computeGradientHistogramConf(const std::vector< cv::Mat_<unsigned char> > &_im_channels, const IMKView &view, const Eigen::Matrix4f &pose)
{
  //kp::ScopeTime tc("IMKRecognizer::computeGradientHistogramConf");
  if (view.conf_desc.size()==0 || view.cloud.rows<5 || view.cloud.cols<5 || view.weight_mask.rows!=param.image_size_conf_desc || view.weight_mask.cols!=param.image_size_conf_desc)
    throw std::runtime_error("[IMKRecognizer::computeGradientHistogramConf] The object model does not fit to the current configuration! Please rebuild the model (delete the auto-generated model folder)!");
  if (_im_channels.size()==0)
    return 0.;

  // warp image to model view
  int max_x = _im_channels[0].cols-1;
  int max_y = _im_channels[0].rows-1;
  Eigen::Vector2f im_pt;
  Eigen::Matrix3f R = pose.topLeftCorner<3,3>();
  Eigen::Vector3f pt, t = pose.block<3,1>(0,3);
  ims_warped.resize(_im_channels.size());
  for (unsigned i=0; i<ims_warped.size(); i++)
    ims_warped[i] = cv::Mat_<unsigned char>(view.cloud.rows, view.cloud.cols);
  for (int v=0; v<view.cloud.rows; v++)
  {
    for (int u=0; u<view.cloud.cols; u++)
    {
      pt = R * view.cloud(v,u) + t;
      if (dist_coeffs.empty())
        projectPointToImage(&pt[0],&intrinsic(0,0),&im_pt[0]);
      else projectPointToImage(&pt[0],&intrinsic(0,0), &dist_coeffs(0,0), &im_pt[0]);
      for (unsigned i=0; i<ims_warped.size(); i++)
      {
        if (im_pt[0]>=0 && im_pt[1]>=0 && im_pt[0]<max_x && im_pt[1]<max_y)
          ims_warped[i](v,u) = getInterpolated(_im_channels[i], im_pt);
        else ims_warped[i](v,u) = 128;
      }
    }
  }
  // compute descriptor
  desc.clear();
  std::vector<float> tmp_desc;
  for (unsigned i=0; i<ims_warped.size(); i++)
  {
    cv::resize(ims_warped[i],im_warped_scaled, cv::Size(param.image_size_conf_desc,param.image_size_conf_desc));
    cp.compute(im_warped_scaled, view.weight_mask, tmp_desc);
    desc.insert(desc.begin(), tmp_desc.begin(), tmp_desc.end());
    #ifdef DEBUG_AR_GUI
    if (!view.im_gray.empty()) cv::imshow("view.im_gray",view.im_gray);
    cv::imshow("im_warped", ims_warped[i]);
//    cv::waitKey(0);
    #endif
  }
  int size = (desc.size()<view.conf_desc.size()?desc.size():view.conf_desc.size());
  double norm = size/128;                    // gradient hist size => 128
  if (size>0)
    return ( 1. - sqrt( squaredDistance(&view.conf_desc[0], &desc[0], size)/norm ) );
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
void IMKRecognizer::poseEstimation(const std::vector< cv::Mat_<unsigned char> > &_im_channels, const std::vector<std::string> &_object_names, const std::vector<IMKView> &views, const std::vector<cv::KeyPoint> &_keys, const cv::Mat &_descs, const std::vector< std::vector< cv::DMatch > > &_matches, const std::vector< boost::shared_ptr<v4r::triple<unsigned, double, std::vector< cv::DMatch > > > > &_clusters, std::vector<v4r::triple<std::string, double, Eigen::Matrix4f> > &objects, const pcl::PointCloud<pcl::PointXYZRGB> &_cloud)
{
    (void)_descs;
    (void)_matches;
  if (_im_channels.size()==0)
    return;

  std::vector<cv::Point3f> points;
  std::vector<cv::Point2f> _im_points;
  std::vector<cv::DMatch> tmp_matches;
  Eigen::Matrix4f pose;
  std::vector<int> _inliers;
  std::vector<float> depth;

  for (int i=0; i<(int)_clusters.size() && i<param.use_n_clusters; i++)
  {

    int nb_ransac_trials;
    const v4r::triple<unsigned, double, std::vector< cv::DMatch > > &ms = *_clusters[i];
    _im_points.clear();
    points.clear();
    tmp_matches.clear();

    if (ms.second<param.min_cluster_size)
      continue;

    for (unsigned j=0; j<ms.third.size(); j++)
    {
      const cv::DMatch &m = ms.third[j];
      if (m.distance<=std::numeric_limits<float>::epsilon())
        continue;
      _im_points.push_back(_keys[m.queryIdx].pt);
      const Eigen::Vector3f &pt = views[m.imgIdx].points[m.trainIdx];
      points.push_back(cv::Point3f(pt[0],pt[1],pt[2]));
      tmp_matches.push_back(m);
    }

    if (_cloud.width == (unsigned)_im_channels[0].cols && _cloud.height == (unsigned)_im_channels[0].rows)
    {
      depth.assign(_im_points.size(), std::numeric_limits<float>::quiet_NaN());
      for (unsigned j=0; j<depth.size(); j++)
      {
        const cv::Point2f &im_pt = _im_points[j];
        if (im_pt.x>=0 && im_pt.y>=0 && im_pt.x<_cloud.width && im_pt.y<_cloud.height)
          depth[j] = _cloud(im_pt.x, im_pt.y).z;
      }
    }

    nb_ransac_trials = pnp.ransac(points, _im_points, pose, _inliers, depth);

    if (nb_ransac_trials<(int)param.pnp_param.max_rand_trials)
    {
      int view_idx = getMaxViewIndex(views, tmp_matches, _inliers);
      //double conf = getConfidenceKeypointMatch(views, keys, descs, pose, getMaxViewIndex(views, tmp_matches, inliers) );
//      double conf = (view_idx>=0 && view_idx<(int)views.size()? (views[view_idx].keys.size()>0? ((double)inliers.size())/(double)views[view_idx].keys.size() : 0.) : 0.);
      double conf = computeGradientHistogramConf(_im_channels, views[view_idx], pose);
      objects.push_back(v4r::triple<std::string, double, Eigen::Matrix4f>(_object_names[ms.first],(conf>1?1.:conf<0?0:conf), pose));
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
void IMKRecognizer::recognize(const cv::Mat &_image, std::vector<v4r::triple<std::string, double, Eigen::Matrix4f> > &objects)
{
  objects.clear();

  if (intrinsic.empty())
    throw std::runtime_error("[IMKRecognizer::detect] Intrinsic camera parameter not set!");

  if( _image.type() != CV_8U )
  {
    cv::cvtColor(_image, im_lab, CV_BGR2Lab);
    cv::split(im_lab, im_channels);
  }
  else
  {
    im_channels.assign(1, _image);
  }

  // get matches
  { //kp::ScopeTime t("IMKRecognizer::recognize - keypoint detection");
  detector->detect(im_channels[0], keys);
  descEstimator->extract(im_channels[0], keys, descs);
  }

#ifdef DEBUG_AR_GUI
//  _image.copyTo(dbg);
  votesClustering.dbg = dbg;
#endif

  { //kp::ScopeTime t("IMKRecognizer::recognize - matching");
  cbMatcher->queryMatches(descs, matches, false);
  }

  { //kp::ScopeTime t("IMKRecognizer::recognize - clustering");
  votesClustering.operate(object_names, object_models, keys, matches, clusters);
  }

  { //kp::ScopeTime t("IMKRecognizer::recognize - pnp pose estimation");
  poseEstimation(im_channels, object_names, object_models, keys, descs, matches, clusters, objects, pcl::PointCloud<pcl::PointXYZRGB>() );
  }

  std::sort(objects.begin(), objects.end(), cmpObjectsDec);
}

/**
 * detect
 */
void IMKRecognizer::recognize(const pcl::PointCloud<pcl::PointXYZRGB> &_cloud, std::vector<v4r::triple<std::string, double, Eigen::Matrix4f> > &objects)
{
  objects.clear();

  if (intrinsic.empty())
    throw std::runtime_error("[IMKRecognizer::detect] Intrinsic camera parameter not set!");

  convertImage(_cloud, image);

  if( image.type() != CV_8U )
  {
    cv::cvtColor(image, im_lab, CV_BGR2Lab);
    cv::split(im_lab, im_channels);
  }
  else
  {
    im_channels.assign(1, image);
  }

  // get matches
  { //kp::ScopeTime t("IMKRecognizer::recognize - keypoint detection");
  detector->detect(im_channels[0], keys);
  descEstimator->extract(im_channels[0], keys, descs);
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
  poseEstimation(im_channels, object_names, object_models, keys, descs, matches, clusters, objects, _cloud);
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
 * @brief IMKRecognizer::setCodebookFilename
 * @param _base_dir
 */
void IMKRecognizer::setCodebookFilename(const std::string &_codebookFilename)
{
  codebookFilename = _codebookFilename;
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
  if ( IMKRecognizerIO::read(base_dir+std::string("/"), object_names, object_models, *cbMatcher, codebookFilename) )
  {
  }
  else
  {
    for (unsigned i=0; i<object_names.size(); i++)
    {
      createObjectModel(i);
    }
    cbMatcher->createCodebook();

    IMKRecognizerIO::write(base_dir+std::string("/"), object_names, object_models, *cbMatcher, codebookFilename);
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












