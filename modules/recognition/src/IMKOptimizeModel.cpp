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



#include <v4r/recognition/IMKOptimizeModel.h>
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
IMKOptimizeModel::IMKOptimizeModel(const Parameter &p,
                                                       const v4r::FeatureDetector::Ptr &_detector,
                                                       const v4r::FeatureDetector::Ptr &_descEstimator)
 : param(p), detector(_detector), descEstimator(_descEstimator)
{
  if (detector.get()==0) detector = descEstimator;
  pnp.setParameter(p.pnp_param);
  matcher = new cv::FlannBasedMatcher(new cv::flann::KDTreeIndexParams(16), new cv::flann::SearchParams(150,0,true));
}

IMKOptimizeModel::~IMKOptimizeModel()
{
}

/**
 * @brief IMKOptimizeModel::convertImage
 * @param cloud
 * @param image
 */
void IMKOptimizeModel::convertImage(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, cv::Mat &_image)
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
 * @brief IMKOptimizeModel::addView
 * @param idx
 * @param keys
 * @param descs
 * @param cloud
 * @param mask
 * @param pose
 */
void IMKOptimizeModel::addPoints3d(const std::vector<cv::KeyPoint> &_keys, const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const cv::Mat_<unsigned char> &mask, View &view)
{
  view.keys.reserve(_keys.size());
  view.points3d.reserve(_keys.size());

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
        view.points3d.push_back(pt.getVector3fMap());
      }
    }
  }
}


/**
 * @brief IMKOptimizeModel::loadObjectIndices
 * @param _filename
 * @param _mask
 * @param _size
 * @return
 */
bool IMKOptimizeModel::loadObjectIndices(const std::string &_filename, cv::Mat_<unsigned char> &_mask, const cv::Size &_size)
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
 * @brief IMKOptimizeModel::createObjectModel
 * @param _object_name
 */
void IMKOptimizeModel::loadObject(const unsigned &idx)
{
  if (detector.get()==0 || descEstimator.get()==0)
    throw std::runtime_error("[IMKOptimizeModel::createObjectModel] Kepoint detector or descriptor estimator not set!");
  if (intrinsic.empty())
    throw std::runtime_error("[IMKOptimizeModel::createObjectModel] Intrinsic camera parameter not set!");


  cv::Mat_<unsigned char> im_gray;
  cv::Mat_<unsigned char> mask;
  std::string pose_file, mask_file, object_indices_file;
  std::string pattern =  std::string("cloud_")+std::string(".*.")+std::string("pcd");
  const std::string &name = object_names[idx];
  std::vector<std::string> cloud_files = v4r::io::getFilesInDirectory(base_dir+std::string("/")+name+std::string("/views/"),pattern,false);
  std::sort(cloud_files.begin(),cloud_files.end());

  pcl::PCDReader pcd;
  Eigen::Vector4f origin;
  Eigen::Quaternionf orientation;
  int version;
  Eigen::Matrix4f pose;
  pcl::PCLPointCloud2::Ptr cloud2(new pcl::PCLPointCloud2);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  std::vector<Eigen::Matrix4f> poses;

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

    object_views[idx].push_back(boost::shared_ptr<View>(new View()));
    View &view = *object_views[idx].back();
    view.idx_part = idx;
    view.idx_view = i;
    view.name = cloud_files[i];
    view.pose = pose;
    detector->detect(im_channels[0],keys);
    addPoints3d(keys, *cloud, mask, view);
    descEstimator->extract(im_channels[0], view.keys, view.descs);

    cout<<"Load "<<(name+std::string("/")+cloud_files[i])<<": detected "<<view.keys.size()<<" keys"<<endl;

    poses.push_back(pose);
  }
}



/**
 * @brief IMKOptimizeModel::estimatePose
 * @param view0
 * @param view1
 * @param matches
 * @param inliers
 *   std::vector<cv::Point3f> cv_points_3d;
  std::vector< cv::Point2f > im_points;
  std::vector<float> depth_values;
  std::vector< int > _inliers;

 */
bool IMKOptimizeModel::estimatePose(const View &view0, const View &view1, const std::vector<std::vector< cv::DMatch > > &matches, Eigen::Matrix4f &pose, std::vector<int> &_inliers)
{
  cv_points_3d.clear();
  im_points.clear();
  depth_values.clear();
  for (unsigned i=0; i<matches.size(); i++)
  {
    if (matches[i].size()<2)
      return false;
    const cv::DMatch &m0 = matches[i][0];
    const cv::DMatch &m1 = matches[i][1];
    if (m0.distance/m1.distance < param.nnr)
    {
      cv_points_3d.push_back(view0.points3d[m0.trainIdx]);
      im_points.push_back(view1.keys[m0.queryIdx].pt);
      depth_values.push_back(view1.points3d[m0.queryIdx][2]);
    }
  }
  if (pnp.ransac(cv_points_3d, im_points, pose, _inliers, depth_values)>=(int)param.pnp_param.max_rand_trials)
    return false;
  return true;
}

/**
 * @brief IMKOptimizeModel::validatePoseAndAddLinks
 */
bool IMKOptimizeModel::validatePoseAndAddLinks(View &view0, View &view1, const std::vector<std::vector< cv::DMatch > > &_matches01, const std::vector<std::vector< cv::DMatch > > &_matches10, const Eigen::Matrix4f &pose01, const Eigen::Matrix4f &pose10, const std::vector<int> &_inliers01, const std::vector<int> &_inliers10)
{
  Eigen::Matrix4f inv_pose10;
  invPose(pose10, inv_pose10);

  if ((pose01.block<3,1>(0,3)-inv_pose10.block<3,1>(0,3)).norm() > param.eq_pose_dist || view0.keys.size()==0 || view1.keys.size()==0)
    return false;

  view0.links.push_back(Link(view1.idx_part, view1.idx_view, ((double)_matches01.size())/(double)view0.keys.size(), pose01 ));
  view1.links.push_back(Link(view0.idx_part, view0.idx_view, ((double)_matches10.size())/(double)view1.keys.size(), pose10 ));

  Link &l01 = view0.links.back();
  for (unsigned i=0; i<_inliers01.size(); i++)
    l01.matches.push_back(_matches01[_inliers01[i]][0]);

  Link &l10 = view1.links.back();
  for (unsigned i=0; i<_inliers10.size(); i++)
    l10.matches.push_back(_matches10[_inliers10[i]][0]);

  return true;
}

/**
 * @brief IMKOptimizeModel::matchViews
 */
void IMKOptimizeModel::matchViews()
{
  Eigen::Matrix4f pose01, pose10;
  std::vector<int> inliers01, inliers10;
  for (unsigned i=0; i<object_views.size(); i++)
  {
    for (unsigned j=0; j<object_views[i].size(); j++)
    {
      View &view0 = *object_views[i][j];
      if (view0.keys.size()<5)
        continue;
      // -- dbg
      std::string file = view0.name;
      boost::replace_last (file, "pcd", "jpg");
      boost::replace_last (file, "cloud_", "image_");
      cv::Mat im0 = cv::imread(base_dir+std::string("/")+object_names[view0.idx_part]+std::string("/views/")+file, CV_LOAD_IMAGE_COLOR);
      cv::imshow("im0", im0);
      // --
      for (unsigned k=0; k<object_views.size(); k++)
      {
        for (unsigned l=0; l<object_views[k].size(); l++)
        {
          bool have=false;
          View &view1 = *object_views[k][l];
          if (i!=k || j!=l)
          {
            if (view1.keys.size()<5)
              continue;
            matcher->knnMatch(view1.descs, view0.descs, matches01,2);
            matcher->knnMatch(view0.descs, view1.descs, matches10,2);
            bool converged01 = estimatePose(view0, view1, matches01, pose01, inliers01);
            bool converged10 = estimatePose(view1, view0, matches10, pose10, inliers10);
            //cout<<" matches: "<<matches01.size()<<", "<<matches10.size()<<" -> "<<inliers01.size()<<", "<<inliers10.size()<<endl;
            if (converged01 && converged10)
            {
              if(validatePoseAndAddLinks(view0, view1, matches01, matches10, pose01, pose10, inliers01, inliers10))
                have = true;
            }
          }
          // -- dbg --
          cout<<((int)have)<<" "<<flush;
          if (have)
          {
            std::string file = view1.name;
            boost::replace_last (file, "pcd", "jpg");
            boost::replace_last (file, "cloud_", "image_");
            cv::Mat im1 = cv::imread(base_dir+std::string("/")+object_names[view1.idx_part]+std::string("/views/")+file, CV_LOAD_IMAGE_COLOR);
            cv::imshow("im1", im1);
            cv::waitKey(0);
          }
          // -- dbg --
        }
      }
      cout<<endl;
    }
  }
}



/******************************* PUBLIC ***************************************/



/**
 * @brief IMKOptimizeModel::clear
 */
void IMKOptimizeModel::clear()
{
  object_names.clear();
  object_views.clear();
}

/**
 * @brief IMKOptimizeModel::setDataDirectory
 * @param _base_dir
 */
void IMKOptimizeModel::setDataDirectory(const std::string &_base_dir)
{
  base_dir = _base_dir;
}

/**
 * @brief IMKOptimizeModel::addObject
 * @param _object_name
 */
void IMKOptimizeModel::addObject(const std::string &_object_name)
{
  object_names.push_back(_object_name);
  object_views.push_back( std::vector< boost::shared_ptr<View> >() );
}

/**
 * @brief IMKOptimizeModel::loadAllObjectViews
 */
void IMKOptimizeModel::loadAllObjectViews()
{
  for (unsigned i=0; i<object_names.size(); i++)
  {
    loadObject(i);
  }
}

/**
 * @brief IMKOptimizeModel::optimize
 */
void IMKOptimizeModel::optimize()
{
  matchViews();
}


/**
 * setSourceCameraParameter
 */
void IMKOptimizeModel::setCameraParameter(const cv::Mat &_intrinsic, const cv::Mat &_dist_coeffs)
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












