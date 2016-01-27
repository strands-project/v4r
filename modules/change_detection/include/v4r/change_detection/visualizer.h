/*
 * Visualiser3D.h
 *
 *  Created on: 23.1.2015
 *      Author: ivelas
 */

#ifndef VISUALISER3D_H_
#define VISUALISER3D_H_

#include <iostream>
#include <opencv/cv.h>

#include <pcl/common/eigen.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>

#include <v4r/change_detection/viewport_checker.h>
#include <v4r/change_detection/object_detection.h>
#include <v4r/core/macros.h>

using namespace std;

namespace v4r {

class V4R_EXPORTS Visualizer3D
{
public:
  Visualizer3D(const std::string &win_name = "3D Viewer");

  ~Visualizer3D();

  static void printRT(const Eigen::Matrix4f &transformation) {
    float x, y, z, roll, pitch, yaw;
    Eigen::Affine3f affineTransf(transformation);
    pcl::getTranslationAndEulerAngles(affineTransf, x, y, z, roll, pitch, yaw);
    cerr << "t: [" << x << ", " << y << ", " << z << "]\t" <<
        "R: [" << roll << ", " << pitch << ", " << yaw << "]" << endl;
  }

  template<typename PointT>
  Visualizer3D& addPointCloud(const pcl::PointCloud<PointT> cloud,
                              const Eigen::Matrix4f &transformation = Eigen::Matrix4f::Identity()) {
    pcl::PointCloud<PointT> cloud_transformed;
    transformPointCloud(cloud, cloud_transformed, transformation);
    if(transformation.isIdentity()) {
      //cerr << "Transformation: Identity" << endl;
    } else {
      cerr << "Transformation:" << endl << transformation.matrix() << endl;
      printRT(transformation);
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr color_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    unsigned rgb[] = { rngU(), rngU(), rngU()} ;

    for (typename pcl::PointCloud<PointT>::const_iterator pt = cloud_transformed.begin();
        pt < cloud_transformed.end(); pt++)
    {
      pcl::PointXYZRGB color_pt(rgb[0], rgb[1], rgb[2]);
      color_pt.x = pt->x;
      color_pt.y = pt->y;
      color_pt.z = pt->z;
      color_cloud->push_back(color_pt);
    }
    return addColorPointCloud(color_cloud);
  }

  Visualizer3D& addColorPointCloud(
      const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
      const Eigen::Matrix4f &transformation = Eigen::Matrix4f::Identity()) {

    pcl::transformPointCloud(*cloud, *cloud, transformation);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_vis(cloud);
    std::string id = getId("cloud");
    viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb_vis, id);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                             2, id);
    return *this;
  }

  template<typename PointT>
  Visualizer3D& addPointClouds(const std::vector< pcl::PointCloud<PointT> > &clouds) {
    for(typename std::vector< pcl::PointCloud<PointT> >::const_iterator cloud = clouds.begin();
        cloud < clouds.end(); cloud++) {
      addPointCloud(*cloud);
    }
    return *this;
  }

  Visualizer3D& addPointClouds(std::vector< pcl::PointCloud<pcl::PointXYZRGB>::Ptr > &clouds) {
    for(typename std::vector< pcl::PointCloud<pcl::PointXYZRGB>::Ptr >::const_iterator cloud = clouds.begin();
        cloud < clouds.end(); cloud++) {
      addColorPointCloud(*cloud);
    }
    return *this;
  }

  template<typename PointT>
  Visualizer3D& addObjects(const std::vector< ObjectDetection<PointT> > &objects) {
    for(typename std::vector< ObjectDetection<PointT> >::const_iterator object = objects.begin();
    		object < objects.end(); object++) {
      addColorPointCloud(object->getCloud());
    }
    return *this;
  }

  Visualizer3D& addSenzor(pcl::PointXYZ position = pcl::PointXYZ(0, 0, 0));

  template<typename PointT>
  Visualizer3D& addLine(const PointT &p1, const PointT &p2,
		  float r = -1.0, float g = -1.0, float b = -1.0) {
	  r = (r >= 0) ? r : rngF();
	  g = (g >= 0) ? g : rngF();
	  b = (b >= 0) ? b : rngF();
	  viewer->addLine(p1, p2, r, g, b, getId("line"));
	  return *this;
  }

  template<typename PointT>
  Visualizer3D& addViewVolume(const ViewVolume<PointT> & volume) {
	pcl::PointCloud<pcl::PointXYZ> borders = volume.getBorders();
	assert(borders.size() == 8);
	float r = 0.9;
	float g = 0.7;
	float b = 0.0;
	for(int i = 0; i < 4; i++) {
		addLine(borders[i], borders[(i+1)%4], r, g, b);
		addLine(borders[i], borders[(i+2)%4], r, g, b);
		addLine(borders[i+4], borders[(i+1)%4 + 4], r, g, b);
		addLine(borders[i+4], borders[(i+2)%4 + 4], r, g, b);
		addLine(borders[i], borders[i+4], r, g, b);
	}
	return *this;
  }

  void show() {
    viewer->spin();
  }

  void showOnce(int time = 1) {
    viewer->spinOnce(time);
  }

  Visualizer3D& saveSnapshot(const std::string &filename);

  void close() {
    viewer->close();
  }

  Visualizer3D& keepOnlyClouds(int count);

  Visualizer3D& setColor(unsigned r, unsigned g, unsigned b) {
    color_stack.push_back(r);
    color_stack.push_back(g);
    color_stack.push_back(b);
    return *this;
  }

  const boost::shared_ptr<pcl::visualization::PCLVisualizer>& getViewer() const
  {
    return viewer;
  }

protected:
  std::string getId(const string &what) {
    std::stringstream ss;
    ss << what << "_" << identifier++;
    all_identifiers.push_back(ss.str());
    return ss.str();
  }

  double rngF() {
    return rng.uniform(0.0, 1.0);
  }

  unsigned rngU() {
    if(color_stack.size() > color_index) {
      return color_stack[color_index++];
    } else {
      return rng(256);
    }
  }

protected:
  cv::RNG& rng;
  unsigned color_index;
  vector<unsigned> color_stack;
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
  int identifier;
  vector<string> all_identifiers;
};

}

#endif /* VISUALISER3D_H_ */
