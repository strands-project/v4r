/*
 * test_tomita.cpp
 *
 *  Created on: Mar 6, 2013
 *      Author: aitor
 */

#include <pcl/common/common.h>
#include <pcl/visualization/pcl_visualizer.h>

void generatePointCloudFilledCube(pcl::PointCloud<pcl::PointXYZ>::Ptr cube,
                                      float step_p = 0.01f)
{
  float step = step_p / 2.f;
  int size = 1.f / step;
  float start = -0.5f;
  for(int x=0; x < size; x++)
  {
    for(int y=0; y < size; y++)
    {
      for(int z=0; z < size; z++)
      {
        pcl::PointXYZ p;
        p.x = start + x * step;
        p.y = start + y * step;
        p.z = start + z * step;
        cube->points.push_back(p);
      }
    }
  }
}

int
main (int argc, char ** argv)
{

  float gc_size_ = 0.005f;
  pcl::PointCloud<pcl::PointXYZ>::Ptr model(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cube(new pcl::PointCloud<pcl::PointXYZ>);

  generatePointCloudFilledCube(cube);

  pcl::PointXYZ p1, p2, p3;
  p1.getVector3fMap() = Eigen::Vector3f(0,0,0);
  p2.getVector3fMap() = Eigen::Vector3f(0,0.3,0);
  p3.getVector3fMap() = Eigen::Vector3f(0,0.2,0.2);

  model->points.push_back(p1);
  model->points.push_back(p2);
  model->points.push_back(p3);

  pcl::visualization::PCLVisualizer vis("gc constraint");
  int v1, v2;
  v1 = v2 = 0;
  vis.createViewPort(0,0,0.5,1.0, v1);
  vis.createViewPort(0.5,0,1.0,1.0, v2);

  vis.addText3D ("p1", p1, 0.02, 0.0, 0.0, 1.0, "text_p1");
  vis.addText3D ("p2", p2, 0.02, 0.0, 0.0, 1.0, "text_p2");
  vis.addText3D ("p3", p3, 0.02, 0.0, 0.0, 1.0, "text_p3");

  vis.addText("model", 10, 10, 16, 125, 125, 125, "text_model", v1);
  vis.addText("scene", 10, 10, 16, 125, 125, 125, "text_scene", v2);
  vis.setBackgroundColor(255,255,255);

  pcl::visualization::PointCloudColorHandlerCustom < pcl::PointXYZ > handler_rgb ( model, 255, 0, 0 );
  vis.addPointCloud(model, handler_rgb, "model");

  float dist_model_12 = (p1.getVector3fMap() - p2.getVector3fMap()).norm();
  float dist_model_13 = (p1.getVector3fMap() - p3.getVector3fMap()).norm();
  float dist_model_23 = (p2.getVector3fMap() - p3.getVector3fMap()).norm();

  pcl::PointCloud<pcl::PointXYZ>::Ptr gc1(new pcl::PointCloud<pcl::PointXYZ>);
  for(size_t i=0; i < cube->points.size(); i++)
  {
    if((cube->points[i].getVector3fMap() - p2.getVector3fMap()).norm() > gc_size_ )
    {
      continue;
    }

    float dist_target_1_i = (p1.getVector3fMap() - cube->points[i].getVector3fMap()).norm();
    float dist_target_2_i = (p2.getVector3fMap() - cube->points[i].getVector3fMap()).norm();
    //float dist_target_3_i = (p3.getVector3fMap() - cube->points[i].getVector3fMap()).norm();
    if( std::abs(dist_model_12 - dist_target_1_i) < gc_size_)
    {
      //find a third point that fulfills the gc constraint with p1 and cube_i
      for(size_t j=0; j < cube->points.size(); j++)
      {
        if(j == i)
          continue;

        float dist_target_1_j = (p1.getVector3fMap() - cube->points[j].getVector3fMap()).norm();
        float dist_target_2_j = (p2.getVector3fMap() - cube->points[j].getVector3fMap()).norm();
        float dist_target_i_j = (cube->points[i].getVector3fMap() - cube->points[j].getVector3fMap()).norm();

        if( std::abs(dist_model_13 - dist_target_1_j) < gc_size_
            &&
            std::abs(dist_model_23 - dist_target_i_j) < gc_size_)
        {
          gc1->points.push_back(cube->points[j]);
          //std::cout << gc1->points.size() << std::endl;
        }
      }

      std::cout << gc1->points.size() << std::endl;

      pcl::visualization::PointCloudColorHandlerCustom < pcl::PointXYZ > handler_rgb ( gc1, 125, 125, 125 );
      vis.addPointCloud(gc1, handler_rgb, "cube", v2);
      vis.spin();
      vis.removePointCloud("cube");
      gc1->points.clear();
    }
  }

}

