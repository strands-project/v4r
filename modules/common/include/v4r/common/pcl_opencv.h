/*
 * pcl_opencv.h
 *
 *  Created on: Oct 17, 2013
 *      Author: aitor
 */

#include <pcl/common/common.h>
#include <opencv2/opencv.hpp>

#ifndef PCL_OPENCV_H_
#define PCL_OPENCV_H_

namespace PCLOpenCV
{

  template<class PointT>
  void
  ConvertPCLCloud2Image (typename pcl::PointCloud<PointT>::Ptr &pcl_cloud, cv::Mat_<cv::Vec3b> &image)
  {
    unsigned pcWidth = pcl_cloud->width;
    unsigned pcHeight = pcl_cloud->height;
    unsigned position = 0;

    image = cv::Mat_<cv::Vec3b> (pcHeight, pcWidth);

    for (unsigned row = 0; row < pcHeight; row++)
    {
      for (unsigned col = 0; col < pcWidth; col++)
      {
        cv::Vec3b & cvp = image.at<cv::Vec3b> (row, col);
        position = row * pcWidth + col;
        const PointT &pt = pcl_cloud->points[position];

        cvp[0] = pt.b;
        cvp[1] = pt.g;
        cvp[2] = pt.r;
      }
    }
  }

  template<class PointT>
  void
  ConvertPCLCloud2Image (const typename pcl::PointCloud<PointT>::Ptr &pcl_cloud, cv::Mat_<cv::Vec3b> &image, bool crop = false)
  {
    unsigned pcWidth = pcl_cloud->width;
    unsigned pcHeight = pcl_cloud->height;
    unsigned position = 0;

    unsigned min_u = pcWidth-1, min_v = pcHeight-1, max_u = 0, max_v = 0;

    image = cv::Mat_<cv::Vec3b> (pcHeight, pcWidth);

    for (unsigned row = 0; row < pcHeight; row++)
    {
      for (unsigned col = 0; col < pcWidth; col++)
      {
        cv::Vec3b & cvp = image.at<cv::Vec3b> (row, col);
        position = row * pcWidth + col;
        const PointT &pt = pcl_cloud->points[position];

        cvp[0] = pt.b;
        cvp[1] = pt.g;
        cvp[2] = pt.r;

        if( pcl::isFinite(pt) )
        {
            if( row < min_v )
                min_v = row;
            if( row > max_v )
                max_v = row;
            if( col < min_u )
                min_u = col;
            if( col > max_u )
                max_u = col;
        }
      }
    }
    cv::Mat cropped_image = image(cv::Rect(min_u, min_v, max_u - min_u, max_v - min_v));

    if(crop)
        image = cropped_image.clone();
  }

  template<class PointT>
  void
  ConvertPCLCloud2DepthImage (typename pcl::PointCloud<PointT>::Ptr &pcl_cloud, cv::Mat_<float> &image)
  {
    unsigned pcWidth = pcl_cloud->width;
    unsigned pcHeight = pcl_cloud->height;
    unsigned position = 0;

    image = cv::Mat_<float> (pcHeight, pcWidth);

    for (unsigned row = 0; row < pcHeight; row++)
    {
      for (unsigned col = 0; col < pcWidth; col++)
      {
        //cv::Vec3b & cvp = image.at<cv::Vec3b> (row, col);
        position = row * pcWidth + col;
        const PointT &pt = pcl_cloud->points[position];
        image.at<float>(row,col) = 1.f / pt.z;
      }
    }
  }

  template<class PointT>
  void
  ConvertUnorganizedPCLCloud2Image (const typename pcl::PointCloud<PointT>::Ptr &pcl_cloud,
                                    cv::Mat_<cv::Vec3b> &image,
                                    bool crop = false,
                                    float bg_r = 255.0f,
                                    float bg_g = 255.0f,
                                    float bg_b = 255.0f,
                                    int width = 640,
                                    int height = 480,
                                    float f = 525.5f,
                                    float cx = 319.5f,
                                    float cy = 239.5f)
  {

      //transform scene_cloud to organized point cloud and then to image
      typename pcl::PointCloud<PointT>::Ptr pScenePCl_organized(new pcl::PointCloud<PointT>);
      pScenePCl_organized->width = width;
      pScenePCl_organized->height = height;
      pScenePCl_organized->is_dense = true;
      pScenePCl_organized->points.resize(width*height);

      for(size_t kk=0; kk < pScenePCl_organized->points.size(); kk++)
      {
          pScenePCl_organized->points[kk].x = pScenePCl_organized->points[kk].y = pScenePCl_organized->points[kk].z =
                  std::numeric_limits<float>::quiet_NaN();

          pScenePCl_organized->points[kk].r = bg_r;
          pScenePCl_organized->points[kk].g = bg_g;
          pScenePCl_organized->points[kk].b = bg_b;
      }

      int ws2 = 1;
      for (size_t kk = 0; kk < pcl_cloud->points.size (); kk++)
      {
          float x = pcl_cloud->points[kk].x;
          float y = pcl_cloud->points[kk].y;
          float z = pcl_cloud->points[kk].z;
          int u = static_cast<int> (f * x / z + cx);
          int v = static_cast<int> (f * y / z + cy);

          for(int uu = (u-ws2); uu < (u+ws2); uu++)
          {
              for(int vv = (v-ws2); vv < (v+ws2); vv++)
              {
                  //Not out of bounds
                  if ((uu >= width) || (vv >= height) || (uu < 0) || (vv < 0))
                      continue;

                  float z_oc = pScenePCl_organized->at (uu, vv).z;

                  if(pcl_isnan(z_oc))
                  {
                      pScenePCl_organized->at (uu, vv) = pcl_cloud->points[kk];
                  }
                  else
                  {
                      if(z < z_oc)
                      {
                          pScenePCl_organized->at (uu, vv) = pcl_cloud->points[kk];
                      }
                  }
              }
          }
      }
      ConvertPCLCloud2Image<PointT> (pScenePCl_organized, image, crop);
  }
}

#endif /* PCL_OPENCV_H_ */
