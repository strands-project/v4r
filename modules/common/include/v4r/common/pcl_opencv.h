/******************************************************************************
 * Copyright (c) 2013 Aitor Aldoma, Thomas Faeulhammer
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


#include <pcl/common/common.h>
#include <opencv2/opencv.hpp>
#include <v4r/core/macros.h>
#include <omp.h>

#ifndef V4R_PCL_OPENCV_H_
#define V4R_PCL_OPENCV_H_

namespace v4r
{
  template<typename PointT>
  V4R_EXPORTS
  inline cv::Mat
  ConvertPCLCloud2Image (const typename pcl::PointCloud<PointT> &pcl_cloud, bool crop = false)
  {
    unsigned pcWidth = pcl_cloud.width;
    unsigned pcHeight = pcl_cloud.height;
    unsigned position = 0;

    unsigned min_u = pcWidth-1, min_v = pcHeight-1, max_u = 0, max_v = 0;

    cv::Mat_<cv::Vec3b> image (pcHeight, pcWidth);

    for (unsigned row = 0; row < pcHeight; row++)
    {
      for (unsigned col = 0; col < pcWidth; col++)
      {
        cv::Vec3b & cvp = image.at<cv::Vec3b> (row, col);
        position = row * pcWidth + col;
        const PointT &pt = pcl_cloud.points[position];

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

    if(crop) {
        cv::Mat cropped_image = image(cv::Rect(min_u, min_v, max_u - min_u, max_v - min_v));
        image = cropped_image.clone();
    }

    return image;
  }

  /**
   *@brief converts a point cloud to an image and crops it to a fixed size
   * @param[in] cloud
   * @param[in] cluster_idx object indices
   * @param[in] desired output height of the image
   * @param[in] desired output width of the image
   * @param[in] desired margin for the bounding box
   */
  template<typename PointT>
  V4R_EXPORTS
  inline cv::Mat
  ConvertPCLCloud2FixedSizeImage(const typename pcl::PointCloud<PointT> &cloud, const std::vector<int> &cluster_idx,
                                 size_t out_height = 256, size_t out_width = 256, size_t margin = 10,
                                 cv::Scalar bg_color = cv::Scalar(255,255,255), bool do_closing_operation = false)
  {
      int min_u, min_v, max_u, max_v;
      max_u = max_v = 0;
      min_u = cloud.width;
      min_v = cloud.height;

      std::vector<int> c_tmp = cluster_idx;

      for(size_t idx=0; idx<c_tmp.size(); idx++)
      {
          int u = c_tmp[idx] % cloud.width;
          int v = (int) (c_tmp[idx] / cloud.width);

          if (u>max_u)
              max_u = u;

          if (v>max_v)
              max_v = v;

          if (u<min_u)
              min_u = u;

          if (v<min_v)
              min_v = v;
      }

      min_u = std::max (0, min_u);
      min_v = std::max (0, min_v);
      max_u = std::min ((int)cloud.width-1, max_u);
      max_v = std::min ((int)cloud.height-1, max_v);

      if(do_closing_operation)
      {
          cv::Mat mask = cv::Mat(cloud.height, cloud.width, CV_8UC1);
          mask.setTo((unsigned char)0);
          for(size_t c_idx=0; c_idx<c_tmp.size(); c_idx++)
          {
             int idx = c_tmp[c_idx];
             int u = idx % cloud.width;
             int v = (int) (idx / cloud.width);
             mask.at<unsigned char> (v, u) = 255;
          }
          cv::Mat const structure_elem = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(20, 20));
          cv::Mat close_result;
          cv::morphologyEx(mask, close_result, cv::MORPH_CLOSE, structure_elem);

          c_tmp.resize(cloud.height* cloud.width);
          size_t kept=0;
          for(size_t v=0;v<cloud.height;v++)
          {
              for(size_t u=0; u<cloud.width; u++)
              {
                  int idx = v * cloud.width + u;
                  if (close_result.at<unsigned char>(v,u) > 128 )
                  {
                      c_tmp[kept] = idx;
                      kept++;
                  }
              }
          }
          c_tmp.resize(kept);
      }

      cv::Mat_<cv::Vec3b> image(cloud.height, cloud.width);
      image.setTo(bg_color);
      for(size_t c_idx=0; c_idx<c_tmp.size(); c_idx++)
      {
          int idx = c_tmp[c_idx];
          int u = idx % cloud.width;
          int v = (int) (idx / cloud.width);
          cv::Vec3b & cvp = image.at<cv::Vec3b> (v, u);
          const PointT &pt = cloud.points[idx];
          cvp[0] = pt.b;
          cvp[1] = pt.g;
          cvp[2] = pt.r;
      }

      int side_length_u = max_u - min_u;
      int side_length_v = max_v - min_v;
      int side_length = std::max<int>(side_length_u , side_length_v);

      // center object in the middle
      min_u = std::max<int>(0, int(min_u - (side_length - side_length_u)/2.f));
      min_v = std::max<int>(0, int(min_v - (side_length - side_length_v)/2.f));

      int side_length_uu = std::min<int>(side_length, image.cols  - min_u - 1);
      int side_length_vv = std::min<int>(side_length, image.rows - min_v - 1);

      cv::Mat image_roi = image( cv::Rect(min_u, min_v, side_length_uu, side_length_vv) );

      cv::Mat_<cv::Vec3b> img_tmp (side_length + 2*margin, side_length + 2*margin);
      img_tmp.setTo(bg_color);
      cv::Mat img_tmp_roi = img_tmp( cv::Rect(margin, margin, side_length_uu, side_length_vv) );
      image_roi.copyTo(img_tmp_roi);

      cv::Mat_<cv::Vec3b> dst(out_height, out_width);
      cv::resize(img_tmp, dst, dst.size(), 0, 0, cv::INTER_CUBIC);

      return dst;
  }



  /**
   *@brief converts a point cloud to an image and crops it to a fixed size
   * @param[in] cloud
   * @param[in] cluster_idx object indices
   * @param[in] desired output height of the image
   * @param[in] desired output width of the image
   */
  template<typename PointT>
  V4R_EXPORTS
  inline cv::Mat
  ConvertPCLCloud2Image(const typename pcl::PointCloud<PointT> &cloud, const std::vector<int> &cluster_idx, size_t out_height, size_t out_width)
    {
        volatile int min_u, min_v, max_u, max_v;
        max_u = max_v = 0;
        min_u = cloud.width;
        min_v = cloud.height;

        for(size_t idx=0; idx<cluster_idx.size(); idx++) {
            int u = cluster_idx[idx] % cloud.width;
            int v = (int) (cluster_idx[idx] / cloud.width);

            if (u>max_u)
                max_u = u;

            if (v>max_v)
                max_v = v;

            if (u<min_u)
                min_u = u;

            if (v<min_v)
                min_v = v;
        }

        if ( (int)out_width > (max_u-min_u) ) {
            int margin_x = out_width - (max_u - min_u);

            volatile int margin_x_left = margin_x / 2.f;
            min_u = std::max (0, min_u - margin_x_left);
            volatile int margin_x_right = out_width - (max_u - min_u);
            max_u = std::min ((int)cloud.width, max_u + margin_x_right);
            margin_x_left = out_width - (max_u - min_u);
            min_u = std::max (0, min_u - margin_x_left);
        }

        if ( (int)out_height > (max_v - min_v) ) {
            int margin_y = out_height - (max_v - min_v);
            volatile int margin_y_left = margin_y / 2.f;
            min_v = std::max (0, min_v - margin_y_left);
            volatile int margin_y_right = out_height - (max_v - min_v);
            max_v = std::min ((int)cloud.height, max_v + margin_y_right);
            margin_y_left = out_height - (max_v - min_v);
            min_v = std::max (0, min_v - margin_y_left);
        }

        cv::Mat_<cv::Vec3b> image(max_v - min_v, max_u - min_u);

        for (int row = 0; row < image.rows; row++) {
          for (int col = 0; col < image.cols; col++) {
            cv::Vec3b & cvp = image.at<cv::Vec3b> (row, col);
            int position = (row + min_v) * cloud.width + (col + min_u);
            const PointT &pt = cloud.points[position];

            cvp[0] = pt.b;
            cvp[1] = pt.g;
            cvp[2] = pt.r;
          }
        }

        cv::Mat_<cv::Vec3b> dst(out_height, out_width);
        cv::resize(image, dst, dst.size(), 0, 0, cv::INTER_CUBIC);

        return dst;
    }




  /**
   * @brief computes a binary image from a point cloud which pixels are true for pixel occupied when raytracing the point cloud
   *
   */
  template<class PointT>
  V4R_EXPORTS
  inline std::vector<bool>
  ConvertPCLCloud2OccupancyImage (const typename pcl::PointCloud<PointT> &cloud,
                                  int width = 640,
                                  int height = 480,
                                  float f = 525.5f,
                                  float cx = 319.5f,
                                  float cy = 239.5f)
  {

    std::vector<bool> mask(height*width, false);

    #pragma omp parallel for schedule (dynamic)
    for (size_t i=0; i<cloud.points.size(); i++)
    {
        const PointT &pt = cloud.points[i];
        int u = static_cast<int> (f * pt.x / pt.z + cx);
        int v = static_cast<int> (f * pt.y / pt.z + cy);

        int idx = v*width + u;
        mask[idx] = true;
    }
    return mask;
  }


  template<class PointT>
  V4R_EXPORTS
  inline cv::Mat
  ConvertPCLCloud2DepthImage (const typename pcl::PointCloud<PointT> &pcl_cloud)
  {
    unsigned pcWidth = pcl_cloud.width;
    unsigned pcHeight = pcl_cloud.height;
    unsigned position = 0;

    cv::Mat_<float> image(pcHeight, pcWidth);

    for (unsigned row = 0; row < pcHeight; row++)
    {
      for (unsigned col = 0; col < pcWidth; col++)
      {
        //cv::Vec3b & cvp = image.at<cv::Vec3b> (row, col);
        position = row * pcWidth + col;
        const PointT &pt = pcl_cloud.points[position];
        image.at<float>(row,col) = 1.f / pt.z;
      }
    }
    return image;
  }




  template<class PointT>
  V4R_EXPORTS
  inline cv::Mat
  ConvertUnorganizedPCLCloud2Image (const typename pcl::PointCloud<PointT> &pcl_cloud,
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
      pcl::PointCloud<PointT> organized_cloud;
      organized_cloud.width = width;
      organized_cloud.height = height;
      organized_cloud.is_dense = true;
      organized_cloud.points.resize(width*height);

      for(size_t kk=0; kk < organized_cloud.points.size(); kk++)
      {
          organized_cloud.points[kk].x = organized_cloud.points[kk].y = organized_cloud.points[kk].z =
                  std::numeric_limits<float>::quiet_NaN();

          organized_cloud.points[kk].r = bg_r;
          organized_cloud.points[kk].g = bg_g;
          organized_cloud.points[kk].b = bg_b;
      }

      int ws2 = 1;
      for (size_t kk = 0; kk < pcl_cloud.points.size (); kk++)
      {
          float x = pcl_cloud.points[kk].x;
          float y = pcl_cloud.points[kk].y;
          float z = pcl_cloud.points[kk].z;
          int u = static_cast<int> (f * x / z + cx);
          int v = static_cast<int> (f * y / z + cy);

          for(int uu = (u-ws2); uu < (u+ws2); uu++)
          {
              for(int vv = (v-ws2); vv < (v+ws2); vv++)
              {
                  //Not out of bounds
                  if ((uu >= width) || (vv >= height) || (uu < 0) || (vv < 0))
                      continue;

                  float z_oc = organized_cloud.at (uu, vv).z;

                  if(pcl_isnan(z_oc))
                  {
                      organized_cloud.at (uu, vv) = pcl_cloud.points[kk];
                  }
                  else
                  {
                      if(z < z_oc)
                      {
                          organized_cloud.at (uu, vv) = pcl_cloud.points[kk];
                      }
                  }
              }
          }
      }
      return ConvertPCLCloud2Image (organized_cloud, crop);
  }  

   /**
     * @brief computes the depth map of a point cloud with fixed size output
     * @param RGB-D cloud
     * @param indices of the points belonging to the object
     * @param out_height
     * @param out_width
     * @return depth image (float)
     */
    template<typename PointT>
    V4R_EXPORTS
    inline cv::Mat
    ConvertPCLCloud2DepthImageFixedSize(const pcl::PointCloud<PointT> &cloud, const std::vector<int> &cluster_idx, size_t out_height, size_t out_width)
    {
        volatile int min_u, min_v, max_u, max_v;
        max_u = max_v = 0;
        min_u = cloud.width;
        min_v = cloud.height;

    //    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster_tmp(new pcl::PointCloud<pcl::PointXYZRGB>);
    //    pcl::copyPointCloud(cloud,cluster_idx,*cluster_tmp);
    //    pcl::visualization::PCLVisualizer vis("vis");
    //    vis.addPointCloud(cluster_tmp,"cloud");
    //    vis.spin();

        for(size_t idx=0; idx<cluster_idx.size(); idx++) {
            int u = cluster_idx[idx] % cloud.width;
            int v = (int) (cluster_idx[idx] / cloud.width);

            if (u>max_u)
                max_u = u;

            if (v>max_v)
                max_v = v;

            if (u<min_u)
                min_u = u;

            if (v<min_v)
                min_v = v;
        }

        if ( (int)out_width > (max_u-min_u) ) {
            int margin_x = out_width - (max_u - min_u);

            volatile int margin_x_left = margin_x / 2.f;
            min_u = std::max (0, min_u - margin_x_left);
            volatile int margin_x_right = out_width - (max_u - min_u);
            max_u = std::min ((int)cloud.width, max_u + margin_x_right);
            margin_x_left = out_width - (max_u - min_u);
            min_u = std::max (0, min_u - margin_x_left);
        }

        if ( (int)out_height > (max_v - min_v) ) {
            int margin_y = out_height - (max_v - min_v);
            volatile int margin_y_left = margin_y / 2.f;
            min_v = std::max (0, min_v - margin_y_left);
            volatile int margin_y_right = out_height - (max_v - min_v);
            max_v = std::min ((int)cloud.height, max_v + margin_y_right);
            margin_y_left = out_height - (max_v - min_v);
            min_v = std::max (0, min_v - margin_y_left);
        }

        cv::Mat_<float> image(max_v - min_v, max_u - min_u);

        for (int row = 0; row < image.rows; row++) {
          for (int col = 0; col < image.cols; col++) {
            float & cvp = image.at<float> (row, col);
            int position = (row + min_v) * cloud.width + (col + min_u);
            const PointT &pt = cloud.points[position];

            cvp = 1.f / pt.z;
          }
        }

        cv::Mat_<float> dst(out_height, out_width);
        cv::resize(image, dst, dst.size(), 0, 0, cv::INTER_CUBIC);

        return dst;
    }



    /**
      * @brief computes the depth map of a point cloud with fixed size output
      * @param RGB-D cloud
      * @param indices of the points belonging to the object
      * @param out_height
      * @param out_width
      * @return depth image (unsigned int)
      */
     template<typename PointT>
     V4R_EXPORTS
     inline cv::Mat
     ConvertPCLCloud2UnsignedDepthImageFixedSize(const pcl::PointCloud<PointT> &cloud, const std::vector<int> &cluster_idx, size_t out_height, size_t out_width)
     {
         volatile int min_u, min_v, max_u, max_v;
         max_u = max_v = 0;
         min_u = cloud.width;
         min_v = cloud.height;

     //    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster_tmp(new pcl::PointCloud<pcl::PointXYZRGB>);
     //    pcl::copyPointCloud(cloud,cluster_idx,*cluster_tmp);
     //    pcl::visualization::PCLVisualizer vis("vis");
     //    vis.addPointCloud(cluster_tmp,"cloud");
     //    vis.spin();

         for(size_t idx=0; idx<cluster_idx.size(); idx++) {
             int u = cluster_idx[idx] % cloud.width;
             int v = (int) (cluster_idx[idx] / cloud.width);

             if (u>max_u)
                 max_u = u;

             if (v>max_v)
                 max_v = v;

             if (u<min_u)
                 min_u = u;

             if (v<min_v)
                 min_v = v;
         }

         if ( (int)out_width > (max_u-min_u) ) {
             int margin_x = out_width - (max_u - min_u);

             volatile int margin_x_left = margin_x / 2.f;
             min_u = std::max (0, min_u - margin_x_left);
             volatile int margin_x_right = out_width - (max_u - min_u);
             max_u = std::min ((int)cloud.width, max_u + margin_x_right);
             margin_x_left = out_width - (max_u - min_u);
             min_u = std::max (0, min_u - margin_x_left);
         }

         if ( (int)out_height > (max_v - min_v) ) {
             int margin_y = out_height - (max_v - min_v);
             volatile int margin_y_left = margin_y / 2.f;
             min_v = std::max (0, min_v - margin_y_left);
             volatile int margin_y_right = out_height - (max_v - min_v);
             max_v = std::min ((int)cloud.height, max_v + margin_y_right);
             margin_y_left = out_height - (max_v - min_v);
             min_v = std::max (0, min_v - margin_y_left);
         }

         cv::Mat_<unsigned int> image(max_v - min_v, max_u - min_u);

         for (int row = 0; row < image.rows; row++) {
           for (int col = 0; col < image.cols; col++) {
             unsigned int & cvp = image.at<unsigned int> (row, col);
             int position = (row + min_v) * cloud.width + (col + min_u);
             const PointT &pt = cloud.points[position];

             cvp = size_t(pt.z * 1000.f);
           }
         }

         cv::Mat_<unsigned int> dst(out_height, out_width);
         cv::resize(image, dst, dst.size(), 0, 0, cv::INTER_CUBIC);

         return dst;
     }
}

#endif /* PCL_OPENCV_H_ */
