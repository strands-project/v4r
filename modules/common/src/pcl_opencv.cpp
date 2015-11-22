#include <v4r/common/pcl_opencv.h>

namespace v4r
{
    template<class PointT>
    void
    ConvertPCLCloud2Image (const typename pcl::PointCloud<PointT>::Ptr &pcl_cloud, cv::Mat_<cv::Vec3b> &image)
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
  ConvertPCLCloud2Image (const typename pcl::PointCloud<PointT>::Ptr &pcl_cloud, cv::Mat_<cv::Vec3b> &image, bool crop)
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
  ConvertPCLCloud2DepthImage (const typename pcl::PointCloud<PointT>::Ptr &pcl_cloud, cv::Mat_<float> &image)
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
                                    cv::Mat_<cv::Vec3b> &image, bool crop, float bg_r,
                                    float bg_g, float bg_b, int width, int height,
                                    float f, float cx, float cy)
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


  template<typename PointT>
  cv::Mat
  ConvertPCLCloud2Image(const typename pcl::PointCloud<PointT> &cloud,
                        const std::vector<int> &cluster_idx,
                        size_t out_height, size_t out_width)
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

      if ( out_width > (max_u-min_u) ) {
          int margin_x = out_width - (max_u - min_u);

          volatile int margin_x_left = margin_x / 2.f;
          min_u = std::max (0, min_u - margin_x_left);
          volatile int margin_x_right = out_width - (max_u - min_u);
          max_u = std::min ((int)cloud.width, max_u + margin_x_right);
          margin_x_left = out_width - (max_u - min_u);
          min_u = std::max (0, min_u - margin_x_left);
      }

      if ( out_height > (max_v - min_v) ) {
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

  //    cv::namedWindow("bla");
  //    cv::imshow("bla",image);
  //    cv::namedWindow("dst");
  //    cv::imshow("dst",dst);
  //    cv::waitKey(0);

      return dst;
  }


  /**
   * @brief transforms an RGB-D point cloud into an image with fixed size
   * @param RGB-D cloud
   * @param indices of the points belonging to the object
   * @param out_height
   * @param out_width
   * @return image
   */
  template<class PointT>
  cv::Mat
  pcl2imageFixedSize(const pcl::PointCloud<PointT> &cloud,
                     const std::vector<int> &cluster_idx,
                     size_t out_height, size_t out_width)
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

      cv::Mat_<cv::Vec3b> image(max_v - min_v, max_u - min_u);

      for (int row = 0; row < image.rows; row++) {
        for (int col = 0; col < image.cols; col++) {
          cv::Vec3b & cvp = image.at<cv::Vec3b> (row, col);
          int position = (row + min_v) * cloud.width + (col + min_u);
          const pcl::PointXYZRGB &pt = cloud.points[position];

          cvp[0] = pt.b;
          cvp[1] = pt.g;
          cvp[2] = pt.r;
        }
      }

      cv::Mat_<cv::Vec3b> dst(out_height, out_width);
      cv::resize(image, dst, dst.size(), 0, 0, cv::INTER_CUBIC);

      return dst;
  }


  template V4R_EXPORTS void ConvertPCLCloud2Image<pcl::PointXYZRGB> (const typename pcl::PointCloud<pcl::PointXYZRGB>::Ptr &, cv::Mat_<cv::Vec3b> &, bool);
  template V4R_EXPORTS cv::Mat ConvertPCLCloud2Image<pcl::PointXYZRGB> (const typename pcl::PointCloud<pcl::PointXYZRGB> &, const std::vector<int> &, size_t, size_t);
  template V4R_EXPORTS void ConvertPCLCloud2DepthImage<pcl::PointXYZ> (const typename pcl::PointCloud<pcl::PointXYZ>::Ptr &, cv::Mat_<float> &);
  template V4R_EXPORTS void ConvertPCLCloud2DepthImage<pcl::PointXYZRGB> (const typename pcl::PointCloud<pcl::PointXYZRGB>::Ptr &, cv::Mat_<float> &);
  template V4R_EXPORTS void ConvertUnorganizedPCLCloud2Image<pcl::PointXYZRGB> (const typename pcl::PointCloud<pcl::PointXYZRGB>::Ptr &, cv::Mat_<cv::Vec3b> &, bool, float, float, float, int, int, float, float, float);
  template V4R_EXPORTS cv::Mat pcl2imageFixedSize<pcl::PointXYZRGB>(const pcl::PointCloud<pcl::PointXYZRGB> &, const std::vector<int>&, size_t, size_t );
}
