#include <v4r/common/pcl_opencv.h>
#include <v4r/common/miscellaneous.h>
#include <glog/logging.h>
#include <omp.h>


namespace v4r
{
    template<typename PointT>
    V4R_EXPORTS
    cv::Mat
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

    template<typename PointT>
    V4R_EXPORTS
    cv::Mat
    ConvertUnorganizedPCLCloud2Image (const typename pcl::PointCloud<PointT> &pcl_cloud,
                                      bool crop,
                                      float bg_r,
                                      float bg_g,
                                      float bg_b,
                                      int width,
                                      int height,
                                      float f,
                                      float cx,
                                      float cy)
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

    template<typename PointT>
    V4R_EXPORTS
    cv::Mat
    ConvertPCLCloud2Image (const typename pcl::PointCloud<PointT> &pcl_cloud, bool crop)
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

      if(crop)
      {
          cv::Mat cropped_image = image(cv::Rect(min_u, min_v, max_u - min_u, max_v - min_v));
          image = cropped_image.clone();
      }

      return image;
    }


    template<typename PointT>
    V4R_EXPORTS
    cv::Mat
    ConvertPCLCloud2FixedSizeImage(const typename pcl::PointCloud<PointT> &cloud, const std::vector<int> &cluster_idx,
                                   size_t out_height, size_t out_width, size_t margin,
                                   cv::Scalar bg_color, bool do_closing_operation)
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

    template<typename PointT>
    V4R_EXPORTS
    cv::Mat
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

    template<class PointT>
    V4R_EXPORTS
    std::vector<bool>
    ConvertPCLCloud2OccupancyImage (const typename pcl::PointCloud<PointT> &cloud, int width, int height, float f, float cx, float cy)
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


    template<typename PointT>
    V4R_EXPORTS
    cv::Mat
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

    template<typename PointT>
    V4R_EXPORTS
    cv::Mat
    pcl2cvMat (const typename pcl::PointCloud<PointT> &cloud, const std::vector<int> &indices, bool crop, bool remove_background, int margin)
    {
        cv::Mat_<cv::Vec3b> image (cloud.height, cloud.width);
        for (size_t v = 0; v < cloud.height; v++)
        {
            for (size_t u = 0; u < cloud.width; u++)
            {
                cv::Vec3b & cvp = image.at<cv::Vec3b> (v, u);
                const PointT &pt = cloud.at(u,v);
                cvp[0] = pt.b;
                cvp[1] = pt.g;
                cvp[2] = pt.r;
            }
        }

        if ( !indices.empty() && remove_background)
        {
            std::vector<bool> fg_mask = v4r::createMaskFromIndices(indices, cloud.width * cloud.height);
            cv::Vec3b bg_color (0,0,0);
            for (size_t row = 0; row < cloud.height; row++)
            {
                for (size_t col = 0; col < cloud.width; col++)
                {
                    if( !fg_mask[row * cloud.width + col] )
                        image.at<cv::Vec3b> (row, col) = bg_color;
                }
            }
        }

        if(  !indices.empty() &&  crop )
        {
            cv::Rect roi = computeBoundingBox(indices, cloud.width, cloud.height, margin);

            // make roi square
            if(roi.width > roi.height)
            {
                int extension_half = (roi.width - roi.height) / 2;
                roi.y = std::max(0, roi.y - extension_half);
                roi.height = std::min<int>(cloud.height, roi.width);
            }
            else
            {
                int extension_half = (roi.height - roi.width) / 2;
                roi.x = std::max(0, roi.x - extension_half);
                roi.width = std::min<int>(cloud.width, roi.height);
            }

            cv::Mat image_roi = image( roi );
            image = image_roi.clone();
        }
        return image;
    }

    template<typename PointT>
    V4R_EXPORTS
    cv::Mat
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


    /**
     * @brief pcl2depthMatDouble extracts depth image from pointcloud whereby depth values correspond to distance in meter
     * @param[in] cloud
     * @return depth image in meter
     */
    template<typename PointT>
    V4R_EXPORTS
    cv::Mat
    pcl2depthMatDouble (const typename pcl::PointCloud<PointT> &cloud)
    {
        cv::Mat_<double> image (cloud.height, cloud.width);
        for (size_t v = 0; v < cloud.height; v++)
        {
            for (size_t u = 0; u < cloud.width; u++)
            {
                double & cvp = image.at<double> (v, u);
                const PointT &pt = cloud.at(u,v);
                pcl_isfinite(pt.z) ? cvp=pt.z : cvp=0.f;
            }
        }
        return image;
    }


    template<typename PointT>
    V4R_EXPORTS
    cv::Mat
    pcl2depthMat (const typename pcl::PointCloud<PointT> &cloud, float min_depth, float max_depth)
    {
        cv::Mat_<uchar> image (cloud.height, cloud.width);

        for (unsigned v = 0; v < cloud.height; v++)
        {
            for (unsigned u = 0; u < cloud.width; u++)
            {
                unsigned char & cvp = image.at<unsigned char> (v, u);
                const PointT &pt = cloud.at(u,v);
                pcl_isfinite(pt.z) ? cvp = std::min<unsigned char>(255, std::max<unsigned char>(0, 255.f*(pt.z-min_depth)/(max_depth-min_depth)) ) : cvp = 0.f;
            }
        }

        return image;
    }

    template<typename PointT>
    V4R_EXPORTS
    cv::Mat
    pcl2depthMat (const typename pcl::PointCloud<PointT> &pcl_cloud,
               const std::vector<int> &indices, bool crop, bool remove_background, int margin)
    {
        CHECK(!indices.empty());

        std::vector<bool> fg_mask = v4r::createMaskFromIndices(indices, pcl_cloud.width * pcl_cloud.height);

        int min_u = pcl_cloud.width-1, min_v = pcl_cloud.height-1, max_u = 0, max_v = 0;

        cv::Mat_<uchar> image (pcl_cloud.height, pcl_cloud.width);
        uchar bg_color = 255;
        image.setTo(bg_color);

        for (unsigned row = 0; row < pcl_cloud.height; row++)
        {
            for (unsigned col = 0; col < pcl_cloud.width; col++)
            {
                unsigned position = row * pcl_cloud.width + col;

                uchar & cvp = image.at<uchar> (row, col);
                const pcl::PointXYZRGB &pt = pcl_cloud.points[position];

                float depth = pt.z;

                if(remove_background && !fg_mask[position])
                    continue;

                cvp = std::min<unsigned char>(255, static_cast<unsigned char> (255.f * depth / 3.0f));

                if(!fg_mask[position])
                    continue;

                if( pcl::isFinite(pt) ) {
                    if( row < min_v ) min_v = row;
                    if( row > max_v ) max_v = row;
                    if( col < min_u ) min_u = col;
                    if( col > max_u ) max_u = col;
                }
            }
        }

        if(crop) {
            int side_length_u = max_u - min_u;
            int side_length_v = max_v - min_v;
            int side_length = std::max<int>(side_length_u , side_length_v) + 2 * margin;

            // center object in the middle
            min_u = std::max<int>(0, int(min_u - (side_length - side_length_u)/2.f));
            min_v = std::max<int>(0, int(min_v - (side_length - side_length_v)/2.f));

            int side_length_uu = std::min<int>(side_length, image.cols  - min_u - 1);
            int side_length_vv = std::min<int>(side_length, image.rows - min_v - 1);

            cv::Mat image_roi = image( cv::Rect(min_u, min_v, side_length_uu, side_length_vv) );
            image = image_roi.clone();
        }

        return image;
    }

    V4R_EXPORTS
    cv::Rect
    computeBoundingBox (const std::vector<int> &indices, size_t width, size_t height, int margin)
    {
        CHECK (!indices.empty());
        int min_u = width-1, min_v = height-1, max_u = 0, max_v = 0;

        for (size_t i = 0; i < indices.size(); i++)
        {
            const int &idx = indices[i];

            int col = idx % width;
            int row = idx / width;

            if( row < min_v ) min_v = row;
            if( row > max_v ) max_v = row;
            if( col < min_u ) min_u = col;
            if( col > max_u ) max_u = col;
        }

        min_u = std::max<int>(0, min_u - margin);
        min_v = std::max<int>(0, min_v - margin);
        max_u = std::min<int>(width-1, max_u + margin);
        max_v = std::min<int>(height-1, max_v + margin);

        return cv::Rect( cv::Point(min_u, min_v), cv::Point (max_u, max_v) );
    }

    template V4R_EXPORTS cv::Mat ConvertPCLCloud2Image<pcl::PointXYZRGB> (const typename pcl::PointCloud<pcl::PointXYZRGB> &, bool);
    template V4R_EXPORTS cv::Mat ConvertPCLCloud2Image<pcl::PointXYZRGBA> (const typename pcl::PointCloud<pcl::PointXYZRGBA> &, bool);
    template V4R_EXPORTS cv::Mat ConvertPCLCloud2FixedSizeImage<pcl::PointXYZRGB>(const typename pcl::PointCloud<pcl::PointXYZRGB> &cloud, const std::vector<int> &, size_t, size_t, size_t, cv::Scalar, bool);
    template V4R_EXPORTS cv::Mat ConvertPCLCloud2DepthImage<pcl::PointXYZRGB>(const pcl::PointCloud<pcl::PointXYZRGB> &);
    template V4R_EXPORTS cv::Mat ConvertPCLCloud2Image<pcl::PointXYZRGB>(const pcl::PointCloud<pcl::PointXYZRGB> &, const std::vector<int> &, size_t, size_t);
    template V4R_EXPORTS cv::Mat ConvertUnorganizedPCLCloud2Image<pcl::PointXYZRGB>(const pcl::PointCloud<pcl::PointXYZRGB> &, bool, float, float, float, int, int, float, float, float);
    template V4R_EXPORTS cv::Mat ConvertPCLCloud2DepthImageFixedSize<pcl::PointXYZRGB>(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const std::vector<int> &, size_t, size_t);
    template V4R_EXPORTS cv::Mat pcl2cvMat<pcl::PointXYZRGB> (const pcl::PointCloud<pcl::PointXYZRGB> &, const std::vector<int>&, bool, bool, int);
    template V4R_EXPORTS cv::Mat ConvertPCLCloud2UnsignedDepthImageFixedSize<pcl::PointXYZRGB>(const pcl::PointCloud<pcl::PointXYZRGB> &, const std::vector<int> &, size_t, size_t);
    template V4R_EXPORTS cv::Mat pcl2depthMatDouble<pcl::PointXYZRGB> (const pcl::PointCloud<pcl::PointXYZRGB> &);
    template V4R_EXPORTS cv::Mat pcl2depthMat<pcl::PointXYZRGB> (const pcl::PointCloud<pcl::PointXYZRGB>&, float, float);
    template V4R_EXPORTS cv::Mat pcl2depthMat<pcl::PointXYZRGB> (const pcl::PointCloud<pcl::PointXYZRGB>&, const std::vector<int>&, bool, bool, int);
}
