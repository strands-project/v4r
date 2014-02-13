/*
 * depth_inpainting.h
 *
 *  Created on: Oct 28, 2013
 *      Author: aitor
 */

#ifndef DEPTH_INPAINTING_H_
#define DEPTH_INPAINTING_H_

#include <pcl/common/common.h>
#include <pcl/common/io.h>

#include <faat_pcl/utils/pcl_opencv.h>

namespace faat_pcl
{
  namespace utils
  {
    namespace depth_inpainting
    {
      template<class PointT>
        class DepthInpainter
        {
        protected:
          typedef typename pcl::PointCloud<PointT>::Ptr PointTPtr;
          PointTPtr input_;
          int cx_, cy_;
          float fx_, fy_;
        public:
          void
          setCameraParameters (int cx, int cy, float fx, float fy)
          {
            cx_ = cx;
            cy_ = cy;
            fx_ = fx;
            fy_ = fy;
          }

          void
          setInputCloud (PointTPtr & cloud)
          {
            input_ = cloud;
          }

          virtual void
          inpaint (PointTPtr & inpainted_cloud) = 0;

          void
          generateCloudOnInterpolated (PointTPtr & inpainted_cloud, cv::Mat & depth)
          {
            inpainted_cloud.reset (new pcl::PointCloud<PointT> (*input_));
            for (int j = 0; j < inpainted_cloud->height; j++)
            {
              for (int i = 0; i < inpainted_cloud->width; i++)
              {
                if (pcl_isfinite(inpainted_cloud->at(i,j).z))
                  continue;

                if (depth.at<float> (j, i) > 0)
                {
                  //generate x,y,z from interpolated data
                  inpainted_cloud->at (i, j).z = 1.f / depth.at<float> (j, i);
                  inpainted_cloud->at (i, j).x = (i - cx_) * inpainted_cloud->at (i, j).z / fx_;
                  inpainted_cloud->at (i, j).y = (j - cy_) * inpainted_cloud->at (i, j).z / fy_;

                  //std::cout << "Adding interpolated point:" << std::endl;
                  //std::cout << inpainted_cloud->at(i,j).getVector3fMap() << std::endl;

                  /*inpainted_cloud->at(i,j).r = 255;
                   inpainted_cloud->at(i,j).g = 0;
                   inpainted_cloud->at(i,j).b = 0;*/
                }
              }
            }
          }
        };

      /*if (pcl_isfinite (depth_focal_length_x_))
       constant_x =  1.0f / static_cast<float> (depth_focal_length_x_);

       if (pcl_isfinite (depth_focal_length_y_))
       constant_y =  1.0f / static_cast<float> (depth_focal_length_y_);

       for (int v = 0; v < depth_height_; ++v)
       {
       for (register int u = 0; u < depth_width_; ++u, ++depth_idx)
       {
       pcl::PointXYZ& pt = cloud->points[depth_idx];
       // Check for invalid measurements
       if (depth_map[depth_idx] == 0 ||
       depth_map[depth_idx] == depth_image->getNoSampleValue () ||
       depth_map[depth_idx] == depth_image->getShadowValue ())
       {
       // not valid
       pt.x = pt.y = pt.z = bad_point;
       continue;
       }
       pt.z = depth_map[depth_idx] * 0.001f;
       pt.x = (static_cast<float> (u) - centerX) * pt.z * constant_x;
       pt.y = (static_cast<float> (v) - centerY) * pt.z * constant_y;
       }
       }*/

      template<class PointT>
        class NNDepthInpainter : public DepthInpainter<PointT>
        {
        private:
          typedef typename pcl::PointCloud<PointT>::Ptr PointTPtr;
          using DepthInpainter<PointT>::input_;
          float MAX_DIST_TO_INTERPOLATE_;

        public:
          NNDepthInpainter ()
          {
            MAX_DIST_TO_INTERPOLATE_ = 3.f;
          }

          void
          setMaxDistInPixels (float f)
          {
            MAX_DIST_TO_INTERPOLATE_ = f;
          }

          void
          inpaint (PointTPtr & inpainted_cloud)
          {
            cv::Mat_<float> dImage;
            PCLOpenCV::ConvertPCLCloud2DepthImage<PointT> (input_, dImage);
            cv::namedWindow ("dImage");
            cv::imshow ("dImage", dImage);

            cv::Mat binary_image = cv::Mat (dImage.rows, dImage.cols, CV_8UC1);
            for (int j = 0; j < binary_image.rows; j++)
            {
              for (int i = 0; i < binary_image.cols; i++)
              {
                if (dImage.at<float> (j, i) > 0)
                {
                  binary_image.at<unsigned char> (j, i) = 0;
                }
                else
                {
                  binary_image.at<unsigned char> (j, i) = 255;
                }
              }
            }

            cv::Mat dt, dt_weighted, labels;
            cv::distanceTransform (binary_image, dt, labels, CV_DIST_L2, 5, cv::DIST_LABEL_PIXEL);

            std::vector<float> label_to_depth;
            label_to_depth.resize (labels.rows * labels.cols, -1);

            for (int j = 0; j < labels.rows; j++)
            {
              for (int i = 0; i < labels.cols; i++)
              {
                if (dImage.at<float> (j, i) > 0)
                {
                  //known depth values, which label is used?
                  label_to_depth[labels.at<int> (j, i)] = dImage.at<float> (j, i);
                }
              }
            }

            for (int j = 0; j < labels.rows; j++)
            {
              for (int i = 0; i < labels.cols; i++)
              {
                if (dImage.at<float> (j, i) > 0)
                {
                  //known depth values, which label is used?
                }
                else
                {
                  if (dt.at<float> (j, i) > MAX_DIST_TO_INTERPOLATE_)
                    continue;

                  dImage.at<float> (j, i) = label_to_depth[labels.at<int> (j, i)];
                }
              }
            }

            cv::namedWindow ("dImage (filled)");
            cv::imshow ("dImage (filled)", dImage);
            cv::waitKey (0);

            generateCloudOnInterpolated (inpainted_cloud, dImage);
          }
        };

      template<class PointT>
        class RMFDepthInpainter : public DepthInpainter<PointT>
        {
        private:
          typedef typename pcl::PointCloud<PointT>::Ptr PointTPtr;
          using DepthInpainter<PointT>::input_;
          float MAX_DIST_TO_INTERPOLATE_;

        public:
          RMFDepthInpainter ()
          {
            MAX_DIST_TO_INTERPOLATE_ = 3.f;
          }

          void
          setMaxDistInPixels (float f)
          {
            MAX_DIST_TO_INTERPOLATE_ = f;
          }

          void
          inpaint (PointTPtr & inpainted_cloud)
          {
            cv::Mat_<float> dImage;
            PCLOpenCV::ConvertPCLCloud2DepthImage<PointT> (input_, dImage);
            cv::namedWindow ("dImage");
            cv::imshow ("dImage", dImage);

            cv::Mat binary_image = cv::Mat (dImage.rows, dImage.cols, CV_8UC1);
            for (int j = 0; j < binary_image.rows; j++)
            {
              for (int i = 0; i < binary_image.cols; i++)
              {
                if (dImage.at<float> (j, i) > 0)
                {
                  binary_image.at<unsigned char> (j, i) = 0;
                }
                else
                {
                  binary_image.at<unsigned char> (j, i) = 255;
                }
              }
            }

            cv::Mat dt, dt_weighted, labels;
            cv::distanceTransform (binary_image, dt, CV_DIST_L2, CV_DIST_MASK_PRECISE);

            typedef std::pair<int, float> idx_distance;
            std::vector<idx_distance> unseen_values_with_distance;
            for (int j = 0; j < binary_image.rows; j++)
            {
              for (int i = 0; i < binary_image.cols; i++)
              {
                if (dImage.at<float> (j, i) > 0)
                {
                }
                else
                {
                  if (dt.at<float> (j, i) > MAX_DIST_TO_INTERPOLATE_)
                    continue;

                  //binary_image.at<unsigned char> (j, i) = 255;
                  idx_distance id = std::make_pair(j * binary_image.cols + i, dt.at<float>(j,i));
                  unseen_values_with_distance.push_back(id);
                }
              }
            }

            std::sort(unseen_values_with_distance.begin(), unseen_values_with_distance.end(),
                      boost::bind(&std::pair<int, float>::second, _1) <
                      boost::bind(&std::pair<int, float>::second, _2));

            int wsize = 5;
            int wsize2 = wsize / 2;
            //apply median filter
            for(size_t kk=0; kk < unseen_values_with_distance.size(); kk++)
            {
              //std::cout << unseen_values_with_distance[kk].first << " " << unseen_values_with_distance[kk].second << std::endl;
              int j,i;
              j = unseen_values_with_distance[kk].first / binary_image.cols;
              i = unseen_values_with_distance[kk].first % binary_image.cols;

              //compute median in the support area
              std::vector<float> support_depths;
              for(int jk=std::max(0,j-wsize2); jk < std::min(binary_image.rows - wsize2, j+wsize2); jk++)
              {
                for(int ik=std::max(0,i-wsize2); ik < std::min(binary_image.cols - wsize2, i+wsize2); ik++)
                {
                  float d = dImage.at<float> (jk, ik);
                  if (d > 0)
                    support_depths.push_back(d);
                }
              }

              if(support_depths.size() > 0)
              {
                std::sort(support_depths.begin(),support_depths.end());
                float new_d = support_depths[support_depths.size() / 2];
                dImage.at<float>(j,i) = new_d;
              }
            }

            cv::namedWindow ("dImage (filled)");
            cv::imshow ("dImage (filled)", dImage);
            cv::waitKey (0);

            generateCloudOnInterpolated (inpainted_cloud, dImage);
          }
        };
    }
  }
}

#endif /* DEPTH_INPAINTING_H_ */
