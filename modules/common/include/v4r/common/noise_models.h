/*
 * noise_models.h
 *
 *  Created on: Oct 28, 2013
 *      Author: aitor
 */

#ifndef NOISE_MODELS_H_
#define NOISE_MODELS_H_

#include <pcl/common/common.h>
#include <pcl/common/io.h>

#include <v4r/core/macros.h>

namespace v4r
{
    namespace noise_models
    {
      template<class PointT>
        class V4R_EXPORTS NguyenNoiseModel
        {
        public:
            class Parameter
            {
            public:
                float lateral_sigma_;
                float max_angle_;
                bool use_depth_edges_;
                float focal_length_;
                float weight_edge_;
                int dilate_width_;
                int dilate_iterations_;

                Parameter(
                        float lateral_sigma = 0.002f,
                        float max_angle = 70.f,
                        bool use_depth_edges = true,
                        float focal_length = 525.f,
                        float weight_edge = 5.f,
                        int dilate_width = 5,
                        int dilate_iterations = 3)
                        :
                          lateral_sigma_ (lateral_sigma),
                          max_angle_ ( max_angle ),
                          use_depth_edges_( use_depth_edges ),
                          focal_length_ (focal_length),
                          weight_edge_ (weight_edge),
                          dilate_width_ (dilate_width),
                          dilate_iterations_ (dilate_iterations)
                        {}
            }param_;

        private:
          typedef typename pcl::PointCloud<PointT>::Ptr PointTPtr;
          typedef typename pcl::PointCloud<pcl::Normal>::Ptr PointNormalTPtr;
          PointTPtr input_;
          PointNormalTPtr normals_;
          std::vector<float> weights_;
          std::vector<float> sigmas_;
          pcl::PointIndices discontinuity_edges_;
          Eigen::Matrix4f pose_to_plane_RF_;
          bool pose_set_;

        public:

          NguyenNoiseModel (const Parameter &param=Parameter());

          //this is the pose used to align a cloud so that its aligned to the RF
          //defined on a plane (z-axis corresponds to the plane normal) and
          //the origin is on the plane

          void setPoseToPlaneRF(Eigen::Matrix4f & pose)
          {
              pose_to_plane_RF_ = pose;
              pose_set_ = true;
          }

          void
          setInputCloud (const PointTPtr & input)
          {
            input_ = input;
          }

          void
          setMaxAngle(float f)
          {
            param_.max_angle_ = f;
          }

          void
          setUseDepthEdges(bool b)
          {
            param_.use_depth_edges_ = b;
          }

          void
          getDiscontinuityEdges(pcl::PointCloud<pcl::PointXYZ>::Ptr & disc) const
          {
            disc.reset(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::copyPointCloud(*input_, discontinuity_edges_, *disc);
          }

          //in meters, lateral sigma (3*s used to downweight points close to depth discontinuities)
          void
          setLateralSigma(float s)
          {
            param_.lateral_sigma_ = s;
          }

          void
          setInputNormals (const PointNormalTPtr & normals)
          {
            normals_ = normals;
          }

          void
          compute ();

          void
          getWeights (std::vector<float> & weights) const
          {
            weights = weights_;
          }

          std::vector<float>
          getSigmas () const
          {
            return sigmas_;
          }

          //void getFilteredCloud(PointTPtr & filtered, float w_t);

          void getFilteredCloudRemovingPoints(PointTPtr & filtered, float w_t);

          void getFilteredCloudRemovingPoints(PointTPtr & filtered, float w_t, std::vector<int> & kept);
        };
    }
}

#endif /* NOISE_MODELS_H_ */

