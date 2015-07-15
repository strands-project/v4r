#ifndef FAAT_PCL_GHV_CUDA_WRAPPER_H_
#define FAAT_PCL_GHV_CUDA_WRAPPER_H_

#include <pcl/common/common.h>
#include <pcl/pcl_macros.h>
#include <v4r/ORRecognition/ghv_cuda.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_types.h>
#include <pcl/pcl_macros.h>
#include <pcl/gpu/containers/device_array.h>
#include <pcl/search/pcl_search.h>
#include <v4r/ORUtils/common_data_structures.h>

namespace faat_pcl
{
    namespace recognition
    {

      template<typename PointT>
      class GHVCudaWrapper
      {
        private:

          static float sRGB_LUT[256];
          static float sXYZ_LUT[4000];

          //////////////////////////////////////////////////////////////////////////////////////////////
          //float sRGB_LUT[256] = {- 1};

          //////////////////////////////////////////////////////////////////////////////////////////////
          //float sXYZ_LUT[4000] = {- 1};

          //////////////////////////////////////////////////////////////////////////////////////////////
          void
          RGB2CIELAB (unsigned char R, unsigned char G, unsigned char B, float &L, float &A,float &B2)
          {
            if (sRGB_LUT[0] < 0)
            {
              for (int i = 0; i < 256; i++)
              {
                float f = static_cast<float> (i) / 255.0f;
                if (f > 0.04045)
                  sRGB_LUT[i] = powf ((f + 0.055f) / 1.055f, 2.4f);
                else
                  sRGB_LUT[i] = f / 12.92f;
              }

              for (int i = 0; i < 4000; i++)
              {
                float f = static_cast<float> (i) / 4000.0f;
                if (f > 0.008856)
                  sXYZ_LUT[i] = static_cast<float> (powf (f, 0.3333f));
                else
                  sXYZ_LUT[i] = static_cast<float>((7.787 * f) + (16.0 / 116.0));
              }
            }

            float fr = sRGB_LUT[R];
            float fg = sRGB_LUT[G];
            float fb = sRGB_LUT[B];

            // Use white = D65
            const float x = fr * 0.412453f + fg * 0.357580f + fb * 0.180423f;
            const float y = fr * 0.212671f + fg * 0.715160f + fb * 0.072169f;
            const float z = fr * 0.019334f + fg * 0.119193f + fb * 0.950227f;

            float vx = x / 0.95047f;
            float vy = y;
            float vz = z / 1.08883f;

            vx = sXYZ_LUT[int(vx*4000)];
            vy = sXYZ_LUT[int(vy*4000)];
            vz = sXYZ_LUT[int(vz*4000)];

            L = 116.0f * vy - 16.0f;
            if (L > 100)
              L = 100.0f;

            A = 500.0f * (vx - vy);
            if (A > 120)
              A = 120.0f;
            else if (A <- 120)
              A = -120.0f;

            B2 = 200.0f * (vy - vz);
            if (B2 > 120)
              B2 = 120.0f;
            else if (B2<- 120)
              B2 = -120.0f;
          }

          typedef typename pcl::PointCloud<PointT>::ConstPtr ConstPointInTPtr;
          typedef typename pcl::PointCloud<PointT>::Ptr PointInTPtr;

          PointInTPtr scene_cloud_;
          PointInTPtr occlusion_cloud_;
          std::vector<ConstPointInTPtr> models_;
          std::vector<pcl::PointCloud<pcl::Normal>::ConstPtr> models_normals_;
          std::vector<Eigen::Matrix4f> transforms_;
          std::vector<int> transforms_to_models_;

          std::vector<faat_pcl::PlaneModel<PointT> > planar_model_hypotheses_;

          pcl::PointCloud<pcl::Normal>::Ptr scene_normals_;

          //smooth segmentation stuff
          double eps_angle_threshold_;
          int min_points_;
          float curvature_threshold_;
          float cluster_tolerance_;
          typename pcl::search::KdTree<PointT>::Ptr scene_downsampled_tree_;
          pcl::PointCloud<pcl::PointXYZL>::Ptr clusters_cloud_;
          pcl::PointCloud<pcl::PointXYZRGBA>::Ptr clusters_cloud_rgb_;

          //private functions
          void uploadToGPU(faat_pcl::recognition_cuda::GHV & ghv_);

          void extractEuclideanClustersSmooth (const typename pcl::PointCloud<PointT> &cloud, const pcl::PointCloud<pcl::Normal> &normals, float tolerance,
                                               const typename pcl::search::Search<PointT>::Ptr &tree, std::vector<pcl::PointIndices> &clusters, double eps_angle,
                                               float curvature_threshold, unsigned int min_pts_per_cluster,
                                               unsigned int max_pts_per_cluster = (std::numeric_limits<int>::max) ());

          void smoothSceneSegmentation();

          boost::shared_ptr<pcl::visualization::PCLVisualizer> vis_;
          std::vector<bool> sol;
          int visible_points_;
          float cues_time_;
          float t_opt_;

          float inlier_threshold;
          float clutter_regularizer_;
          float outlier_regularizer_;
          float clutter_radius_;
          float color_sigma_y_, color_sigma_ab_;

        public:
          GHVCudaWrapper();

          void addModels(std::vector<ConstPointInTPtr> & models,
                         std::vector<Eigen::Matrix4f> & transforms,
                         std::vector<int> & transforms_to_models)
          {
               models_ = models;
               transforms_ = transforms;
               transforms_to_models_ = transforms_to_models;
          }

          void setColorSigma(float cs_y, float cs_ab)
          {
              color_sigma_y_ = cs_y;
              color_sigma_ab_ = cs_ab;
          }

          void setRadiusClutter(float f)
          {
              clutter_radius_ = f;
          }

          void setInlierThreshold(float i)
          {
              inlier_threshold = i;
          }

          void setRegularizer(float i) //Outlier weight
          {
              outlier_regularizer_ = i;
          }

          void setClutterRegularizer(float i)
          {
              clutter_regularizer_ = i;
          }

          void addModelNormals(std::vector<pcl::PointCloud<pcl::Normal>::ConstPtr> & models)
          {
              models_normals_ = models;
          }

          void setNormalsForClutterTerm(pcl::PointCloud<pcl::Normal>::Ptr & normals)
          {
              scene_normals_ = normals;
          }

          void setSceneCloud(PointInTPtr & cloud)
          {
              scene_cloud_ = cloud;
          }

          void setOcclusionCloud(PointInTPtr & occlusion_cloud)
          {
              occlusion_cloud_ = occlusion_cloud;
          }

          void addPlanarModels(std::vector<faat_pcl::PlaneModel<PointT> > & models);

          void verify();

          void getMask(std::vector<bool> &mask)
          {
              mask = sol;
          }

          int getNumberOfVisiblePoints()
          {
              return visible_points_;
          }

          float getOptimizationTime()
          {
              return t_opt_;
          }

          float getCuesComputationTime()
          {
              return cues_time_;
          }
      };

    }
}

#endif
