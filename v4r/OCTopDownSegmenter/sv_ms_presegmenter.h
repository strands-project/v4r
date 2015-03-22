#ifndef V4R_MUMFORD_SV
#define V4R_MUMFORD_SV

#include "pre_segmenter.h"
#include "opencv2/opencv.hpp"
#include <pcl/visualization/pcl_visualizer.h>
#include "ms_utils.h"

namespace v4rOCTopDownSegmenter
{

    template<typename PointT>
    class SVMumfordShahPreSegmenter : public PreSegmenter<PointT>
    {
        protected:
            std::vector<Line> los_;
            pcl::PointCloud<pcl::Normal>::Ptr surface_normals_;
            using PreSegmenter<PointT>::cloud_;
            Eigen::VectorXd cloud_z_;

            std::vector<uint32_t> label_colors_;
            pcl::PointCloud<pcl::PointXYZL>::Ptr supervoxels_labels_cloud_;
            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr supervoxels_rgb_;
            virtual int extractSVBoundaries();

            void fillPixelMoves(std::vector<std::vector<PixelMove> > & pixel_moves,
                                int rs, int cs, int rend, int cend);

            virtual void refinement(std::vector<boost::shared_ptr<Region<PointT> > > & regions);

            virtual void computePlaneError(MergeCandidate<PointT> & m)
            {
                m.planar_error_ = m.computePlaneError(cloud_, los_);
            }

            virtual void computePlaneError(boost::shared_ptr<Region<PointT> > & r)
            {
                r->planar_error_ = r->computePlaneError(cloud_, los_);
            }

            virtual void computeBSplineError(MergeCandidate<PointT> & m, int order, int Cpx, int Cpy)
            {
                //m.bspline_error_ = m.computeBsplineError(cloud_z_, cloud_->width, order, Cpx, Cpy);
                m.bspline_error_ = m.computeBsplineErrorUnorganized(cloud_, order, Cpx, Cpy);
            }

            //std::vector< std::vector<MergeCandidate<PointT> > > merge_candidates_;
            std::vector< MergeCandidate<PointT> > merge_candidates_;
            std::vector< std::vector<int> > adjacent_;

            virtual void visualizeRegions(std::vector<boost::shared_ptr<Region<PointT> > > & regions, int viewport);

            virtual void visualizeSegmentation(std::vector<boost::shared_ptr<Region<PointT> > > & regions, int viewport);

            void computeMeanAndCovarianceMatrixSVMS(Eigen::Matrix3d & covariance,
                                                    Eigen::Vector3d & mean,
                                                    Eigen::Matrix<double, 1, 9, Eigen::RowMajor> & accu,
                                                    std::vector<int> & indices);

            void assertAdjacencyAndCandidateConsistency();

            void computeColorMean(Eigen::Vector3d & mean, std::vector<int> & indices);

            void createLabelCloud(std::vector<boost::shared_ptr<Region<PointT> > > & regions);

            float nyu_; //boundary length penalizer
            float lambda_; //model complexity penalizer
            float sigma_; //color regularizer
            float alpha_; //data penalty
            float ds_resolution_;
            bool vis_at_each_move_;
            boost::shared_ptr<pcl::visualization::PCLVisualizer> vis_segmentation_;
            boost::shared_ptr<pcl::visualization::PCLVisualizer> vis_cc;

            void fillLineOfSights();

            void addSupervoxelConnectionsToViewer (PointT &supervoxel_center,
                                                  typename pcl::PointCloud<PointT> &adjacent_supervoxel_centers,
                                                  std::string supervoxel_name,
                                                  boost::shared_ptr<pcl::visualization::PCLVisualizer> & viewer);

            void overSegmentation(std::vector< std::vector<int> > & adjacent,
                                  std::vector<int> & label_to_idx);

            int CURRENT_MODEL_TYPE_;
            int MAX_MODEL_TYPE_;
            bool pixelwise_refinement_;

            virtual void projectRegionsOnImage(std::vector<boost::shared_ptr<Region<PointT> > > & regions,
                                       std::string append);

            virtual void prepare(std::vector<int> & label_to_idx,
                                 std::vector<std::vector<int> > & indices);

            std::string save_impath_;
            pcl::PointCloud<pcl::PointXYZL>::Ptr labeled_cloud_;

            //supervoxels parameters
            float supervoxel_seed_resolution_;
            float supervoxel_resolution_;
            float color_importance_;
            float spatial_importance_;
            float normal_importance_;

            int boundary_window_;
            bool use_SLIC_RGBD_;

            std::vector<Eigen::Vector3d> scene_LAB_values_;

            std::vector<float> sRGB_LUT;
            std::vector<float> sXYZ_LUT;

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

              assert(R < 256 && R >= 0);
              assert(G < 256 && G >= 0);
              assert(B < 256 && B >= 0);

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

              vx = sXYZ_LUT[std::max(int(vx*4000), 4000)];
              vy = sXYZ_LUT[std::max(int(vy*4000), 4000)];
              vz = sXYZ_LUT[std::max(int(vz*4000), 4000)];

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


        public:

            SVMumfordShahPreSegmenter();
            ~SVMumfordShahPreSegmenter();

            void setSaveImPath(std::string & s)
            {
                save_impath_ = s;
            }

            void setBoundaryWindow(int w)
            {
                boundary_window_ = w;
            }

            void setSVParams(float seed_res, float res)
            {
                supervoxel_seed_resolution_ = seed_res;
                supervoxel_resolution_ = res;
            }

            void setSVImportanceValues(float color, float spatial, float normal)
            {
                color_importance_ = color;
                spatial_importance_ = spatial;
                normal_importance_ = normal;
            }

            void getLabelCloud(pcl::PointCloud<pcl::PointXYZL>::Ptr & labeled)
            {
                labeled.reset(new pcl::PointCloud<pcl::PointXYZL>(*labeled_cloud_));
            }

            //compute a pre-segmentation of the scene based on supervoxels and MS functional (Koepfler)
            void process();

            void setSurfaceNormals(pcl::PointCloud<pcl::Normal>::Ptr & normals)
            {
                surface_normals_ = normals;
            }

            void setPixelWiseRefinement(bool b)
            {
                pixelwise_refinement_ = b;
            }

            void setNyu(float n)
            {
                nyu_ = n;
            }

            void setLambda(float f)
            {
                lambda_ = f;
            }

            void setSigma(float f)
            {
                sigma_ = f;
            }

            void setAlpha(float f)
            {
                alpha_ = f;
            }

            void setVisEachMove(bool v)
            {
                vis_at_each_move_ = v;
            }

            void setMaxModelType(int mt)
            {
                MAX_MODEL_TYPE_ = mt;
            }

            void setUseSLIC(bool b)
            {
                use_SLIC_RGBD_ = b;
            }
    };
}

#endif
