/******************************************************************************
 * Copyright (c) 2015 Thomas Faeulhammer
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


#ifndef V4R_OBJECT_MODELLING_MODELVIEW_H__
#define V4R_OBJECT_MODELLING_MODELVIEW_H__

#include <pcl/common/common.h>
#include <v4r/core/macros.h>
#include <v4r/keypoints/ClusterNormalsToPlanes.h>

namespace v4r
{
    namespace object_modelling
    {
        /**
        * @brief This class represents a training view of the object model used for learning
        * @author Thomas Faeulhammer
        * @date July 2015
        * */
        class V4R_EXPORTS modelView
        {
        public:
            class SuperPlane : public ClusterNormalsToPlanes::Plane
            {
            public:
                std::vector<size_t> visible_indices;
                std::vector<size_t> object_indices;
                std::vector<size_t> within_chop_z_indices;
                bool is_filtered;
                SuperPlane() : ClusterNormalsToPlanes::Plane()
                {
                    is_filtered = false;
                }
            };

        public:
            typedef pcl::PointXYZRGB PointT;
            typedef pcl::Histogram<128> FeatureT;

            pcl::PointCloud<PointT>::Ptr  cloud_;
            pcl::PointCloud<pcl::Normal>::Ptr  normal_;
            pcl::PointCloud<PointT>::Ptr  transferred_cluster_;

            std::vector<std::vector<float> >  sift_signatures_;
            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr  supervoxel_cloud_;
            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr  supervoxel_cloud_organized_;

            std::vector<SuperPlane> planes_;

            std::vector< size_t > scene_points_;
            std::vector< size_t > sift_keypoint_indices_;

            std::vector< std::vector <bool> > obj_mask_step_;
            Eigen::Matrix4f camera_pose_;
            Eigen::Matrix4f tracking_pose_;
            bool tracking_pose_set_ = false;
            bool camera_pose_set_ = false;

            size_t id_; // might be redundant

            bool is_pre_labelled_;

            modelView()
            {
                cloud_.reset(new pcl::PointCloud<PointT>());
                normal_.reset(new pcl::PointCloud<pcl::Normal>());
                transferred_cluster_.reset(new pcl::PointCloud<PointT>());
                supervoxel_cloud_.reset(new pcl::PointCloud<pcl::PointXYZRGBA>());
                supervoxel_cloud_organized_.reset(new pcl::PointCloud<pcl::PointXYZRGBA>());
                is_pre_labelled_ = false;
            }
        };
    }
}


#endif //V4R_OBJECT_MODELLING_MODELVIEW_H__
