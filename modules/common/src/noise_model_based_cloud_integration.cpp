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

#include <pcl/common/angles.h>
#include <pcl/common/io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>

#include <v4r/common/noise_model_based_cloud_integration.h>
#include <v4r/common/organized_edge_detection.h>

#include <glog/logging.h>

namespace v4r
{
template<typename PointT>
NMBasedCloudIntegration<PointT>::NMBasedCloudIntegration(const Parameter &p) : param_(p)
{

}

template<typename PointT>
void
NMBasedCloudIntegration<PointT>::compute (const PointTPtr & output)
{
    if(input_clouds_.empty()) {
        std::cerr << "No input clouds set for cloud integration!" << std::endl;
        return;
    }

    std::vector<PointInfo> big_cloud_info;

    PointTPtr big_cloud(new pcl::PointCloud<PointT>);
    big_cloud_normals_.reset(new pcl::PointCloud<pcl::Normal>);
    big_cloud_sigmas_.clear();
    big_cloud_origin_cloud_id_.clear();

    for(size_t i=0; i < input_clouds_.size(); i++)
    {
        pcl::PointCloud<PointT> cloud_aligned;
        pcl::transformPointCloud(*input_clouds_[i], cloud_aligned, transformations_to_global_[i]);
        pcl::PointCloud<pcl::Normal> normals_aligned;
        transformNormals(*input_normals_[i], normals_aligned, transformations_to_global_[i]);

        size_t existing_pts = big_cloud_info.size();

        if (indices_.empty())
        {
            *big_cloud += cloud_aligned;

            big_cloud_info.resize( existing_pts + cloud_aligned->points.size());

            *big_cloud_normals_ += normals_aligned;

//            big_cloud_weights_.insert(big_cloud_weights_.end(), sigmas_combined_[i].begin(), sigmas_combined_[i].end());
            big_cloud_sigmas_.insert(big_cloud_sigmas_.end(), sigmas_[i].begin(), sigmas_[i].end());

            std::vector<int> origin(cloud_aligned.points.size(), i);
            big_cloud_origin_cloud_id_.insert(big_cloud_origin_cloud_id_.end(), origin.begin(), origin.end());
        }
        else
        {
            pcl::copyPointCloud(cloud_aligned, indices_[i], cloud_aligned);
            *big_cloud += cloud_aligned;

            pcl::copyPointCloud(normals_aligned, indices_[i], normals_aligned);
            *big_cloud_normals_ += normals_aligned;

            for (size_t j=0;j<indices_[i].size();j++)
                big_cloud_sigmas_.push_back( sigmas_[i][indices_[i][j]]);

            std::vector<int> origin(indices_[i].size(), i);
            big_cloud_origin_cloud_id_.insert(big_cloud_origin_cloud_id_.end(), origin.begin(), origin.end());
        }
    }

    std::vector< std::vector<size_t> > pt_property (big_cloud->points.size());
               // for each point store the number of viewpoints in which the point
                   //  [0] is occluded;
                   //  [1] can be explained by a nearby point;
                   //  [2] could be seen but does not make sense (i.e. the view-ray is not blocked, but there is no sensed point)


    octree_.reset(new pcl::octree::OctreePointCloudPointVector<PointT>( param_.octree_resolution_ ) );
    octree_->setInputCloud( big_cloud );
    octree_->addPointsFromInputCloud();

    size_t leaf_node_counter = 0;
    typename pcl::octree::OctreePointCloudPointVector<PointT>::LeafNodeIterator leaf_it;
    const typename pcl::octree::OctreePointCloudPointVector<PointT>::LeafNodeIterator it2_end = octree_->leaf_end();

    output->points.resize( big_cloud->points.size() );
    output_normals_.reset(new pcl::PointCloud<pcl::Normal>);
    output_normals_->points.resize( big_cloud->points.size());

    size_t kept = 0;
    size_t total_used = 0;

    for (leaf_it = octree_->leaf_begin(); leaf_it != it2_end; ++leaf_it)
    {
        ++leaf_node_counter;
        pcl::octree::OctreeContainerPointIndices& container = leaf_it.getLeafContainer();

        // add points from leaf node to indexVector
        std::vector<int> indexVector;
        container.getPointIndices (indexVector);

        if(indexVector.size() < param_.min_points_per_voxel_)
            continue;

        if( param_.final_resolution_ < 0.f)
        {
            for(size_t k=0; k < indexVector.size(); k++)
            {
                output->points[kept] = octree_->getInputCloud()->points[indexVector[k]];
                kept++;
            }
            continue;
        }

        PointT p;
        p.getVector3fMap() = Eigen::Vector3f::Zero();
        int r, g, b;
        r = g = b = 0;
        pcl::Normal n;
        n.getNormalVector3fMap() = Eigen::Vector3f::Zero();
        n.curvature = 0.f;
        size_t used = 0;

        if(param_.average_)
        {
            for(size_t k=0; k < indexVector.size(); k++)
            {
//                if(big_cloud_weights_[indexVector[k]] < param_.min_weight_)
//                    continue;

                p.getVector3fMap() = p.getVector3fMap() +  octree_->getInputCloud()->points[indexVector[k]].getVector3fMap();
                r += octree_->getInputCloud()->points[indexVector[k]].r;
                g += octree_->getInputCloud()->points[indexVector[k]].g;
                b += octree_->getInputCloud()->points[indexVector[k]].b;

                Eigen::Vector3f normal = big_cloud_normals_->points[indexVector[k]].getNormalVector3fMap();
                normal.normalize();
                n.getNormalVector3fMap() = n.getNormalVector3fMap() + normal;
                n.curvature += big_cloud_normals_->points[indexVector[k]].curvature;
                used++;
            }

            if( used == 0 )
                continue;

            total_used += used;

            //std::cout << "n.curvature" << n.curvature << std::endl;

            p.getVector3fMap() = p.getVector3fMap() / used;
            p.r = r / used;
            p.g = g / used;
            p.b = b / used;

            n.getNormalVector3fMap() = n.getNormalVector3fMap() / used;
            n.getNormalVector3fMap()[3] = 0;
            n.curvature /= used;
        }
        else // take only point with max probability
        {
            std::vector<PointInfo> pts (indexVector.size());

            for(size_t k=0; k < indexVector.size(); k++)
            {
                int origin = big_cloud_origin_cloud_id_ [ indexVector[k] ];
                const Eigen::Matrix4f &tf = transformations_to_global_[ origin ];

                Eigen::Matrix3f sigma = Eigen::Matrix3f::Zero(), sigma_aligned = Eigen::Matrix3f::Zero();
                sigma(0,0) = big_cloud_sigmas_[ indexVector[k] ][0];  //lateral
                sigma(1,1) = big_cloud_sigmas_[ indexVector[k] ][0];  //lateral
                sigma(2,2) = big_cloud_sigmas_[ indexVector[k] ][1];  //axial

                Eigen::Matrix3f rotation = tf.block<3,3>(0,0); // or inverse?
                sigma_aligned = rotation * sigma * rotation.transpose();

                pts[k].probability = 1/ sqrt(2 * M_PI * sigma_aligned.determinant());
                pts[k].index_in_big_cloud = indexVector[k];
                pts[k].distance_to_depth_discontinuity = big_cloud_sigmas_[ indexVector[k] ][2];
            }

            std::sort(pts.begin(), pts.end());

            for(const auto &pt_tmp : pts)
            {
                if (pt_tmp.distance_to_depth_discontinuity > 2.0f)
                {
                    const PointT &pt = octree_->getInputCloud()->points[ pt_tmp.index_in_big_cloud ];
                    p.getVector3fMap() = pt.getVector3fMap();
                    p.r = pt.r;
                    p.g = pt.g;
                    p.b = pt.b;

                    n.getNormalVector3fMap() = big_cloud_normals_->points[ pt_tmp.index_in_big_cloud ].getNormalVector3fMap();
                    n.curvature = big_cloud_normals_->points[ pt_tmp.index_in_big_cloud ].curvature;
                    used++;
                    break;
                }
            }

            if(used == 0)
                continue;

            total_used++;
        }

        output->points[kept] = p;
        output_normals_->points[kept] = n;
        kept++;
    }

    std::cout << "Number of points in final model:" << kept << " used:" << total_used << std::endl;

    output->points.resize(kept);
    output_normals_->points.resize(kept);
    output->width = output_normals_->width = kept;
    output->height = output_normals_->height = 1;
    output->is_dense = output_normals_->is_dense = true;

    //finally mask input_clouds_ to know which points were actually used
    //FIX This, was commented out before?
//    for(size_t k=0; k < big_cloud->points.size(); k++)
//    {
//        if(!indices_big_cloud_keep[k])
//        {
//            size_t idx_c = big_cloud_to_input_clouds[k].first;
//            size_t idx_p = big_cloud_to_input_clouds[k].second;
//            input_clouds_used_[idx_c]->points[idx_p].x =
//            input_clouds_used_[idx_c]->points[idx_p].y =
//            input_clouds_used_[idx_c]->points[idx_p].z = bad_value;
//        }
//    }
}

template class V4R_EXPORTS NMBasedCloudIntegration<pcl::PointXYZRGB>;
//template class noise_models::NguyenNoiseModel<pcl::PointXYZ>;
}
