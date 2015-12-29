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
NMBasedCloudIntegration<PointT>::



compute (const PointTPtr & output)
{
    input_clouds_used_.resize(input_clouds_.size());

    for(size_t i=0; i < input_clouds_.size(); i++)
        input_clouds_used_[i].reset(new pcl::PointCloud<PointT>(*input_clouds_[i]));

    //process clouds and weights to remove points based on distance and add weights based on noise
    float bad_value = std::numeric_limits<float>::quiet_NaN();
    float start_dist = 1.f;
    for(size_t i=0; i < input_clouds_used_.size(); i++)
    {
        for(size_t k=0; k < input_clouds_used_[i]->points.size(); k++)
        {
            if(!pcl_isfinite(input_clouds_used_[i]->points[k].z))
                continue;

            if(sigmas_combined_[i][k] > 1.f)
            {
                input_clouds_used_[i]->points[k].x = input_clouds_used_[i]->points[k].y = input_clouds_used_[i]->points[k].z = bad_value;
                noise_weights_[i][k] = 0.f;
            }

//            float dist = input_clouds_used_[i]->points[k].getVector3fMap().norm();

//            if(dist > param_.max_distance_)
//            {
//                input_clouds_used_[i]->points[k].x = input_clouds_used_[i]->points[k].y = input_clouds_used_[i]->points[k].z = bad_value;
//                noise_weights_[i][k] = 0.f;
//                continue;
//            }

//            if(noise_weights_[i][k] < param_.min_weight_)
//            {
//                input_clouds_used_[i]->points[k].x = input_clouds_used_[i]->points[k].y = input_clouds_used_[i]->points[k].z = bad_value;
//                noise_weights_[i][k] = 0.f;
//            }
//            else
//            {
//                //adapt weight based on distance
////                float capped_dist = std::min(std::max(start_dist, dist), nm_params_.max_distance_); //[start,end]
////                float w =  1.f - (capped_dist - start_dist) / (nm_params_.max_distance_ - start_dist);
//                //noise_weights_[i][k] *=  w;
//            }
        }
    }

    int width = input_clouds_[0]->width;
    int height = input_clouds_[0]->height;
    float cx_ = width / 2.f;
    float cy_ = height / 2.f;

    PointTPtr big_cloud(new pcl::PointCloud<PointT>);
    big_cloud_normals_.reset(new pcl::PointCloud<pcl::Normal>);
    big_cloud_weights_.clear();
    big_cloud_sigmas_.clear();
    big_cloud_origin_cloud_id_.clear();

    for(size_t i=0; i < input_clouds_used_.size(); i++)
    {
        PointTPtr cloud(new pcl::PointCloud<PointT>);
        PointNormalTPtr normal_cloud(new pcl::PointCloud<pcl::Normal>);
        transformNormals(*input_normals_[i], *normal_cloud, transformations_to_global_[i]);
        pcl::transformPointCloud(*input_clouds_used_[i], *cloud, transformations_to_global_[i]);

        /*float sum_curv = 0;
        float sum_curv_orig = 0;
        for(size_t k=0; k < normal_cloud->points.size(); k++)
        {
            sum_curv += normal_cloud->points[k].curvature;
            sum_curv_orig += input_normals_[i]->points[k].curvature;
        }

        std::cout << sum_curv << " " << sum_curv_orig << std::endl;*/

        if (indices_.empty())
        {
            *big_cloud += *cloud;
            *big_cloud_normals_ += *normal_cloud;

            big_cloud_weights_.insert(big_cloud_weights_.end(), sigmas_combined_[i].begin(), sigmas_combined_[i].end());
            big_cloud_sigmas_.insert(big_cloud_sigmas_.end(), sigmas_[i].begin(), sigmas_[i].end());

            std::vector<int> origin(cloud->points.size(), i);
            big_cloud_origin_cloud_id_.insert(big_cloud_origin_cloud_id_.end(), origin.begin(), origin.end());
        }
        else
        {
            pcl::copyPointCloud(*cloud, indices_[i], *cloud);
            *big_cloud += *cloud;

            pcl::copyPointCloud(*normal_cloud, indices_[i], *normal_cloud);
            *big_cloud_normals_ += *normal_cloud;

            for (size_t j=0;j<indices_[i].size();j++) {
                big_cloud_weights_.push_back(sigmas_combined_[i][indices_[i][j]]);
                big_cloud_sigmas_.push_back(sigmas_[i][indices_[i][j]]);
            }


            std::vector<int> origin(indices_[i].size(), i);
            big_cloud_origin_cloud_id_.insert(big_cloud_origin_cloud_id_.end(), origin.begin(), origin.end());

            std::cout << "input size: " << input_clouds_used_[i]->points.size() << std::endl
                      << "filtered size: " << cloud->points.size() << std::endl
                      << "filtered normal cloud size: " << normal_cloud->points.size() << std::endl
                      << "bigcloud size: " << big_cloud->points.size() << std::endl
                      << "noise weights size: " << noise_weights_[i].size() << std::endl
                      << "weights octree size: " << big_cloud_weights_.size() << std::endl
                      << std::endl;
        }
    }


//    // for each point of the accumulated cloud store information from which point of which input cloud it comes from
//    std::vector<std::pair<size_t, size_t> > big_cloud_to_input_clouds;
//    big_cloud_to_input_clouds.resize(big_cloud->points.size());
//    int idx = 0;
//    for(size_t i=0; i < input_clouds_used_.size(); i++)
//    {
//        if (indices_.empty())
//        {
//            for(size_t k=0; k < input_clouds_used_[i]->points.size(); k++, idx++)
//                big_cloud_to_input_clouds[idx] = std::make_pair(i, k);
//        }
//        else
//        {
//            std::cout << "input point size: " << input_clouds_used_[i]->points.size() << std::endl;
//            std::cout << "indices size: " << indices_[i].size() << std::endl;

//            for (size_t k=0; k<indices_[i].size(); k++, idx++)
//                big_cloud_to_input_clouds[idx] = std::make_pair(i, indices_[i][k]);
//        }
//    }


    // check each point of the big point cloud if there is a point from an organized point cloud which is in front or behind this point and has a better weight
//    std::vector<bool> indices_big_cloud_keep(big_cloud->points.size(), true);
//    for(size_t i=0; i < input_clouds_used_.size(); i++)
//    {
//        PointTPtr organized_cloud = input_clouds_used_[i];
//        Eigen::Matrix4f global_to_cloud = transformations_to_global_[i].inverse ();
//        int infront_or_behind = 0;
//        int rejected = 0;

//        for(size_t k=0; k < big_cloud->points.size(); k++)
//        {
//            if(!indices_big_cloud_keep[k]) //already rejected
//                continue;

//            Eigen::Vector4f pt_big_cloud = global_to_cloud * big_cloud->points[k].getVector4fMap();
//            int u = static_cast<int> (param_.focal_length_ * pt_big_cloud[0] / pt_big_cloud[2] + cx_);
//            int v = static_cast<int> (param_.focal_length_ * pt_big_cloud[1] / pt_big_cloud[2] + cy_);

//            //Not out of bounds or invalid depth
//            if ( (u >= width) || (v >= height) || (u < 0) || (v < 0) || !pcl::isFinite (organized_cloud->at (u, v)))
//              continue;

//            int idx_org = v * width + u;

//            Eigen::Vector3f normal_organized = input_normals_[i]->at (u,v).getNormalVector3fMap();
//            Eigen::Vector3f normal_from_big_cloud = big_cloud_normals_->points[k].getNormalVector3fMap();
//            Eigen::Vector3f normal_from_big_cloud_aligned;
//            transformNormal(normal_from_big_cloud, normal_from_big_cloud_aligned, global_to_cloud);

//            if(normal_from_big_cloud_aligned.dot(normal_organized) < 0)
//                continue;

//            float z_oc = organized_cloud->at (u, v).z;

//            //Check if point depth (distance to camera) is greater than the (u,v)
//            if(std::abs(pt_big_cloud[2] - z_oc) > param_.threshold_ss_) // point is in front or behind
//            {
//                infront_or_behind++;
//                if(big_cloud_weights_[k] < noise_weights_[i][idx_org])
//                {
//                    indices_big_cloud_keep[k] = false; //FIX THIS
//                    rejected++;
//                }
//            }
//        }
//    }

//    PointTPtr big_cloud_filtered (new pcl::PointCloud<PointT>);
//    std::vector<int> indices_kept(big_cloud->points.size(), 0);
//    size_t kept = 0;
//    for(size_t k=0; k < big_cloud->points.size(); k++)
//    {
//        if(indices_big_cloud_keep[k])
//        {
//            indices_kept[kept] = k;
//            kept++;
//        }
//    }
//    indices_kept.resize(kept);
//    pcl::copyPointCloud(*big_cloud, indices_kept, *big_cloud_filtered);
//    pcl::copyPointCloud(*big_cloud_normals_, indices_kept, *big_cloud_normals_);
//    std::vector<float> new_weights_(indices_kept.size());
//    for(size_t k=0; k < indices_kept.size(); k++)
//        new_weights_[k] = big_cloud_weights_[indices_kept[k]];

//    big_cloud_weights_ = new_weights_;

    octree_.reset(new pcl::octree::OctreePointCloudPointVector<PointT>(param_.octree_resolution_));
    octree_->setInputCloud(big_cloud);
    octree_->addPointsFromInputCloud();

    size_t leaf_node_counter = 0;
    typename pcl::octree::OctreePointCloudPointVector<PointT>::LeafNodeIterator leaf_it;
    const typename pcl::octree::OctreePointCloudPointVector<PointT>::LeafNodeIterator it2_end = octree_->leaf_end();

    output->points.resize(big_cloud_sigmas_.size());
    output_normals_.reset(new pcl::PointCloud<pcl::Normal>);
    output_normals_->points.resize(big_cloud_sigmas_.size());

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
        else if(param_.weighted_average_)
        {
            // take only point with max probability
            float max_prob = std::numeric_limits<float>::min();

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



//                if (prob>max_prob)
//                {
//                    p.getVector3fMap() = octree_->getInputCloud()->points[indexVector[k]].getVector3fMap();
//                    p.r = octree_->getInputCloud()->points[indexVector[k]].r;
//                    p.g = octree_->getInputCloud()->points[indexVector[k]].g;
//                    p.b = octree_->getInputCloud()->points[indexVector[k]].b;

//                    n.getNormalVector3fMap() = big_cloud_normals_->points[indexVector[k]].getNormalVector3fMap();
//                    n.curvature = big_cloud_normals_->points[indexVector[k]].curvature;
//                    used++;

//                    max_prob = prob;
//                }
            }

            std::sort(pts.begin(), pts.end());

            for(const auto &pt_tmp : pts)
            {
                if (pt_tmp.distance_to_depth_discontinuity > 0.003f)
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
        else
        {
            //take the max only
            float min_sigma = std::numeric_limits<float>::max();

            for(size_t k=0; k < indexVector.size(); k++)
            {
//                if(big_cloud_weights_[indexVector[k]] < param_.min_weight_)
//                    continue;

                if(big_cloud_weights_[indexVector[k]] > min_sigma)
                    continue;

                p.getVector3fMap() = octree_->getInputCloud()->points[indexVector[k]].getVector3fMap();
                p.r = octree_->getInputCloud()->points[indexVector[k]].r;
                p.g = octree_->getInputCloud()->points[indexVector[k]].g;
                p.b = octree_->getInputCloud()->points[indexVector[k]].b;

                n.getNormalVector3fMap() = big_cloud_normals_->points[indexVector[k]].getNormalVector3fMap();
                n.curvature = big_cloud_normals_->points[indexVector[k]].curvature;
                used++;

                min_sigma = big_cloud_weights_[indexVector[k]];
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
