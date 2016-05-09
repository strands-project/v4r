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

/**
*
*      @author Aitor Aldoma
*      @author Thomas Faeulhammer (faeulhammer@acin.tuwien.ac.at)
*      @date Feb, 2013
*      @brief object instance recognizer
*/

#include <v4r/common/miscellaneous.h>
#include <v4r/common/convertCloud.h>
#include <v4r/common/convertNormals.h>
#include <v4r/recognition/recognizer.h>
#include <v4r/recognition/hypotheses_verification.h>
#include <v4r/recognition/voxel_based_correspondence_estimation.h>
#include <v4r/segmentation/multiplane_segmentation.h>
#include <v4r/segmentation/ClusterNormalsToPlanesPCL.h>

#include <pcl/common/centroid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <sstream>

namespace v4r
{

template<typename PointT>
void
Recognizer<PointT>::hypothesisVerification ()
{
    verified_hypotheses_.clear();

    if(obj_hypotheses_.empty())
    {
        std::cout << "No generated models to verify!" << std::endl;
        return;
    }

    hv_algorithm_->setSceneCloud (scene_);
    hv_algorithm_->setNormals( scene_normals_ );
    hv_algorithm_->setHypotheses( obj_hypotheses_ );
    hv_algorithm_->verify ();
    verified_hypotheses_ = hv_algorithm_->getVerifiedHypotheses( );
}

template<typename PointT>
void
Recognizer<PointT>::visualize() const
{
    if(!vis_) {
        vis_.reset(new pcl::visualization::PCLVisualizer("single-view recognition results"));
        vis_->createViewPort(0,0,1,0.33,vp1_);
        vis_->createViewPort(0,0.33,1,0.66,vp2_);
        vis_->createViewPort(0,0.66,1,1,vp3_);
        vis_->addText("input cloud", 10, 10, 20, 1, 1, 1, "input", vp1_);
        vis_->addText("generated hypotheses", 10, 10, 20, 0, 0, 0, "generated hypotheses", vp2_);
        vis_->addText("verified hypotheses", 10, 10, 20, 0, 0, 0, "verified hypotheses", vp3_);
    }

    vis_->removeAllPointClouds();
    vis_->removeAllPointClouds(vp1_);
    vis_->removeAllPointClouds(vp2_);
    vis_->removeAllPointClouds(vp3_);

    typename pcl::PointCloud<PointT>::Ptr vis_cloud (new pcl::PointCloud<PointT>);
    pcl::copyPointCloud(*scene_, *vis_cloud);
    vis_cloud->sensor_origin_ = Eigen::Vector4f::Zero(4);
    vis_cloud->sensor_orientation_ = Eigen::Quaternionf::Identity();
    vis_->addPointCloud(vis_cloud, "input cloud", vp1_);
    vis_->setBackgroundColor(.0f, .0f, .0f, vp2_);

    for(size_t i=0; i<obj_hypotheses_.size(); i++)
    {
        for(size_t jj=0; jj<obj_hypotheses_[i].ohs_.size(); jj++)
        {
            const ObjectHypothesis<PointT> &oh = *obj_hypotheses_[i].ohs_[jj];
            ModelT &m = *oh.model_;
            const std::string model_id = m.id_.substr(0, m.id_.length() - 4);
            std::stringstream model_label;
            model_label << model_id << "_" << i;
            typename pcl::PointCloud<PointT>::Ptr model_aligned ( new pcl::PointCloud<PointT>() );
            typename pcl::PointCloud<PointT>::ConstPtr model_cloud = m.getAssembled( param_.resolution_mm_model_assembly_ );
            pcl::transformPointCloud( *model_cloud, *model_aligned, oh.transform_);
            vis_->addPointCloud(model_aligned, model_label.str(), vp2_);
        }
    }
    vis_->setBackgroundColor(.5f, .5f, .5f, vp2_);


    for(size_t i=0; i<verified_hypotheses_.size(); i++)
    {
        const ObjectHypothesis<PointT> &oh = *verified_hypotheses_[i];
        ModelT &m = *oh.model_;
        const std::string model_id = m.id_.substr(0, m.id_.length() - 4);
        std::stringstream model_label;
        model_label << model_id << "_v_" << i;
        typename pcl::PointCloud<PointT>::Ptr model_aligned ( new pcl::PointCloud<PointT>() );
        typename pcl::PointCloud<PointT>::ConstPtr model_cloud = m.getAssembled( param_.resolution_mm_model_assembly_ );
        pcl::transformPointCloud( *model_cloud, *model_aligned, oh.transform_);
        vis_->addPointCloud(model_aligned, model_label.str(), vp3_);
    }
    vis_->setBackgroundColor(1.f, 1.f, 1.f, vp3_);
    vis_->spin();
}


template<>
void
V4R_EXPORTS
Recognizer<pcl::PointXYZRGB>::visualize() const
{
    typedef pcl::PointXYZRGB PointT;

    if(!vis_) {
        vis_.reset(new pcl::visualization::PCLVisualizer("single-view recognition results"));
        vis_->createViewPort(0,0,1,0.33,vp1_);
        vis_->createViewPort(0,0.33,1,0.66,vp2_);
        vis_->createViewPort(0,0.66,1,1,vp3_);

        if(!param_.vis_for_paper_)
        {
            vis_->addText("input cloud", 10, 10, 20, 1, 1, 1, "input", vp1_);
            vis_->addText("generated hypotheses", 10, 10, 20, 0, 0, 0, "generated hypotheses", vp2_);
            vis_->addText("verified hypotheses", 10, 10, 20, 0, 0, 0, "verified hypotheses", vp3_);
        }
    }

    vis_->removeAllPointClouds();
    vis_->removeAllPointClouds(vp1_);
    vis_->removeAllPointClouds(vp2_);
    vis_->removeAllPointClouds(vp3_);

    Eigen::Vector4f zero_origin; zero_origin[0] = zero_origin[1] = zero_origin[2] = zero_origin[3] = 0.f;
    typename pcl::PointCloud<pcl::PointXYZRGB>::Ptr vis_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud(*scene_, *vis_cloud);
    vis_cloud->sensor_origin_ = zero_origin;
    vis_cloud->sensor_orientation_ = Eigen::Quaternionf::Identity();
    vis_->addPointCloud(vis_cloud, "input cloud", vp1_);

    if(param_.vis_for_paper_)
    {
        vis_->setBackgroundColor(1,1,1,vp1_);
#if PCL_VERSION >= 100702
        for(size_t co_id=0; co_id<coordinate_axis_ids_.size(); co_id++)
            vis_->removeCoordinateSystem( coordinate_axis_ids_[co_id] );
#endif
        coordinate_axis_ids_.clear();
    }
    else
        vis_->setBackgroundColor(.0f, .0f, .0f, vp1_);

    for(size_t i=0; i<obj_hypotheses_.size(); i++)
    {
        for(size_t jj=0; jj<obj_hypotheses_[i].ohs_.size(); jj++)
        {
            const ObjectHypothesis<PointT> &oh = *obj_hypotheses_[i].ohs_[jj];
            ModelT &m = *oh.model_;
            const std::string model_id = m.id_.substr(0, m.id_.length() - 4);
            std::stringstream model_label;
            model_label << model_id << "_" << i;
            typename pcl::PointCloud<PointT>::Ptr model_aligned ( new pcl::PointCloud<PointT>() );
            typename pcl::PointCloud<PointT>::ConstPtr model_cloud = m.getAssembled( param_.resolution_mm_model_assembly_ );
            pcl::transformPointCloud( *model_cloud, *model_aligned, oh.transform_);
            vis_->addPointCloud(model_aligned, model_label.str(), vp2_);

    #if PCL_VERSION >= 100702
            if(param_.vis_for_paper_)
            {
                Eigen::Matrix4f tf_tmp = oh.transform_;
                Eigen::Matrix3f rot_tmp  = tf_tmp.block<3,3>(0,0);
                Eigen::Vector3f trans_tmp = tf_tmp.block<3,1>(0,3);
                Eigen::Affine3f affine_trans;
                affine_trans.fromPositionOrientationScale(trans_tmp, rot_tmp, Eigen::Vector3f::Ones());
                std::stringstream co_id; co_id << i << "vp2";
                vis_->addCoordinateSystem(0.15f, affine_trans, co_id.str(), vp2_);
                coordinate_axis_ids_.push_back(co_id.str());
            }
    #endif
        }
    }
    if(param_.vis_for_paper_)
        vis_->setBackgroundColor(1,1,1,vp2_);
    else
        vis_->setBackgroundColor(.5f, .5f, .5f, vp2_);

    for(size_t i=0; i<verified_hypotheses_.size(); i++)
    {
        ObjectHypothesis<PointT> &oh = *verified_hypotheses_[i];
        ModelT &m = *oh.model_;
        const std::string model_id = m.id_.substr(0, m.id_.length() - 4);
        std::stringstream model_label;
        model_label << model_id << "_v_" << i;
        typename pcl::PointCloud<PointT>::Ptr model_aligned ( new pcl::PointCloud<PointT>() );
        typename pcl::PointCloud<PointT>::ConstPtr model_cloud = m.getAssembled( param_.resolution_mm_model_assembly_ );
        pcl::transformPointCloud( *model_cloud, *model_aligned, oh.transform_);
        vis_->addPointCloud(model_aligned, model_label.str(), vp3_);

#if PCL_VERSION >= 100702
        if(param_.vis_for_paper_)
        {
            Eigen::Matrix4f tf_tmp = oh.transform_;
            Eigen::Matrix3f rot_tmp  = tf_tmp.block<3,3>(0,0);
            Eigen::Vector3f trans_tmp = tf_tmp.block<3,1>(0,3);
            Eigen::Affine3f affine_trans;
            affine_trans.fromPositionOrientationScale(trans_tmp, rot_tmp, Eigen::Vector3f::Ones());
            std::stringstream co_id; co_id << i << "vp3";
            vis_->addCoordinateSystem(0.15f, affine_trans, co_id.str(), vp3_);
            coordinate_axis_ids_.push_back(co_id.str());
        }
#endif
    }

    vis_->setBackgroundColor(1.f, 1.f, 1.f, vp3_);
    vis_->resetCamera();
    vis_->spin();
}

//template<typename PointT>
//void
//Recognizer<PointT>::visualizePlanes() const
//{
//    Eigen::Vector4f zero_origin; zero_origin[0] = zero_origin[1] = zero_origin[2] = zero_origin[3] = 0.f;
//    for(size_t plane_id=0; plane_id < verified_planes_.size(); plane_id++)
//    {
//        std::stringstream plane_name;
//        plane_name << "plane_" << plane_id;
//        verified_planes_[plane_id]->sensor_origin_ = zero_origin;
//        verified_planes_[plane_id]->sensor_orientation_ = Eigen::Quaternionf::Identity();
//        vis_->addPointCloud<PointT> ( verified_planes_[plane_id], plane_name.str (), vp3_ );
//    }
//}

//template <>
//V4R_EXPORTS void
//Recognizer<pcl::PointXYZRGB>::visualizePlanes() const
//{
//    Eigen::Vector4f zero_origin; zero_origin[0] = zero_origin[1] = zero_origin[2] = zero_origin[3] = 0.f;
//    for(size_t plane_id=0; plane_id < verified_planes_.size(); plane_id++)
//    {
//        std::stringstream plane_name;
//        plane_name << "plane_" << plane_id;
//        verified_planes_[plane_id]->sensor_origin_ = zero_origin;
//        verified_planes_[plane_id]->sensor_orientation_ = Eigen::Quaternionf::Identity();
//        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_handler ( verified_planes_[plane_id] );
//        vis_->addPointCloud<pcl::PointXYZRGB> ( verified_planes_[plane_id], rgb_handler, plane_name.str (), vp3_ );
//    }
//}


template class V4R_EXPORTS Recognizer<pcl::PointXYZRGB>;
//template class V4R_EXPORTS Recognizer<pcl::PointXYZ>;
}


