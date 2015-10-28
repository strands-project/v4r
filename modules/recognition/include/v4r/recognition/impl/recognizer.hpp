/******************************************************************************
 * Copyright (c) 2013 Aitor Aldoma
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
#include <v4r/keypoints/ClusterNormalsToPlanes.h>
#include <v4r/recognition/recognizer.h>
#include <v4r/recognition/ghv.h>
#include <v4r/recognition/hypotheses_verification.h>
#include <v4r/segmentation/multiplane_segmentation.h>
#include <pcl/common/centroid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

namespace v4r
{

bool
gcGraphCorrespSorter (pcl::Correspondence i, pcl::Correspondence j)
{
    return (i.distance < j.distance);
}

template <typename PointT>
void
ObjectHypothesis<PointT>::visualize(const typename pcl::PointCloud<PointT> & scene) const
{
    (void)scene;
    std::cerr << "This function is not implemented for this point cloud type!" << std::endl;
}

template <>
void
ObjectHypothesis<pcl::PointXYZRGB>::visualize(const pcl::PointCloud<pcl::PointXYZRGB> & scene) const
{
    if(!vis_)
        vis_.reset(new pcl::visualization::PCLVisualizer("correspondences for hypothesis"));

    pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr model_cloud = model_->getAssembled( 0.003f );
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr model_aligned ( new pcl::PointCloud<pcl::PointXYZRGB>() );
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_vis ( new pcl::PointCloud<pcl::PointXYZRGB>() );
    Eigen::Vector4f zero_origin; zero_origin[0] = zero_origin[1] = zero_origin[2] = zero_origin[3] = 0.f;
    pcl::copyPointCloud( scene, *scene_vis);
    scene_vis->sensor_origin_ = zero_origin;
    scene_vis->sensor_orientation_ = Eigen::Quaternionf::Identity();
    pcl::copyPointCloud( *model_cloud, *model_aligned);
    vis_->addPointCloud(scene_vis, "scene");
    vis_->addPointCloud(model_aligned, "model_aligned");
    vis_->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal> (model_->keypoints_, model_->kp_normals_, 10, 0.05, "normals_model");

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr kp_colored_scene ( new pcl::PointCloud<pcl::PointXYZRGB>() );
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr kp_colored_model ( new pcl::PointCloud<pcl::PointXYZRGB>() );
    kp_colored_scene->points.resize(model_scene_corresp_->size());
    kp_colored_model->points.resize(model_scene_corresp_->size());

    pcl::CorrespondencesPtr sorted_corrs (new pcl::Correspondences (*model_scene_corresp_));
    std::sort (sorted_corrs->begin (), sorted_corrs->end (), gcGraphCorrespSorter);

    vis_->addPointCloud(kp_colored_scene, "kps_s");
    vis_->addPointCloud(kp_colored_model, "kps_m");

    for(size_t j=0; j<5; j++)
    {
        for (size_t i=(size_t)((j/5.f)*sorted_corrs->size()); i<(size_t)(((j+1.f)/5.f)*sorted_corrs->size()); i++)
        {
            const pcl::Correspondence &c = sorted_corrs->at(i);
            pcl::PointXYZRGB kp_m = model_->keypoints_->points[c.index_query];
            pcl::PointXYZRGB kp_s = scene.points[c.index_match];

            const float r = kp_m.r = kp_s.r = 100 + rand() % 155;
            const float g = kp_m.g = kp_s.g = 100 + rand() % 155;
            const float b = kp_m.b = kp_s.b = 100 + rand() % 155;
            kp_colored_scene->points[i] = kp_s;
            kp_colored_model->points[i] = kp_m;

            std::stringstream ss; ss << "correspondence " << i << j;
            vis_->addLine(kp_s, kp_m, r/255, g/255, b/255, ss.str());
            vis_->addSphere(kp_s, 2, r/255, g/255, b/255, ss.str() + "kp_s", vp1_);
            vis_->addSphere(kp_m, 2, r/255, g/255, b/255, ss.str() + "kp_m", vp1_);
        }
        vis_->spin();
    }

}

template<typename PointT>
void
Recognizer<PointT>::hypothesisVerification ()
{
    std::vector<typename pcl::PointCloud<PointT>::ConstPtr> aligned_models (models_.size ());
    std::vector<pcl::PointCloud<pcl::Normal>::ConstPtr> aligned_normals (models_.size ());

    model_or_plane_is_verified_.clear();
    planes_.clear();

    if(models_.empty())
    {
        std::cout << "No models to verify, returning... Cancelling service request." << std::endl;
        return;
    }

    for(size_t i=0; i<models_.size(); i++)
    {
        typename pcl::PointCloud<PointT>::Ptr aligned_model_tmp (new pcl::PointCloud<PointT>);
        pcl::PointCloud<pcl::Normal>::Ptr aligned_normal_tmp (new pcl::PointCloud<pcl::Normal>);
        ConstPointTPtr model_cloud = models_[i]->getAssembled (hv_algorithm_->getResolution());
        pcl::transformPointCloud (*model_cloud, *aligned_model_tmp, transforms_[i]);
        aligned_models[i] = aligned_model_tmp;
        pcl::PointCloud<pcl::Normal>::ConstPtr normal_cloud_const = models_[i]->getNormalsAssembled (hv_algorithm_->getResolution());
        transformNormals(*normal_cloud_const, *aligned_normal_tmp, transforms_[i]);
        aligned_normals[i] = aligned_normal_tmp;
    }


    boost::shared_ptr<GHV<PointT, PointT> > hv_algorithm_ghv;
    if( hv_algorithm_ )
        hv_algorithm_ghv = boost::dynamic_pointer_cast<GHV<PointT, PointT>> (hv_algorithm_);

    hv_algorithm_->setOcclusionCloud (scene_);
    hv_algorithm_->setSceneCloud (scene_);
    hv_algorithm_->addModels (aligned_models, true);

    if (hv_algorithm_->getRequiresNormals ())
        hv_algorithm_->addNormalsClouds (aligned_normals);

    if( hv_algorithm_ghv ) {
        hv_algorithm_ghv->setRequiresNormals(false);
        hv_algorithm_ghv->setNormalsForClutterTerm(scene_normals_);

        if( hv_algorithm_ghv->param_.add_planes_ ) {
            if( hv_algorithm_ghv->param_.plane_method_ == 0 ) {
                MultiPlaneSegmentation<PointT> mps;
                mps.setInputCloud( scene_ );
                mps.setMinPlaneInliers( hv_algorithm_ghv->param_.min_plane_inliers_ );
                mps.setResolution( hv_algorithm_ghv->param_.resolution_ );
                mps.setNormals( scene_normals_ );
                mps.setMergePlanes( true );
                mps.segment();
                planes_ = mps.getModels();
            }
            else {
                ClusterNormalsToPlanes::Parameter p_param;
                p_param.inlDistSmooth = 0.05f;
                p_param.minPoints = hv_algorithm_ghv->param_.min_plane_inliers_;
                ClusterNormalsToPlanes pest(p_param);
                DataMatrix2D<Eigen::Vector3f>::Ptr kp_cloud( new DataMatrix2D<Eigen::Vector3f>() );
                DataMatrix2D<Eigen::Vector3f>::Ptr kp_normals( new DataMatrix2D<Eigen::Vector3f>() );
                convertCloud(*scene_, *kp_cloud);
                convertNormals(*scene_normals_, *kp_normals);

                std::vector<ClusterNormalsToPlanes::Plane::Ptr> all_planes;
                pest.compute(*kp_cloud, *kp_normals, all_planes);
                planes_.reserve(all_planes.size());
                for(size_t p_id=0; p_id < all_planes.size(); p_id++) {
                    const ClusterNormalsToPlanes::Plane::Ptr &cpm = all_planes[p_id];
                    if(cpm->is_plane) {
                        PlaneModel<PointT> pm;
                        typename pcl::PointCloud<PointT>::Ptr plane (new pcl::PointCloud<PointT>);
                        pcl::copyPointCloud(*scene_, cpm->indices, *plane);
                        pm.cloud_ = scene_;
                        pm.plane_cloud_ = plane;
                        pm.coefficients_.values.resize( 4 );
                        pm.inliers_.indices = cpm->indices;

                        Eigen::Vector4f clust_centroid;
                        Eigen::Matrix3f clust_cov;
                        pcl::computeMeanAndCovarianceMatrix (*plane, clust_cov, clust_centroid);
                        Eigen::Vector4f plane_params;

                        EIGEN_ALIGN16 Eigen::Vector3f::Scalar eigen_value;
                        EIGEN_ALIGN16 Eigen::Vector3f eigen_vector;
                        pcl::eigen33 (clust_cov, eigen_value, eigen_vector);
                        plane_params[0] = eigen_vector[0];
                        plane_params[1] = eigen_vector[1];
                        plane_params[2] = eigen_vector[2];
                        plane_params[3] = 0;
                        plane_params[3] = -1 * plane_params.dot (clust_centroid);


                        const Eigen::Vector4f vp = -clust_centroid;
                        float cos_theta = vp.dot (plane_params);
                        if (cos_theta < 0) {
                            plane_params *= -1;
                            plane_params[3] = 0;
                            plane_params[3] = -1 * plane_params.dot (clust_centroid);
                        }

                        // Compute the curvature surface change
                        float curvature;
                        float eig_sum = clust_cov.coeff (0) + clust_cov.coeff (4) + clust_cov.coeff (8);
                        if (eig_sum != 0)
                            curvature = fabsf (eigen_value / eig_sum);
                        else
                            curvature = 0;

                        if (curvature < hv_algorithm_ghv->param_.curvature_threshold_)  {
                            pm.coefficients_.values[0] = plane_params[0];
                            pm.coefficients_.values[1] = plane_params[1];
                            pm.coefficients_.values[2] = plane_params[2];
                            pm.coefficients_.values[3] = plane_params[3];
                        }
                        planes_.push_back(pm);
                    }
                }
            }

            hv_algorithm_ghv->addPlanarModels(planes_);
        }

    }

    hv_algorithm_->verify ();
    hv_algorithm_->getMask (model_or_plane_is_verified_);
}


template<typename PointT>
void
Recognizer<PointT>::poseRefinement()
{
    PointTPtr scene_voxelized (new pcl::PointCloud<PointT> ());
    pcl::VoxelGrid<PointT> voxel_grid_icp;
    voxel_grid_icp.setInputCloud (scene_);

    if(icp_scene_indices_ && icp_scene_indices_->indices.size() > 0)
        voxel_grid_icp.setIndices(icp_scene_indices_);

    voxel_grid_icp.setLeafSize (param_.voxel_size_icp_, param_.voxel_size_icp_, param_.voxel_size_icp_);
    voxel_grid_icp.filter (*scene_voxelized);

    switch (param_.icp_type_)
    {
    case 0:
    {
#pragma omp parallel for schedule(dynamic,1) num_threads(omp_get_num_procs())
        for (size_t i = 0; i < models_.size (); i++)
        {
            ConstPointTPtr model_cloud = models_[i]->getAssembled (param_.voxel_size_icp_);
            PointTPtr model_aligned (new pcl::PointCloud<PointT>);
            pcl::transformPointCloud (*model_cloud, *model_aligned, transforms_[i]);

            typename pcl::registration::CorrespondenceRejectorSampleConsensus<PointT>::Ptr
                    rej (new pcl::registration::CorrespondenceRejectorSampleConsensus<PointT> ());

            rej->setInputTarget (scene_voxelized);
            rej->setMaximumIterations (1000);
            rej->setInlierThreshold (0.005f);
            rej->setInputSource (model_aligned);

            pcl::IterativeClosestPoint<PointT, PointT> reg;
            reg.addCorrespondenceRejector (rej);
            reg.setInputTarget (scene_voxelized);
            reg.setInputSource (model_aligned);
            reg.setMaximumIterations (param_.icp_iterations_);
            reg.setMaxCorrespondenceDistance (param_.max_corr_distance_);

            typename pcl::PointCloud<PointT>::Ptr output_ (new pcl::PointCloud<PointT> ());
            reg.align (*output_);

            Eigen::Matrix4f icp_trans = reg.getFinalTransformation ();
            transforms_[i] = icp_trans * transforms_[i];
        }
    }
        break;
    default:
    {
#pragma omp parallel for schedule(dynamic,1) num_threads(omp_get_num_procs())
        for (size_t i = 0; i < models_.size(); i++)
        {
            //        std::cout << "Doing ICP for model " << models_[i]->id_ << " (" << i << " / " << models_.size() << ")" << std::endl;
            typename VoxelBasedCorrespondenceEstimation<PointT, PointT>::Ptr
                    est (new VoxelBasedCorrespondenceEstimation<PointT, PointT> ());

            typename pcl::registration::CorrespondenceRejectorSampleConsensus<PointT>::Ptr
                    rej (new pcl::registration::CorrespondenceRejectorSampleConsensus<PointT> ());

            Eigen::Matrix4f scene_to_model_trans = transforms_[i].inverse ();
            boost::shared_ptr<distance_field::PropagationDistanceField<PointT> > dt;
            models_[i]->getVGDT (dt);

            PointTPtr model_aligned (new pcl::PointCloud<PointT>);
            PointTPtr scene_voxelized_icp_cropped (new pcl::PointCloud<PointT>);
            typename pcl::PointCloud<PointT>::ConstPtr cloud;
            dt->getInputCloud(cloud);
            model_aligned.reset(new pcl::PointCloud<PointT>(*cloud));

            pcl::transformPointCloud (*scene_voxelized, *scene_voxelized_icp_cropped, scene_to_model_trans);

            PointT minPoint, maxPoint;
            pcl::getMinMax3D(*cloud, minPoint, maxPoint);
            minPoint.x -= param_.max_corr_distance_;
            minPoint.y -= param_.max_corr_distance_;
            minPoint.z -= param_.max_corr_distance_;

            maxPoint.x += param_.max_corr_distance_;
            maxPoint.y += param_.max_corr_distance_;
            maxPoint.z += param_.max_corr_distance_;

            pcl::CropBox<PointT> cropFilter;
            cropFilter.setInputCloud (scene_voxelized_icp_cropped);
            cropFilter.setMin(minPoint.getVector4fMap());
            cropFilter.setMax(maxPoint.getVector4fMap());
            cropFilter.filter (*scene_voxelized_icp_cropped);

            est->setVoxelRepresentationTarget (dt);
            est->setInputSource (scene_voxelized_icp_cropped);
            est->setInputTarget (model_aligned);
            est->setMaxCorrespondenceDistance (param_.max_corr_distance_);

            rej->setInputTarget (model_aligned);
            rej->setMaximumIterations (1000);
            rej->setInlierThreshold (0.005f);
            rej->setInputSource (scene_voxelized_icp_cropped);

            pcl::IterativeClosestPoint<PointT, PointT, float> reg;
            reg.setCorrespondenceEstimation (est);
            reg.addCorrespondenceRejector (rej);
            reg.setInputTarget (model_aligned);
            reg.setInputSource (scene_voxelized_icp_cropped);
            reg.setMaximumIterations (param_.icp_iterations_);
            reg.setEuclideanFitnessEpsilon(1e-5);
            reg.setTransformationEpsilon(0.001f * 0.001f);

            pcl::registration::DefaultConvergenceCriteria<float>::Ptr convergence_criteria;
            convergence_criteria = reg.getConvergeCriteria();
            convergence_criteria->setAbsoluteMSE(1e-12);
            convergence_criteria->setMaximumIterationsSimilarTransforms(15);
            convergence_criteria->setFailureAfterMaximumIterations(false);

            typename pcl::PointCloud<PointT>::Ptr output_ (new pcl::PointCloud<PointT> ());
            reg.align (*output_);

            Eigen::Matrix4f icp_trans;
            icp_trans = reg.getFinalTransformation () * scene_to_model_trans;
            transforms_[i] = icp_trans.inverse ();

            //        std::cout << "Done ICP for model  " << models_[i]->id_ << " (" << i << " / " << models_.size() << ")" << std::endl;

        }
    }
    }
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

    Eigen::Vector4f zero_origin; zero_origin[0] = zero_origin[1] = zero_origin[2] = zero_origin[3] = 0.f;
    typename pcl::PointCloud<PointT>::Ptr vis_cloud (new pcl::PointCloud<PointT>);
    pcl::copyPointCloud(*scene_, *vis_cloud);
    vis_cloud->sensor_origin_ = zero_origin;
    vis_cloud->sensor_orientation_ = Eigen::Quaternionf::Identity();
    vis_->addPointCloud(vis_cloud, "input cloud", vp1_);
    vis_->setBackgroundColor(.0f, .0f, .0f, vp2_);

    for(size_t i=0; i<models_.size(); i++)
    {
        ModelT &m = *models_[i];
        const std::string model_id = m.id_.substr(0, m.id_.length() - 4);
        std::stringstream model_label;
        model_label << model_id << "_" << i;
        typename pcl::PointCloud<PointT>::Ptr model_aligned ( new pcl::PointCloud<PointT>() );
        typename pcl::PointCloud<PointT>::ConstPtr model_cloud = m.getAssembled( 0.003f );
        pcl::transformPointCloud( *model_cloud, *model_aligned, transforms_[i]);
        vis_->addPointCloud(model_aligned, model_label.str(), vp2_);
    }
    vis_->setBackgroundColor(.5f, .5f, .5f, vp2_);

    for(size_t i=0; i<models_.size(); i++)
    {
        if(!model_or_plane_is_verified_[i])
            continue;

        ModelT &m = *models_[i];
        const std::string model_id = m.id_.substr(0, m.id_.length() - 4);
        std::stringstream model_label;
        model_label << model_id << "_v_" << i;
        typename pcl::PointCloud<PointT>::Ptr model_aligned ( new pcl::PointCloud<PointT>() );
        typename pcl::PointCloud<PointT>::ConstPtr model_cloud = m.getAssembled( 0.003f );
        pcl::transformPointCloud( *model_cloud, *model_aligned, transforms_[i]);
        vis_->addPointCloud(model_aligned, model_label.str(), vp3_);

        PointT centroid;
        pcl::computeCentroid(*model_aligned, centroid);
        model_label << "_text3d";
        vis_->addText3D(model_label.str(), centroid, 0.015, 0, 0, 0, model_label.str(), vp3_);
    }

    for(size_t plane_id=0; plane_id < planes_.size(); plane_id++)
    {
        if(!model_or_plane_is_verified_[models_.size() + plane_id])
            continue;

        std::stringstream plane_name;
        plane_name << "plane_" << plane_id;
        planes_[plane_id].plane_cloud_->sensor_origin_ = zero_origin;
        planes_[plane_id].plane_cloud_->sensor_orientation_ = Eigen::Quaternionf::Identity();
        pcl::visualization::PointCloudColorHandlerRandom<PointT> plane_handler(planes_[plane_id].plane_cloud_);
        vis_->addPointCloud<PointT> ( planes_[plane_id].plane_cloud_, plane_handler, plane_name.str (), vp3_ );
    }
    vis_->setBackgroundColor(1.f, 1.f, 1.f, vp3_);
    vis_->spin();
}


template<>
V4R_EXPORTS void
Recognizer<pcl::PointXYZRGB>::visualize() const
{
    typedef pcl::PointXYZRGB PointT;

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

    Eigen::Vector4f zero_origin; zero_origin[0] = zero_origin[1] = zero_origin[2] = zero_origin[3] = 0.f;
    typename pcl::PointCloud<pcl::PointXYZRGB>::Ptr vis_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud(*scene_, *vis_cloud);
    vis_cloud->sensor_origin_ = zero_origin;
    vis_cloud->sensor_orientation_ = Eigen::Quaternionf::Identity();
    vis_->addPointCloud(vis_cloud, "input cloud", vp1_);
    vis_->setBackgroundColor(.0f, .0f, .0f, vp2_);

    for(size_t i=0; i<models_.size(); i++)
    {
        ModelT &m = *models_[i];
        const std::string model_id = m.id_.substr(0, m.id_.length() - 4);
        std::stringstream model_label;
        model_label << model_id << "_" << i;
        typename pcl::PointCloud<PointT>::Ptr model_aligned ( new pcl::PointCloud<PointT>() );
        typename pcl::PointCloud<PointT>::ConstPtr model_cloud = m.getAssembled( 0.003f );
        pcl::transformPointCloud( *model_cloud, *model_aligned, transforms_[i]);
        vis_->addPointCloud(model_aligned, model_label.str(), vp2_);
    }
    vis_->setBackgroundColor(.5f, .5f, .5f, vp2_);

    for(size_t i=0; i<models_.size(); i++)
    {
        if(!model_or_plane_is_verified_[i])
            continue;

        ModelT &m = *models_[i];
        const std::string model_id = m.id_.substr(0, m.id_.length() - 4);
        std::stringstream model_label;
        model_label << model_id << "_v_" << i;
        typename pcl::PointCloud<PointT>::Ptr model_aligned ( new pcl::PointCloud<PointT>() );
        typename pcl::PointCloud<PointT>::ConstPtr model_cloud = m.getAssembled( 0.003f );
        pcl::transformPointCloud( *model_cloud, *model_aligned, transforms_[i]);
        vis_->addPointCloud(model_aligned, model_label.str(), vp3_);

        PointT centroid;
        pcl::computeCentroid(*model_aligned, centroid);
        model_label << "_text3d";
        vis_->addText3D(model_label.str(), centroid, 0.015, 0, 0, 0, model_label.str(), vp3_);
    }

    for(size_t plane_id=0; plane_id < planes_.size(); plane_id++)
    {
        if(!model_or_plane_is_verified_[models_.size() + plane_id])
            continue;

        std::stringstream plane_name;
        plane_name << "plane_" << plane_id;
        planes_[plane_id].plane_cloud_->sensor_origin_ = zero_origin;
        planes_[plane_id].plane_cloud_->sensor_orientation_ = Eigen::Quaternionf::Identity();
        pcl::visualization::PointCloudColorHandlerRandom<PointT> plane_handler(planes_[plane_id].plane_cloud_);
        vis_->addPointCloud<PointT> ( planes_[plane_id].plane_cloud_, plane_handler, plane_name.str (), vp3_ );
    }

    vis_->setBackgroundColor(1.f, 1.f, 1.f, vp3_);
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


}
