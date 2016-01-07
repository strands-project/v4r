#include <v4r/common/miscellaneous.h>
#include <v4r/common/normals.h>
#include <v4r/io/eigen.h>
#include <v4r/recognition/local_recognizer.h>

#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <glog/logging.h>
#include <sstream>

namespace v4r
{

template<template<class > class Distance, typename PointT, typename FeatureT>
bool
LocalRecognitionPipeline<Distance, PointT, FeatureT>::loadFeaturesAndCreateFLANN ()
{
    std::vector<ModelTPtr> models = source_->getModels();
    flann_models_.clear();

    for (size_t i = 0; i < models.size (); i++)
    {
        ModelTPtr m = models[i];
        const std::string out_train_path = models_dir_  + "/" + m->class_ + "/" + m->id_ + "/" + descr_name_;
        const std::string in_train_path = models_dir_  + "/" + m->class_ + "/" + m->id_ + "/views/";

        for(size_t v_id=0; v_id< m->view_filenames_.size(); v_id++)
        {
            std::string signature_basename (m->view_filenames_[v_id]);
            boost::replace_last(signature_basename, "cloud_", "/descriptors_");

            typename pcl::PointCloud<FeatureT>::Ptr signature (new pcl::PointCloud<FeatureT> ());
            pcl::io::loadPCDFile (out_train_path + signature_basename, *signature);
            int size_feat = sizeof(signature->points[0].histogram) / sizeof(float);

            flann_model descr_model;
            descr_model.model = m;
            descr_model.view_id = m->view_filenames_[v_id];

            size_t kp_id_offset = 0;

            if (param_.use_cache_) //load model data (keypoints, pose and normals for each training view) and save them to cache
            {
                std::string pose_basename (m->view_filenames_[v_id]);
                boost::replace_last(pose_basename, "cloud_", "/pose_");
                boost::replace_last(pose_basename, ".pcd", ".txt");

                Eigen::Matrix4f pose_matrix = io::readMatrixFromFile( in_train_path + pose_basename);

                std::string keypoint_basename (m->view_filenames_[v_id]);
                boost::replace_last(keypoint_basename, "cloud_", + "/keypoints_");
                typename pcl::PointCloud<PointT>::Ptr keypoints (new pcl::PointCloud<PointT> ());
                pcl::io::loadPCDFile (out_train_path + keypoint_basename, *keypoints);

                std::string kp_normals_basename (m->view_filenames_[v_id]);
                boost::replace_last(kp_normals_basename, "cloud_", "/keypoint_normals_");
                pcl::PointCloud<pcl::Normal>::Ptr kp_normals (new pcl::PointCloud<pcl::Normal> ());
                pcl::io::loadPCDFile (out_train_path + kp_normals_basename, *kp_normals);

                for (size_t kp_id=0; kp_id<keypoints->points.size(); kp_id++)
                {
                    keypoints->points[ kp_id ].getVector4fMap () = pose_matrix * keypoints->points[ kp_id ].getVector4fMap ();
                    kp_normals->points[ kp_id ].getNormalVector3fMap () = pose_matrix.block<3,3>(0,0) * kp_normals->points[ kp_id ].getNormalVector3fMap ();
                }

                if( !m->keypoints_ )
                    m->keypoints_.reset(new pcl::PointCloud<PointT>());

                if ( !m->kp_normals_ )
                    m->kp_normals_.reset(new pcl::PointCloud<pcl::Normal>());

                kp_id_offset = m->keypoints_->points.size();

                m->keypoints_->points.insert(m->keypoints_->points.end(),
                                             keypoints->points.begin(),
                                             keypoints->points.end());

                m->kp_normals_->points.insert(m->kp_normals_->points.end(),
                                             kp_normals->points.begin(),
                                             kp_normals->points.end());
            }

            for (size_t dd = 0; dd < signature->points.size (); dd++)
            {
                descr_model.keypoint_id = kp_id_offset + dd;
                descr_model.descr.resize (size_feat);
                memcpy (&descr_model.descr[0], &signature->points[dd].histogram[0], size_feat * sizeof(float));
                flann_models_.push_back (descr_model);
            }
        }
    }

    specificLoadFeaturesAndCreateFLANN();
    std::cout << "Number of features:" << flann_models_.size () << std::endl;

    std::string filename;
    convertToFLANN<flann_model> (flann_models_, flann_data_);
    filename = models_dir_ + "/" + descr_name_ + "_flann.idx";

    if(io::existsFile(filename)) // Loading flann index from frile
    {
        try
        {
            flann_index_.reset( new flann::Index<DistT> (flann_data_, flann::SavedIndexParams (filename)));
        }
        catch(std::runtime_error &e)
        {
            std::cerr << "Existing flann index cannot be loaded. Removing file and creating a new flann file." << std::endl;
            boost::filesystem::remove(boost::filesystem::path(filename));
            return false;
        }
    }
    else // Building and saving flann index
    {
        flann_index_.reset( new flann::Index<DistT> (flann_data_, flann::KDTreeIndexParams (4)));
        flann_index_->buildIndex ();
        flann_index_->save (filename);
    }

    //once the descriptors in flann_models_ have benn converted to flann_data_, i can delete them
    for(size_t i=0; i < flann_models_.size(); i++)
        flann_models_[i].descr.clear();

    std::cout << "End load feature and create flann" << std::endl;
    return true;
}


template<template<class > class Distance, typename PointT, typename FeatureT>
bool
LocalRecognitionPipeline<Distance, PointT, FeatureT>::initialize (bool force_retrain)
{
    if(!estimator_)
    {
        std::cerr << "Keypoint and feature estimator is not set!" << std::endl;
        return false;
    }

    descr_name_ = estimator_->getFeatureDescriptorName();

    std::vector<ModelTPtr> models = source_->getModels();

    std::cout << "Models size:" << models.size () << std::endl;

    if (force_retrain)
    {
        for (size_t i = 0; i < models.size (); i++)
            source_->removeDescDirectory (*models[i], models_dir_, descr_name_);
    }

    for (size_t i = 0; i < models.size (); i++)
    {
        ModelTPtr &m = models[i];
        std::cout << m->class_ << " " << m->id_ << std::endl;
        const std::string dir = models_dir_ + "/" + m->class_ + "/" + m->id_ + "/" + descr_name_;

        if (!io::existsFolder(dir))
        {
            std::cout << "Model not trained..." << m->views_.size () << std::endl;
            if(!source_->getLoadIntoMemory())
            {
                try{
                    source_->loadInMemorySpecificModel(*m);
                }
                catch (std::runtime_error &e)
                {
                    std::cerr << "Load In Memory Specific Model failed. If this within a multi-pipeline recognizer, I will re-initialize now." << std::endl;
                    return false;
                }
            }


            for (size_t v = 0; v < m->view_filenames_.size(); v++)
            {
                typename pcl::PointCloud<FeatureT>::Ptr all_signatures (new pcl::PointCloud<FeatureT> ());
                typename pcl::PointCloud<FeatureT>::Ptr object_signatures (new pcl::PointCloud<FeatureT> ());
                typename pcl::PointCloud<PointT>::Ptr all_keypoints;
                typename pcl::PointCloud<PointT>::Ptr object_keypoints (new pcl::PointCloud<PointT>);
                typename pcl::PointCloud<PointT>::Ptr foo (new pcl::PointCloud<PointT>);
                pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

                computeNormals<PointT>(m->views_[v], normals, param_.normal_computation_method_);

                pcl::PointIndices all_kp_indices, obj_kp_indices;
                estimator_->setNormals(normals);
                bool success = estimator_->estimate (m->views_[v], foo, all_keypoints, all_signatures);
                (void) success;
                estimator_->getKeypointIndices(all_kp_indices);

                // remove signatures and keypoints which do not belong to object
                std::vector<bool> obj_mask = createMaskFromIndices(m->indices_[v].indices, m->views_[v]->points.size());
                obj_kp_indices.indices.resize( all_kp_indices.indices.size() );
                object_signatures->points.resize( all_kp_indices.indices.size() ) ;
                size_t kept=0;
                for (size_t kp_id = 0; kp_id < all_kp_indices.indices.size(); kp_id++)
                {
                    const int idx = all_kp_indices.indices[kp_id];
                    if ( obj_mask[idx] )
                    {
                        obj_kp_indices.indices[kept] = idx;
                        object_signatures->points[kept] = all_signatures->points[kp_id];
                        kept++;
                    }
                }
                object_signatures->points.resize( kept );
                obj_kp_indices.indices.resize( kept );
                pcl::copyPointCloud( *m->views_[v], obj_kp_indices, *object_keypoints);

                if (object_keypoints->points.size()) //save descriptors and keypoints to disk
                {
                    io::createDirIfNotExist(dir);
                    std::string descriptor_basename (m->view_filenames_[v]);
                    boost::replace_last(descriptor_basename, "cloud_", "/descriptors_");
                    pcl::io::savePCDFileBinary (dir + descriptor_basename, *object_signatures);

                    std::string keypoint_basename (m->view_filenames_[v]);
                    boost::replace_last(keypoint_basename, "cloud_", "/keypoints_");
                    pcl::io::savePCDFileBinary (dir + keypoint_basename, *object_keypoints);

                    std::string kp_normals_basename (m->view_filenames_[v]);
                    boost::replace_last(kp_normals_basename, "cloud_", "/keypoint_normals_");
                    pcl::PointCloud<pcl::Normal>::Ptr normals_keypoints(new pcl::PointCloud<pcl::Normal>);
                    pcl::copyPointCloud(*normals, obj_kp_indices, *normals_keypoints);
                    pcl::io::savePCDFileBinary (dir + kp_normals_basename, *normals_keypoints);
                }
            }

            if(!source_->getLoadIntoMemory())
                m->views_.clear();
        }
        else
        {
            std::cout << "Model already trained..." << std::endl;
            //there is no need to keep the views in memory once the model has been trained
            m->views_.clear();
        }
    }

    if (!loadFeaturesAndCreateFLANN ())
        return false;

    if(param_.icp_iterations_ > 0 && param_.icp_type_ == 1)
        source_->createVoxelGridAndDistanceTransform(param_.voxel_size_icp_);

    return true;
}

template<template<class > class Distance, typename PointT, typename FeatureT>
void
LocalRecognitionPipeline<Distance, PointT, FeatureT>::recognize ()
{
    models_.clear();
    transforms_.clear();
    scene_keypoints_.reset(new pcl::PointCloud<PointT>);

    if (feat_kp_set_from_outside_)
    {
        pcl::copyPointCloud(*scene_, scene_kp_indices_, *scene_keypoints_);
        LOG(INFO) << "Signatures and Keypoints set from outside ...";
        feat_kp_set_from_outside_ = false;
    }
    else
    {
        if(!estimator_)
            LOG(FATAL) << "No feature estimator set!";

        signatures_.reset(new pcl::PointCloud<FeatureT>);
        scene_kp_indices_.indices.clear();

        estimator_->setNormals(scene_normals_);
        typename pcl::PointCloud<PointT>::Ptr processed_foo;
        estimator_->estimate (scene_, processed_foo, scene_keypoints_, signatures_);

        estimator_->getKeypointIndices(scene_kp_indices_);
    }

    for(size_t i=0; i<scene_keypoints_->points.size(); i++)
    {
        if(!pcl::isFinite(scene_keypoints_->points[i]))
            throw std::runtime_error("Keypoint is not finite!");
    }

    if (scene_keypoints_->points.size() != signatures_->points.size())
        throw std::runtime_error("Size of keypoint cloud is not equal to number of signatures!");

    obj_hypotheses_.clear();

    int size_feat = sizeof(signatures_->points[0].histogram) / sizeof(float);

    flann::Matrix<float> distances (new float[param_.knn_], 1, param_.knn_);
    flann::Matrix<int> indices (new int[param_.knn_], 1, param_.knn_);
    flann::Matrix<float> p (new float[size_feat], 1, size_feat);

    for (size_t idx = 0; idx < signatures_->points.size (); idx++)
    {
        memcpy (&p.ptr ()[0], &signatures_->points[idx].histogram[0], size_feat * sizeof(float));
        nearestKSearch (flann_index_, p, param_.knn_, indices, distances);

        int dist = distances[0][0];
        if(dist > param_.max_descriptor_distance_)
            continue;

        std::vector<int> flann_models_indices(param_.knn_);
        std::vector<float> model_distances(param_.knn_);

        std::vector<PointT> corresponding_model_kps;
        std::vector<std::string> model_id_for_scene_keypoint;

        for (size_t i = 0; i < (size_t)param_.knn_; i++)
        {
            flann_models_indices[i] = indices[0][i];
            model_distances[i] = distances[0][i];
            const flann_model &f = flann_models_[flann_models_indices[i] ];
            PointT m_kp = getKeypoint (*f.model, f.keypoint_id, f.view_id);

            bool found = false; // check if a keypoint from same model and close distance already exists
            for(size_t kk=0; kk < corresponding_model_kps.size(); kk++)
            {
                const float m_kp_dist = (corresponding_model_kps[kk].getVector3fMap() - m_kp.getVector3fMap()).squaredNorm();
                if(model_id_for_scene_keypoint[kk].compare( f.model->id_ ) == 0 && m_kp_dist < param_.distance_same_keypoint_)
                {
                    found = true;
                    break;
                }
            }

            if(found)
                continue;

            corresponding_model_kps.push_back( m_kp );
            model_id_for_scene_keypoint.push_back( f.model->id_ );

            float m_dist = model_distances[i];

            typename symHyp::iterator it_map;
            if ((it_map = obj_hypotheses_.find (f.model->id_)) != obj_hypotheses_.end ())
            {
                ObjectHypothesis<PointT> &oh = it_map->second;

                pcl::Correspondence c ( (int)f.keypoint_id, scene_kp_indices_.indices[idx], m_dist);
                oh.model_scene_corresp_->push_back(c);
                oh.indices_to_flann_models_.push_back(flann_models_indices[i]);
            }
            else //create object hypothesis
            {
                ObjectHypothesis<PointT> oh;

                oh.model_ = f.model;
                oh.model_scene_corresp_->reserve (signatures_->points.size () * param_.knn_);
                oh.indices_to_flann_models_.reserve(signatures_->points.size () * param_.knn_);
                oh.model_scene_corresp_->push_back( pcl::Correspondence ((int)f.keypoint_id, scene_kp_indices_.indices[idx], m_dist) );
                oh.indices_to_flann_models_.push_back( flann_models_indices[i] );

                obj_hypotheses_[oh.model_->id_] = oh;
            }
        }
    }

    delete[] indices.ptr ();
    delete[] distances.ptr ();
    delete[] p.ptr ();

    typename symHyp::iterator it_map;
    for (it_map = obj_hypotheses_.begin(); it_map != obj_hypotheses_.end (); it_map++)
        it_map->second.model_scene_corresp_->shrink_to_fit();   // free memory

    if( param_.correspondence_distance_constant_weight_ != 1.f )
    {
        PCL_WARN("correspondence_distance_constant_weight_ activated! %f", param_.correspondence_distance_constant_weight_);
        //go through the object hypotheses and multiply the correspondences distances by the weight
        //this is done to favour correspondences from different pipelines that are more reliable than other (SIFT and SHOT corr. simultaneously fed into CG)

        for (it_map = obj_hypotheses_.begin (); it_map != obj_hypotheses_.end (); it_map++)
        {
            for(size_t k=0; k < (*it_map).second.model_scene_corresp_->size(); k++)
                it_map->second.model_scene_corresp_->at(k).distance *= param_.correspondence_distance_constant_weight_;
        }
    }

    if(cg_algorithm_ && !param_.save_hypotheses_)    // correspondence grouping is not done outside
    {
        correspondenceGrouping();

        //Prepare scene and model clouds for the pose refinement step
        if ( param_.icp_iterations_ > 0 || hv_algorithm_ )
            source_->voxelizeAllModels (param_.voxel_size_icp_);

        if ( param_.icp_iterations_ > 0)
            poseRefinement();

        if ( hv_algorithm_ && models_.size() )
            hypothesisVerification();
    }
    scene_normals_.reset();
}

template<template<class > class Distance, typename PointT, typename FeatureT>
void
LocalRecognitionPipeline<Distance, PointT, FeatureT>::getView (const ModelT & model, const std::string &view_id, typename pcl::PointCloud<PointT>::Ptr &view)
{
    view.reset (new pcl::PointCloud<PointT>);
    pcl::io::loadPCDFile (models_dir_ + "/" + model.class_ + "/" + model.id_ + "/" + descr_name_ + "/" + view_id, *view);
}

template<template<class > class Distance, typename PointT, typename FeatureT>
pcl::Normal
LocalRecognitionPipeline<Distance, PointT, FeatureT>::getKpNormal (const ModelT & model, size_t keypoint_id, const std::string &view_id)
{
    if (param_.use_cache_)
        return model.kp_normals_->points[keypoint_id];

    std::string kp_normals_basename (view_id);
    boost::replace_last(kp_normals_basename, "cloud_", descr_name_ + "/" + "keypoint_normals_");
    pcl::PointCloud<pcl::Normal> normals_cloud;
    pcl::io::loadPCDFile (kp_normals_basename, normals_cloud);

    std::string pose_basename (view_id);
    boost::replace_last(pose_basename, "cloud_", "pose_");
    boost::replace_last(pose_basename, ".pcd", ".txt");
    Eigen::Matrix4f pose_matrix = io::readMatrixFromFile( models_dir_ + "/" + model.class_ + "/" + model.id_ + "/" + pose_basename);

    pcl::Normal n;
    n.getNormalVector3fMap () = pose_matrix.block<3,3>(0,0) * normals_cloud.points[ keypoint_id ].getNormalVector3fMap ();
    return n;
}


template<template<class > class Distance, typename PointT, typename FeatureT>
PointT
LocalRecognitionPipeline<Distance, PointT, FeatureT>::getKeypoint (const ModelT & model, size_t keypoint_id, const std::string &view_id)
{
    if (param_.use_cache_)
        return model.keypoints_->points[keypoint_id];

    std::string keypoint_basename (view_id);
    boost::replace_last(keypoint_basename, "cloud_", descr_name_ + "/" + "keypoints_");
    pcl::PointCloud<PointT> keypoint_cloud;
    pcl::io::loadPCDFile (models_dir_ + "/" + model.class_ + "/" + model.id_ + "/" + keypoint_basename, keypoint_cloud);

    std::string pose_basename (view_id);
    boost::replace_last(pose_basename, "cloud_", "pose_");
    boost::replace_last(pose_basename, ".pcd", ".txt");
    Eigen::Matrix4f pose_matrix = io::readMatrixFromFile( models_dir_ + "/" + model.class_ + "/" + model.id_ + "/" + pose_basename);

    PointT kp;
    kp.getVector4fMap () = pose_matrix * keypoint_cloud[ keypoint_id ].getVector4fMap ();
    return kp;
}

template<template<class > class Distance, typename PointT, typename FeatureT>
void
LocalRecognitionPipeline<Distance, PointT, FeatureT>::correspondenceGrouping ()
{
    if(cg_algorithm_->getRequiresNormals() && (!scene_normals_ || scene_normals_->points.size() != scene_->points.size()))
        computeNormals<PointT>(scene_, scene_normals_, param_.normal_computation_method_);

    typename std::map<std::string, ObjectHypothesis<PointT> >::iterator it;
    for (it = obj_hypotheses_.begin (); it != obj_hypotheses_.end (); it++)
    {
        ObjectHypothesis<PointT> &oh = it->second;

        if(oh.model_scene_corresp_->size() < 3)
            continue;

        std::vector < pcl::Correspondences > corresp_clusters;
        cg_algorithm_->setSceneCloud (scene_);
        cg_algorithm_->setInputCloud (oh.model_->keypoints_);

        if(cg_algorithm_->getRequiresNormals())
            cg_algorithm_->setInputAndSceneNormals(oh.model_->kp_normals_, scene_normals_);

        //we need to pass the keypoints_pointcloud and the specific object hypothesis
        cg_algorithm_->setModelSceneCorrespondences (oh.model_scene_corresp_);
        cg_algorithm_->cluster (corresp_clusters);

        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > new_transforms (corresp_clusters.size());
        typename pcl::registration::TransformationEstimationSVD < PointT, PointT > t_est;

        for (size_t i = 0; i < corresp_clusters.size(); i++)
            t_est.estimateRigidTransformation (*oh.model_->keypoints_, *scene_, corresp_clusters[i], new_transforms[i]);

        if(param_.merge_close_hypotheses_) {
            std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > merged_transforms (corresp_clusters.size());
            std::vector<bool> cluster_has_been_taken(corresp_clusters.size(), false);
            const double angle_thresh_rad = param_.merge_close_hypotheses_angle_ * M_PI / 180.f ;

            size_t kept=0;
            for (size_t i = 0; i < new_transforms.size(); i++) {

                if (cluster_has_been_taken[i])
                    continue;

                cluster_has_been_taken[i] = true;
                const Eigen::Vector3f centroid1 = new_transforms[i].block<3, 1> (0, 3);
                const Eigen::Matrix3f rot1 = new_transforms[i].block<3, 3> (0, 0);

                pcl::Correspondences merged_corrs = corresp_clusters[i];

                for(size_t j=i; j < new_transforms.size(); j++) {
                    const Eigen::Vector3f centroid2 = new_transforms[j].block<3, 1> (0, 3);
                    const Eigen::Matrix3f rot2 = new_transforms[j].block<3, 3> (0, 0);
                    const Eigen::Matrix3f rot_diff = rot2 * rot1.transpose();

                    double rotx = atan2(rot_diff(2,1), rot_diff(2,2));
                    double roty = atan2(-rot_diff(2,0), sqrt(rot_diff(2,1) * rot_diff(2,1) + rot_diff(2,2) * rot_diff(2,2)));
                    double rotz = atan2(rot_diff(1,0), rot_diff(0,0));
                    double dist = (centroid1 - centroid2).norm();

                    if ( (dist < param_.merge_close_hypotheses_dist_) && (rotx < angle_thresh_rad) && (roty < angle_thresh_rad) && (rotz < angle_thresh_rad) ) {
                        merged_corrs.insert( merged_corrs.end(), corresp_clusters[j].begin(), corresp_clusters[j].end() );
                        cluster_has_been_taken[j] = true;
                    }
                }

                t_est.estimateRigidTransformation (*oh.model_->keypoints_, *scene_, merged_corrs, merged_transforms[kept]);
                kept++;
            }
            merged_transforms.resize(kept);
            new_transforms = merged_transforms;
        }

        std::cout << "Merged " << corresp_clusters.size() << " clusters into " << new_transforms.size() << " clusters. Total correspondences: " << oh.model_scene_corresp_->size () << " " << it->first << std::endl;


        //        oh.visualize(*scene_);

        size_t existing_hypotheses = models_.size();
        models_.resize( existing_hypotheses + new_transforms.size(), oh.model_  );
        transforms_.insert(transforms_.end(), new_transforms.begin(), new_transforms.end());
    }
}

}
