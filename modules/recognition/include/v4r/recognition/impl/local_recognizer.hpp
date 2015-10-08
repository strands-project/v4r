#include <v4r/recognition/local_recognizer.h>
#include <v4r/io/eigen.h>
#include <v4r/common/miscellaneous.h>
#include <sstream>

#include <pcl/visualization/pcl_visualizer.h>

template<template<class > class Distance, typename PointT, typename FeatureT>
void
v4r::LocalRecognitionPipeline<Distance, PointT, FeatureT>::loadFeaturesAndCreateFLANN ()
{
    std::vector<ModelTPtr> models = source_->getModels();

    for (size_t i = 0; i < models.size (); i++)
    {
        ModelTPtr m = models[i];
        const std::string out_train_path = training_dir_  + "/" + m->class_ + "/" + m->id_ + "/" + descr_name_;
        const std::string in_train_path = training_dir_  + "/" + m->class_ + "/" + m->id_ + "/";
        const std::string descriptor_pattern = ".*descriptors.*.pcd";

        std::vector<std::string> desc_files;
        v4r::io::getFilesInDirectory(out_train_path, desc_files, "", descriptor_pattern, false);
        std::sort(desc_files.begin(), desc_files.end());

        for(size_t v_id=0; v_id<desc_files.size(); v_id++)
        {
            const std::string signature_file_name = out_train_path + "/" + desc_files[v_id];
            typename pcl::PointCloud<FeatureT>::Ptr signature (new pcl::PointCloud<FeatureT> ());
            pcl::io::loadPCDFile (signature_file_name, *signature);
            int size_feat = sizeof(signature->points[0].histogram) / sizeof(float);

            std::string view_name = desc_files[v_id];
            boost::replace_last(view_name, ".pcd", "");
            std::vector < std::string > strs;
            boost::split (strs, view_name, boost::is_any_of ("_"));
            view_id_length_ = strs[1].size();

            flann_model descr_model;
            descr_model.model = m;

            std::istringstream iss(strs[1]);
            iss >> descr_model.view_id;

            size_t kp_id_offset = 0;

            if (param_.use_cache_) //load model data (keypoints, pose and normals for each training view) and save them to cache
            {
                std::stringstream pose_fn;
                pose_fn << in_train_path << "/pose_" << setfill('0') << setw(view_id_length_) << descr_model.view_id << ".txt";
                Eigen::Matrix4f pose_matrix;
                v4r::io::readMatrixFromFile( pose_fn.str (), pose_matrix);

                std::stringstream dir_keypoints; dir_keypoints << out_train_path << "/keypoints_" << setfill('0') << setw(view_id_length_) << descr_model.view_id << ".pcd";
                typename pcl::PointCloud<PointT>::Ptr keypoints (new pcl::PointCloud<PointT> ());
                pcl::io::loadPCDFile (dir_keypoints.str (), *keypoints);

                std::stringstream dir_normals; dir_normals << out_train_path << "/keypoint_normals_" << setfill('0') << setw(view_id_length_) << descr_model.view_id << ".pcd";
                pcl::PointCloud<pcl::Normal>::Ptr kp_normals (new pcl::PointCloud<pcl::Normal> ());
                pcl::io::loadPCDFile (dir_normals.str (), *kp_normals);

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
    filename = training_dir_ + "/" + descr_name_ + "_flann.idx";

    if(v4r::io::existsFile(filename)) // Loading flann index from frile
    {
        try
        {
            flann_index_.reset( new flann::Index<DistT> (flann_data_, flann::SavedIndexParams (filename)));
        }
        catch(std::runtime_error &e)
        {
            std::cerr << "Existing flann index cannot be loaded. Removing file and creating a new flann file." << std::endl;
            boost::filesystem::remove(boost::filesystem::path(filename));
            flann_index_.reset( new flann::Index<DistT> (flann_data_, flann::KDTreeIndexParams (4)));
            flann_index_->buildIndex ();
            flann_index_->save (filename);
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
}

template<template<class > class Distance, typename PointT, typename FeatureT>
void
v4r::LocalRecognitionPipeline<Distance, PointT, FeatureT>::nearestKSearch (boost::shared_ptr<flann::Index<DistT> > &index,
                                                                             /*float * descr, int descr_size*/
                                                                             flann::Matrix<float> & p, int k,
                                                                             flann::Matrix<int> &indices,
                                                                             flann::Matrix<float> &distances)
{
    index->knnSearch (p, indices, distances, k, flann::SearchParams (param_.kdtree_splits_));
}

template<template<class > class Distance, typename PointT, typename FeatureT>
void
v4r::LocalRecognitionPipeline<Distance, PointT, FeatureT>::reinitialize(const std::vector<std::string> & load_ids)
{
    PCL_WARN("Reinitialize LocalRecognitionPipeline with list of load_ids\n");
    std::cout << "List of models being loaded:" << load_ids.size() << std::endl;

    for(size_t i=0; i < load_ids.size(); i++)
        std::cout << " ---------- " << load_ids[i] << std::endl;

    flann_models_.clear();

    source_->setModelList(load_ids);
    source_->generate(training_dir_);

    initialize(false);
}

template<template<class > class Distance, typename PointT, typename FeatureT>
void
v4r::LocalRecognitionPipeline<Distance, PointT, FeatureT>::initialize (bool force_retrain)
{
    if(!estimator_)
    {
        std::cerr << "Keypoint and feature estimator is not set!" << std::endl;
        return;
    }

    descr_name_ = estimator_->getFeatureDescriptorName();

    std::vector<ModelTPtr> models = source_->getModels();

    std::cout << "Models size:" << models.size () << std::endl;

    if (force_retrain)
    {
        for (size_t i = 0; i < models.size (); i++)
            source_->removeDescDirectory (*models[i], training_dir_, descr_name_);
    }

    for (size_t i = 0; i < models.size (); i++)
    {
        ModelTPtr &m = models[i];
        std::cout << m->class_ << " " << m->id_ << std::endl;
        const std::string dir = training_dir_ + "/" + m->class_ + "/" + m->id_ + "/" + descr_name_;

        if (!v4r::io::existsFolder(dir))
        {
            std::cout << "Model not trained..." << m->views_.size () << std::endl;
            if(!source_->getLoadIntoMemory())
                source_->loadInMemorySpecificModel(training_dir_, *m);

            for (size_t v = 0; v < m->view_filenames_.size(); v++)
            {
                typename pcl::PointCloud<FeatureT>::Ptr all_signatures (new pcl::PointCloud<FeatureT> ());
                typename pcl::PointCloud<FeatureT>::Ptr object_signatures (new pcl::PointCloud<FeatureT> ());
                typename pcl::PointCloud<PointT>::Ptr all_keypoints;
                typename pcl::PointCloud<PointT>::Ptr object_keypoints (new pcl::PointCloud<PointT>);
                typename pcl::PointCloud<PointT>::Ptr foo (new pcl::PointCloud<PointT>);

                std::vector<std::string> strs;
                boost::split (strs, m->view_filenames_[v], boost::is_any_of ("_"));
                boost::replace_last(strs[1], ".pcd", "");
                view_id_length_ = strs[1].size();

                pcl::PointIndices all_kp_indices, obj_kp_indices;
                bool success = estimator_->estimate (m->views_[v], foo, all_keypoints, all_signatures);
                (void) success;
                estimator_->getKeypointIndices(all_kp_indices);

                // remove signatures and keypoints which do not belong to object
                std::vector<bool> obj_mask = v4r::createMaskFromIndices(m->indices_[v].indices, m->views_[v]->points.size());
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

                if (object_keypoints->points.size())
                {
                    v4r::io::createDirIfNotExist(dir);

                    //save keypoints and descriptors to disk
                    std::stringstream kp_fn; kp_fn << dir << "/keypoints_" << setfill('0') << setw(view_id_length_) << v << ".pcd";
                    pcl::io::savePCDFileBinary (kp_fn.str (), *object_keypoints);

                    std::stringstream desc_fn; desc_fn << dir << "/descriptors_" << setfill('0') << setw(view_id_length_) << v << ".pcd";
                    pcl::io::savePCDFileBinary (desc_fn.str (), *object_signatures);

                    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
                    pcl::PointCloud<pcl::Normal>::Ptr normals_keypoints(new pcl::PointCloud<pcl::Normal>);
                    v4r::computeNormals<PointT>(m->views_[v], normals, param_.normal_computation_method_);
                    pcl::copyPointCloud(*normals, obj_kp_indices, *normals_keypoints);
                    std::stringstream normals_fn; normals_fn << dir << "/keypoint_normals_" << setfill('0') << setw(view_id_length_) << v << ".pcd";
                    pcl::io::savePCDFileBinary (normals_fn.str (), *normals_keypoints);
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

    loadFeaturesAndCreateFLANN ();

    if(param_.icp_iterations_ > 0 && param_.icp_type_ == 1)
        source_->createVoxelGridAndDistanceTransform(param_.voxel_size_icp_);
}

template<template<class > class Distance, typename PointT, typename FeatureT>
void
v4r::LocalRecognitionPipeline<Distance, PointT, FeatureT>::recognize ()
{
    models_.clear();
    transforms_.clear();

    if (feat_kp_set_from_outside_)
    {
        pcl::copyPointCloud(*scene_, scene_kp_indices_, *scene_keypoints_);
        std::cout << "Signatures and Keypoints set from outside ..." << std::endl;
        feat_kp_set_from_outside_ = false;
    }
    else
    {
        signatures_.reset(new pcl::PointCloud<FeatureT>);
        scene_keypoints_.reset(new pcl::PointCloud<PointT>);
        scene_kp_indices_.indices.clear();

        typename pcl::PointCloud<PointT>::Ptr processed(new pcl::PointCloud<PointT>);
        if (indices_.size () > 0)
        {
            if(estimator_->acceptsIndices())
            {
                estimator_->setIndices(indices_);
                estimator_->estimate (scene_, processed, scene_keypoints_, signatures_);
            }
            else
            {
                PointTPtr sub_input (new pcl::PointCloud<PointT>);
                pcl::copyPointCloud (*scene_, indices_, *sub_input);
                estimator_->estimate (sub_input, processed, scene_keypoints_, signatures_);
            }
        }
        else
            estimator_->estimate (scene_, processed, scene_keypoints_, signatures_);

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

            typename std::map<std::string, ObjectHypothesis<PointT> >::iterator it_map;
            if ((it_map = obj_hypotheses_.find (f.model->id_)) != obj_hypotheses_.end ())
            {
                ObjectHypothesis<PointT> &oh = it_map->second;

                pcl::Correspondence c ( (int)f.keypoint_id, scene_kp_indices_.indices[idx], m_dist);
                oh.model_scene_corresp_->push_back(c);
                oh.indices_to_flann_models_.push_back(flann_models_indices[i]);
                assert(oh.indices_to_flann_models_.size() == oh.model_scene_corresp_->size());
                //            (*it_map).second.num_corr_++;
            }
            else //create object hypothesis
            {
                ObjectHypothesis<PointT> oh;
                oh.scene_ = scene_;
                oh.model_scene_corresp_.reset (new pcl::Correspondences ());

                oh.model_scene_corresp_->resize (1);
                oh.indices_to_flann_models_.resize(1);

                oh.model_scene_corresp_->reserve (signatures_->points.size () * param_.knn_);
                oh.indices_to_flann_models_.reserve(signatures_->points.size () * param_.knn_);

                oh.model_scene_corresp_->at (0) = pcl::Correspondence ((int)f.keypoint_id, scene_kp_indices_.indices[idx], m_dist);
                oh.indices_to_flann_models_[0] = flann_models_indices[i];
                oh.model_ = f.model;

                assert(oh.indices_to_flann_models_.size() == oh.model_scene_corresp_->size());
                obj_hypotheses_[oh.model_->id_] = oh;
            }
        }
    }

    delete[] indices.ptr ();
    delete[] distances.ptr ();
    delete[] p.ptr ();

    typename std::map<std::string, ObjectHypothesis<PointT> >::iterator it_map;
    for (it_map = obj_hypotheses_.begin(); it_map != obj_hypotheses_.end (); it_map++)
    {   // WHAT? Is this because of the .reserve?
        ObjectHypothesis<PointT> &oh = it_map->second;
        size_t num_corr = oh.model_scene_corresp_->size();
        oh.model_scene_corresp_->resize(num_corr);
    }

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
        throw std::runtime_error("This has not been implemented properly!");

        if(!scene_normals_ || scene_normals_->points.size() != scene_->points.size())
            v4r::computeNormals<PointT>(scene_, scene_normals_, param_.normal_computation_method_);

        prepareSpecificCG(scene_, scene_keypoints_);

        for (it_map = obj_hypotheses_.begin (); it_map != obj_hypotheses_.end (); it_map++)
        {
            ObjectHypothesis<PointT> &oh = it_map->second;
            oh.scene_normals_ = scene_normals_;

            std::vector < pcl::Correspondences > corresp_clusters;
            cg_algorithm_->setSceneCloud (oh.scene_);
            cg_algorithm_->setInputCloud (oh.model_->keypoints_);

            if(cg_algorithm_->getRequiresNormals())
                cg_algorithm_->setInputAndSceneNormals(oh.model_->kp_normals_, oh.scene_normals_);

            //we need to pass the keypoints_pointcloud and the specific object hypothesis
            specificCG(scene_, scene_keypoints_, oh);
            cg_algorithm_->setModelSceneCorrespondences (oh.model_scene_corresp_);
            cg_algorithm_->cluster (corresp_clusters);

            std::cout << "Instances: " << corresp_clusters.size () << " Total correspondences:" << oh.model_scene_corresp_->size () << " " << it_map->first << std::endl;
            std::vector<bool> good_indices_for_hypothesis (corresp_clusters.size (), true);

            if (param_.threshold_accept_model_hypothesis_ < 1.f)
            {
                //sort the hypotheses for each model according to their correspondences and take those that are threshold_accept_model_hypothesis_ over the max cardinality
                int max_cardinality = -1;
                for (size_t i = 0; i < corresp_clusters.size (); i++)
                {
                    //std::cout <<  (corresp_clusters[i]).size() << " -- " << (*(*it_map).second.model_scene_corresp).size() << std::endl;
                    if (max_cardinality < static_cast<int> (corresp_clusters[i].size ()))
                        max_cardinality = static_cast<int> (corresp_clusters[i].size ());
                }

                for (size_t i = 0; i < corresp_clusters.size (); i++)
                {
                    if (static_cast<float> ((corresp_clusters[i]).size ()) < (param_.threshold_accept_model_hypothesis_ * static_cast<float> (max_cardinality)))
                        good_indices_for_hypothesis[i] = false;
                }
            }

            size_t kept = 0;
            for (size_t i = 0; i < corresp_clusters.size (); i++)
            {
                if (!good_indices_for_hypothesis[i])
                    continue;

                Eigen::Matrix4f best_trans;
                typename pcl::registration::TransformationEstimationSVD < PointT, PointT > t_est;
                t_est.estimateRigidTransformation (*oh.model_->keypoints_, *oh.scene_, corresp_clusters[i], best_trans);

                models_.push_back (oh.model_);
                transforms_.push_back (best_trans);

                kept++;
            }

            std::cout << "kept " << kept << " out of " << corresp_clusters.size () << std::endl;
        }
        clearSpecificCG();

        std::cout << "Number of hypotheses:" << models_.size() << std::endl;

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
v4r::LocalRecognitionPipeline<Distance, PointT, FeatureT>::getView (const ModelT & model, size_t view_id, typename pcl::PointCloud<PointT>::Ptr &view)
{
    view.reset (new pcl::PointCloud<PointT>);
    std::stringstream view_fn; view_fn << training_dir_ << "/" << model.class_ << "/" << model.id_ << "/" << descr_name_ + "/view_" << setfill('0') << setw(view_id_length_) << view_id << ".pcd";
    pcl::io::loadPCDFile (view_fn.str (), *view);
}

template<template<class > class Distance, typename PointT, typename FeatureT>
pcl::Normal
v4r::LocalRecognitionPipeline<Distance, PointT, FeatureT>::getKpNormal (const ModelT & model, size_t keypoint_id, size_t view_id)
{
    if (param_.use_cache_)
        return model.kp_normals_->points[keypoint_id];

    pcl::PointCloud<pcl::Normal> normals_cloud;
    std::stringstream normals_fn; normals_fn << training_dir_ << "/" << model.class_ << "/" << model.id_ << "/" << descr_name_ + "/keypoint_normals_" << setfill('0') << setw(view_id_length_) << view_id << ".pcd";
    pcl::io::loadPCDFile (normals_fn.str(), normals_cloud);

    std::stringstream pose_fn; pose_fn << training_dir_ << "/" << model.class_ << "/" << model.id_ << "/pose_" << setfill('0') << setw(view_id_length_) << view_id << ".txt";
    Eigen::Matrix4f pose_matrix;
    v4r::io::readMatrixFromFile( pose_fn.str (), pose_matrix);

    pcl::Normal n;
    n.getNormalVector3fMap () = pose_matrix.block<3,3>(0,0) * normals_cloud.points[ keypoint_id ].getNormalVector3fMap ();
    return n;
}


template<template<class > class Distance, typename PointT, typename FeatureT>
PointT
v4r::LocalRecognitionPipeline<Distance, PointT, FeatureT>::getKeypoint (const ModelT & model, size_t keypoint_id, size_t view_id)
{
    if (param_.use_cache_)
        return model.keypoints_->points[keypoint_id];

    std::stringstream kp_fn; kp_fn << training_dir_ << "/" << model.class_ << "/" << model.id_ << "/" << descr_name_ + "/keypoints_" << setfill('0') << setw(view_id_length_) << view_id << ".pcd";
    pcl::PointCloud<PointT> keypoint_cloud;
    pcl::io::loadPCDFile (kp_fn.str(), keypoint_cloud);

    std::stringstream pose_fn; pose_fn << training_dir_ << "/" << model.class_ << "/" << model.id_ << "/pose_" << setfill('0') << setw(view_id_length_) << view_id << ".txt";
    Eigen::Matrix4f pose_matrix;
    v4r::io::readMatrixFromFile( pose_fn.str (), pose_matrix);

    PointT kp;
    kp.getVector4fMap () = pose_matrix * keypoint_cloud[ keypoint_id ].getVector4fMap ();
    return kp;
}
