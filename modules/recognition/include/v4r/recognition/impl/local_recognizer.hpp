#include <v4r/recognition/local_recognizer.h>
#include <v4r/io/eigen.h>
#include <v4r/common/miscellaneous.h>
#include <sstream>

//#include <pcl/visualization/pcl_visualizer.h>
template<template<class > class Distance, typename PointInT, typename FeatureT>
void
v4r::LocalRecognitionPipeline<Distance, PointInT, FeatureT>::loadFeaturesAndCreateFLANN ()
{
    boost::shared_ptr < std::vector<ModelTPtr> > models = source_->getModels ();

    size_t idx_flann_models = 0;
    for (size_t i = 0; i < models->size (); i++)
    {
        ModelT &m = *(models->at(i));
        const std::string out_train_path = training_out_dir_  + "/" + m.class_ + "/" + m.id_ + "/" + descr_name_;
        const std::string in_train_path = training_in_dir_  + "/" + m.class_ + "/" + m.id_ + "/";
        const std::string descriptor_pattern = ".*descriptors.*.pcd";

        std::vector<std::string> desc_files;
        v4r::io::getFilesInDirectory(out_train_path, desc_files, "", descriptor_pattern, false);
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
            descr_model.model = models->at (i);

            std::istringstream iss(strs[1]);
            iss >> descr_model.view_id;

            if (use_cache_)
            {
                std::stringstream dir_pose;
                dir_pose << in_train_path << "/pose_" << setfill('0') << setw(view_id_length_) << descr_model.view_id << ".txt";

                Eigen::Matrix4f pose_matrix;
                v4r::io::readMatrixFromFile( dir_pose.str (), pose_matrix);
                std::pair<std::string, size_t> pair_model_view = std::make_pair (m.id_, descr_model.view_id);
                poses_cache_[pair_model_view] = pose_matrix;

                //load keypoints and save them to cache
                std::stringstream dir_keypoints; dir_keypoints << out_train_path << "/keypoints_" << setfill('0') << setw(view_id_length_) << descr_model.view_id << ".pcd";
                typename pcl::PointCloud<PointInT>::Ptr keypoints (new pcl::PointCloud<PointInT> ());
                pcl::io::loadPCDFile (dir_keypoints.str (), *keypoints);
                keypoints_cache_[pair_model_view] = keypoints;

                std::stringstream dir_normals; dir_normals << out_train_path << "/keypoint_normals_" << setfill('0') << setw(view_id_length_) << descr_model.view_id << ".pcd";
                pcl::PointCloud<pcl::Normal>::Ptr normals_cloud (new pcl::PointCloud<pcl::Normal> ());
                pcl::io::loadPCDFile (dir_normals.str (), *normals_cloud);
                normals_cache_[pair_model_view] = normals_cloud;
            }

            std::vector<size_t> idx_flann_models_for_this_view (signature->points.size ());

            for (size_t dd = 0; dd < signature->points.size (); dd++)
            {
                descr_model.keypoint_id = dd;
                descr_model.descr.resize (size_feat);
                memcpy (&descr_model.descr[0], &signature->points[dd].histogram[0], size_feat * sizeof(float));
                flann_models_.push_back (descr_model);
                idx_flann_models_for_this_view[dd] = idx_flann_models;
                idx_flann_models++;
            }

            std::pair< ModelTPtr, size_t > pp = std::make_pair(descr_model.model, descr_model.view_id);
            model_view_id_to_flann_models_[pp] = idx_flann_models_for_this_view;
        }
    }

    specificLoadFeaturesAndCreateFLANN();
    std::cout << "Number of features:" << flann_models_.size () << std::endl;

    std::string filename;
    if(codebook_models_.size() > 0) {
        convertToFLANN<codebook_model> (codebook_models_, flann_data_);
        filename = cb_flann_index_fn_;
    }
    else {
        convertToFLANN<flann_model> (flann_models_, flann_data_);
        filename = flann_index_fn_;
    }

#if defined (_WIN32)
    flann_index_.reset( new flann::Index<DistT> (flann_data_, flann::KDTreeIndexParams (4)));
    flann_index_->buildIndex ();
#else
    if(v4r::io::existsFile(filename))
    {
        pcl::ScopeTime t("Loading flann index");
        try
        {
            flann_index_.reset( new flann::Index<DistT> (flann_data_, flann::SavedIndexParams (filename)));
        }
        catch(std::runtime_error &e)
        {
            std::cerr << "Existing flann index cannot be loaded. Removing file and creating a new flann file..." << std::endl;
            boost::filesystem::remove(boost::filesystem::path(filename));
            flann_index_.reset( new flann::Index<DistT> (flann_data_, flann::KDTreeIndexParams (4)));
            flann_index_->buildIndex ();
            flann_index_->save (filename);
        }
    }
    else
    {
        pcl::ScopeTime t("Building and saving flann index");
        flann_index_.reset( new flann::Index<DistT> (flann_data_, flann::KDTreeIndexParams (4)));
        flann_index_->buildIndex ();
        flann_index_->save (filename);
    }
#endif

    //once the descriptors in flann_models_ have benn converted to flann_data_, i can delete them
    for(size_t i=0; i < flann_models_.size(); i++)
        flann_models_[i].descr.clear();

    for(size_t i=0; i < codebook_models_.size(); i++)
        codebook_models_[i].descr.clear();

    std::cout << "End load feature and create flann" << std::endl;
}

template<template<class > class Distance, typename PointInT, typename FeatureT>
void
v4r::LocalRecognitionPipeline<Distance, PointInT, FeatureT>::nearestKSearch (boost::shared_ptr<flann::Index<DistT> > &index,
                                                                             /*float * descr, int descr_size*/
                                                                             flann::Matrix<float> & p, int k,
                                                                             flann::Matrix<int> &indices,
                                                                             flann::Matrix<float> &distances)
{
    index->knnSearch (p, indices, distances, k, flann::SearchParams (kdtree_splits_));
}

template<template<class > class Distance, typename PointInT, typename FeatureT>
void
v4r::LocalRecognitionPipeline<Distance, PointInT, FeatureT>::reinitialize(const std::vector<std::string> & load_ids)
{
    PCL_WARN("Reinitialize LocalRecognitionPipeline with list of load_ids\n");
    std::cout << "List of models being loaded:" << load_ids.size() << std::endl;

    for(size_t i=0; i < load_ids.size(); i++)
    {
        std::cout << " ---------- " << load_ids[i] << std::endl;
    }

    flann_models_.clear();
    poses_cache_.clear();
    keypoints_cache_.clear();
    normals_cache_.clear();

    source_->setModelList(load_ids);
    source_->generate(training_out_dir_);

    initialize(false);
}

template<template<class > class Distance, typename PointInT, typename FeatureT>
void
v4r::LocalRecognitionPipeline<Distance, PointInT, FeatureT>::initialize (bool force_retrain)
{
    boost::shared_ptr < std::vector<ModelTPtr> > models;

    if(search_model_.compare("") == 0) {
        models = source_->getModels ();
    }
    else {
        models = source_->getModels (search_model_);
        //reset cache and flann structures
        if(flann_index_)
            flann_index_.reset();

        flann_models_.clear();
        poses_cache_.clear();
        keypoints_cache_.clear();
        normals_cache_.clear();
    }

    std::cout << "Models size:" << models->size () << std::endl;

    if (force_retrain)
    {
        for (size_t i = 0; i < models->size (); i++)
            source_->removeDescDirectory (*models->at (i), training_out_dir_, descr_name_);
    }

    for (size_t i = 0; i < models->size (); i++)
    {
        ModelT &m = *(models->at(i));
        std::cout << m.class_ << " " << m.id_ << std::endl;
        const std::string dir = training_out_dir_ + "/" + m.class_ + "/" + m.id_ + "/" + descr_name_;

        if (!v4r::io::existsFolder(dir))
        {
            std::cout << "Model not trained..." << m.views_->size () << std::endl;
            if(!source_->getLoadIntoMemory())
                source_->loadInMemorySpecificModel(training_in_dir_, m);

            for (size_t v = 0; v < m.view_filenames_.size(); v++)
            {
                typename pcl::PointCloud<FeatureT>::Ptr signatures (new pcl::PointCloud<FeatureT> ());
                PointInTPtr keypoints_pointcloud;
                PointInTPtr foo (new pcl::PointCloud<PointInT>);

                std::vector<std::string> strs;
                boost::split (strs, m.view_filenames_[v], boost::is_any_of ("_"));
                boost::replace_last(strs[1], ".pcd", "");
                view_id_length_ = strs[1].size();

                if(m.indices_ && (m.indices_->at (v).indices.size() > 0) && estimator_->acceptsIndices())
                    estimator_->setIndices(m.indices_->at (v));

                bool success = estimator_->estimate (m.views_->at (v), foo, keypoints_pointcloud, signatures);

                if (success && keypoints_pointcloud->points.size() && signatures->points.size())
                {
                    v4r::io::createDirIfNotExist(dir);

                    //save keypoints and descriptors to disk
                    std::stringstream keypoints_sstr;
                    keypoints_sstr << dir << "/keypoints_" << setfill('0') << setw(view_id_length_) << v << ".pcd";
                    pcl::io::savePCDFileBinary (keypoints_sstr.str (), *keypoints_pointcloud);

                    std::stringstream path_descriptor;
                    path_descriptor << dir << "/descriptors_" << setfill('0') << setw(view_id_length_) << v << ".pcd";
                    pcl::io::savePCDFileBinary (path_descriptor.str (), *signatures);

                    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
                    pcl::PointCloud<pcl::Normal>::Ptr normals_keypoints(new pcl::PointCloud<pcl::Normal>);
                    v4r::computeNormals<PointInT>(m.views_->at (v), normals, normal_computation_method_);

                    if(estimator_->acceptsIndices())
                    {
                        pcl::PointIndices indices;
                        estimator_->getKeypointIndices(indices);
                        if (indices.indices.size() != keypoints_pointcloud->points.size())
                            throw std::runtime_error("indices of computed keypoints does not have the same size as keypoint cloud!");

                        pcl::copyPointCloud(*normals, indices, *normals_keypoints);
                    }
                    else
                    {
                        std::cerr << "Estimator does not return keypoint indices!" << std::endl;
                        pcl::copyPointCloud(*normals, *normals_keypoints);
                    }

                    std::stringstream normals_fn;
                    normals_fn << dir << "/keypoint_normals_" << setfill('0') << setw(view_id_length_) << v << ".pcd";
                    pcl::io::savePCDFileBinary (normals_fn.str (), *normals_keypoints);
                }
            }

            if(!source_->getLoadIntoMemory())
                m.views_->clear();
        }
        else
        {
            std::cout << "Model already trained..." << std::endl;
            //there is no need to keep the views in memory once the model has been trained
            m.views_->clear();
        }
    }

    loadFeaturesAndCreateFLANN ();

    if(ICP_iterations_ > 0 && icp_type_ == 1)
        source_->createVoxelGridAndDistanceTransform(VOXEL_SIZE_ICP_);
}

template<template<class > class Distance, typename PointInT, typename FeatureT>
void
v4r::LocalRecognitionPipeline<Distance, PointInT, FeatureT>::recognize ()
{
    models_.reset (new std::vector<ModelTPtr>);
    transforms_.reset (new std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >);

    PointInTPtr processed;
    typename pcl::PointCloud<FeatureT>::Ptr signatures (new pcl::PointCloud<FeatureT> ());
    PointInTPtr keypoints_pointcloud;

    {
        pcl::ScopeTime t("Compute keypoints and features");
        if (signatures_ != 0 && processed_ != 0 && (signatures_->size () == keypoints_input_->points.size ()))
        {
            keypoints_pointcloud = keypoints_input_;
            signatures = signatures_;
            processed = processed_;
            std::cout << "Using the ISPK ..." << std::endl;
        }
        else
        {
            processed.reset( (new pcl::PointCloud<PointInT>));
            if (indices_.size () > 0)
            {
                if(estimator_->acceptsIndices())
                {
                    estimator_->setIndices(indices_);
                    estimator_->estimate (input_, processed, keypoints_pointcloud, signatures);
                }
                else
                {
                    PointInTPtr sub_input (new pcl::PointCloud<PointInT>);
                    pcl::copyPointCloud (*input_, indices_, *sub_input);
                    estimator_->estimate (sub_input, processed, keypoints_pointcloud, signatures);
                }
            }
            else
                estimator_->estimate (input_, processed, keypoints_pointcloud, signatures);

            processed_ = processed;
            estimator_->getKeypointIndices(keypoint_indices_);
        }
        std::cout << "Number of keypoints:" << keypoints_pointcloud->points.size () << std::endl;

        for(size_t i=0; i<keypoints_pointcloud->points.size(); i++)
        {
            assert(pcl::isFinite(keypoints_pointcloud->points[i]));
        }
    }

    keypoint_cloud_ = keypoints_pointcloud;

    int size_feat = sizeof(signatures->points[0].histogram) / sizeof(float);

    //feature matching and object hypotheses
    typename std::map<std::string, ObjectHypothesis<PointInT> > object_hypotheses;
    {
        flann::Matrix<int> indices;
        flann::Matrix<float> distances;
        distances = flann::Matrix<float> (new float[knn_], 1, knn_);
        indices = flann::Matrix<int> (new int[knn_], 1, knn_);

        Eigen::Matrix4f homMatrixPose;
        pcl::PointCloud<pcl::Normal>::Ptr normals_model_view_cloud (new pcl::PointCloud<pcl::Normal> ());
        pcl::Normal model_view_normal;

        flann::Matrix<float> p = flann::Matrix<float> (new float[size_feat], 1, size_feat);

        pcl::ScopeTime t("Generating object hypotheses");
        for (size_t idx = 0; idx < signatures->points.size (); idx++)
        {
            memcpy (&p.ptr ()[0], &signatures->points[idx].histogram[0], size_feat * sizeof(float));
            nearestKSearch (flann_index_, p, knn_, indices, distances);

            int dist = distances[0][0];
            if(dist > max_descriptor_distance_)
                continue;

            std::vector<int> flann_models_indices;
            std::vector<float> model_distances;
            if(use_codebook_)
            {
                //indices[0][0] points to the codebook entry
                int cb_entry = indices[0][0];
                //          int cb_entries = static_cast<int>(codebook_models_[cb_entry].clustered_indices_to_flann_models_.size());
                //std::cout << "Codebook entries:" << cb_entries << " " << cb_entry << " " << codebook_models_.size() << std::endl;
                flann_models_indices = codebook_models_[cb_entry].clustered_indices_to_flann_models_;
                model_distances.reserve(flann_models_indices.size());
                for(size_t ii=0; ii < flann_models_indices.size(); ii++)
                {
                    model_distances.push_back(dist);
                }
            }
            else
            {
                flann_models_indices.resize(knn_);
                model_distances.resize(knn_);
                for(size_t ii=0; ii < knn_; ii++)
                {
                    flann_models_indices[ii] = indices[0][ii];
                    model_distances[ii] = distances[0][ii];
                }
            }

            std::vector<PointInT> model_keypoints_for_scene_keypoint;
            std::vector<std::string> model_id_for_scene_keypoint;

            for (size_t ii = 0; ii < flann_models_indices.size(); ii++)
            {
                const flann_model &f = flann_models_.at (flann_models_indices[ii]);
                homMatrixPose = getPose ( *f.model, f.view_id);
                PointInT m_kp = getKeypoint (*(f.model), f.view_id, f.keypoint_id);

                PointInT m_kp_aligned;
                m_kp_aligned.getVector4fMap () = homMatrixPose.inverse () * m_kp.getVector4fMap ();
                if(model_keypoints_for_scene_keypoint.size() != 0)
                {
                    bool found = false;
                    for(size_t kk=0; kk < model_keypoints_for_scene_keypoint.size(); kk++)
                    {
                        if(model_id_for_scene_keypoint[kk].compare(flann_models_.at (flann_models_indices[kk]).model->id_) == 0)
                        {
                            if( (model_keypoints_for_scene_keypoint[kk].getVector3fMap() - m_kp_aligned.getVector3fMap()).squaredNorm() < distance_same_keypoint_)
                            {
                                found = true;
                                break;
                            }
                        }
                    }

                    if(found)
                        continue;
                }
                else
                {
                    model_keypoints_for_scene_keypoint.push_back(m_kp_aligned);
                    model_id_for_scene_keypoint.push_back(f.model->id_);
                }

                if((cg_algorithm_ && cg_algorithm_->getRequiresNormals()) || save_hypotheses_)
                {
                    getNormals (*(f.model), f.view_id, normals_model_view_cloud);
                    model_view_normal.getNormalVector3fMap () = homMatrixPose.block<3,3>(0,0).inverse () * normals_model_view_cloud->points[ii].getNormalVector3fMap ();
                }

                float m_dist = model_distances[ii];

                typename std::map<std::string, ObjectHypothesis<PointInT> >::iterator it_map;
                if ((it_map = object_hypotheses.find (f.model->id_)) != object_hypotheses.end ())
                {
                    it_map->second.correspondences_pointcloud->points.push_back(m_kp_aligned);
                    //if(estimator_->needNormals())
                    if((cg_algorithm_ && cg_algorithm_->getRequiresNormals()) || save_hypotheses_)
                    {
                        it_map->second.normals_pointcloud->points.push_back(model_view_normal);
                    }

                    pcl::Correspondence c ( it_map->second.correspondences_pointcloud->points.size()-1, static_cast<int> (idx), m_dist);
                    it_map->second.correspondences_to_inputcloud->push_back(c);
                    //            (*(*it_map).second.feature_distances_).push_back(dist);
                    it_map->second.indices_to_flann_models_.push_back(flann_models_indices[ii]);
                    assert(it_map->second.indices_to_flann_models_.size() == it_map->second.correspondences_to_inputcloud->size());
                    //            (*it_map).second.num_corr_++;
                }
                else
                {
                    //create object hypothesis
                    ObjectHypothesis<PointInT> oh;
                    oh.correspondences_pointcloud.reset (new pcl::PointCloud<PointInT> ());
                    //if(estimator_->needNormals())
                    if((cg_algorithm_ && cg_algorithm_->getRequiresNormals()) || save_hypotheses_)
                    {
                        oh.normals_pointcloud.reset (new pcl::PointCloud<pcl::Normal> ());
                        oh.normals_pointcloud->points.resize (1);
                        oh.normals_pointcloud->points.reserve (signatures->points.size ());
                        oh.normals_pointcloud->points[0] = model_view_normal;
                    }

                    //            oh.feature_distances_.reset (new std::vector<float>);
                    oh.correspondences_to_inputcloud.reset (new pcl::Correspondences ());

                    oh.correspondences_pointcloud->points.resize (1);
                    oh.correspondences_to_inputcloud->resize (1);
                    //            oh.feature_distances_->resize (1);
                    oh.indices_to_flann_models_.resize(1);

                    oh.correspondences_pointcloud->points.reserve (signatures->points.size ());
                    oh.correspondences_to_inputcloud->reserve (signatures->points.size ());
                    //            oh.feature_distances_->reserve (signatures->points.size ());
                    oh.indices_to_flann_models_.reserve(signatures->points.size ());

                    oh.correspondences_pointcloud->points[0] = m_kp_aligned;
                    oh.correspondences_to_inputcloud->at (0) = pcl::Correspondence (0, static_cast<int> (idx), m_dist);
                    //            oh.feature_distances_->at (0) = dist;
                    oh.indices_to_flann_models_[0] = flann_models_indices[ii];
                    //            oh.num_corr_ = 1;
                    oh.model_ = f.model;

                    assert(oh.indices_to_flann_models_.size() == oh.correspondences_to_inputcloud->size());
                    object_hypotheses[oh.model_->id_] = oh;
                }
            }
        }

        delete[] indices.ptr ();
        delete[] distances.ptr ();
        delete[] p.ptr ();

        typename std::map<std::string, ObjectHypothesis<PointInT> >::iterator it_map;
        for (it_map = object_hypotheses.begin(); it_map != object_hypotheses.end (); it_map++) {

            //          std::cout << "Showing local recognizer keypoints for " << it_map->first << std::endl;
            //                     pcl::visualization::PCLVisualizer viewer("Keypoint Viewer for local recognizer");
            //                     pcl::PointCloud<pcl::PointXYZRGB>::Ptr model_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
            //                     ConstPointInTPtr model_cloud_templated = it_map->second.model_->getAssembled (0.003f);
            //                     pcl::copyPointCloud(*model_cloud_templated, *model_cloud);
            //                     pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler_rgb (model_cloud);
            //                     viewer.addPointCloud<pcl::PointXYZRGB>(model_cloud, handler_rgb, "scene");

            //                     pcl::PointCloud<pcl::PointXYZRGB>::Ptr pKeypointCloud (new pcl::PointCloud<pcl::PointXYZRGB>);
            //                     PointInTPtr pKeypointCloudTemplated = it_map->second.correspondences_pointcloud;
            //                     pcl::copyPointCloud(*pKeypointCloudTemplated, *pKeypointCloud);
            //                     viewer.addPointCloudNormals<pcl::PointXYZRGB,pcl::Normal>(pKeypointCloud, it_map->second.normals_pointcloud, 5, 0.04);
            //                     viewer.spin();

            ObjectHypothesis<PointInT> oh = (*it_map).second;
            size_t num_corr = oh.correspondences_to_inputcloud->size();
            oh.correspondences_pointcloud->points.resize(num_corr);
            if((cg_algorithm_ && cg_algorithm_->getRequiresNormals()) || save_hypotheses_)
                oh.normals_pointcloud->points.resize(num_corr);

            oh.correspondences_to_inputcloud->resize(num_corr);
            //        oh.feature_distances_->resize(num_corr);
        }
    }

    if(save_hypotheses_)
    {
        pcl::ScopeTime t("Saving hypotheses...");

        if(correspondence_distance_constant_weight_ != 1.f)
        {
            PCL_WARN("correspondence_distance_constant_weight_ activated! %f", correspondence_distance_constant_weight_);
            //go through the object hypotheses and multiply the correspondences distances by the weight
            //this is done to favour correspondences from different pipelines that are more reliable than other (SIFT and SHOT corr. simultaneously fed into CG)
            typename std::map<std::string, ObjectHypothesis<PointInT> >::iterator it_map;
            for (it_map = object_hypotheses.begin (); it_map != object_hypotheses.end (); it_map++)
            {
                for(size_t k=0; k < (*it_map).second.correspondences_to_inputcloud->size(); k++)
                {
                    (*it_map).second.correspondences_to_inputcloud->at(k).distance *= correspondence_distance_constant_weight_;
                }
            }
        }

        saved_object_hypotheses_ = object_hypotheses;
    }
    else
    {
        throw std::runtime_error("This has not been implemented properly!");
        typename std::map<std::string, ObjectHypothesis<PointInT> >::iterator it_map;
        pcl::PointCloud<pcl::Normal>::Ptr all_scene_normals(new pcl::PointCloud<pcl::Normal>);
        pcl::PointCloud<pcl::Normal>::Ptr scene_normals(new pcl::PointCloud<pcl::Normal>);
        v4r::computeNormals<PointInT>(input_, scene_normals, normal_computation_method_);
        std::vector<int> correct_indices;
        v4r::getIndicesFromCloud<PointInT>(input_, keypoints_pointcloud, correct_indices);
        pcl::copyPointCloud(*all_scene_normals, correct_indices, *scene_normals);

        prepareSpecificCG(processed, keypoints_pointcloud);
        pcl::ScopeTime t("Geometric verification, RANSAC and transform estimation");
        for (it_map = object_hypotheses.begin (); it_map != object_hypotheses.end (); it_map++)
        {
            std::vector < pcl::Correspondences > corresp_clusters;
            cg_algorithm_->setSceneCloud (keypoints_pointcloud);
            cg_algorithm_->setInputCloud ((*it_map).second.correspondences_pointcloud);

            if((cg_algorithm_ && cg_algorithm_->getRequiresNormals()) || save_hypotheses_)
            {
                std::cout << "CG alg requires normals..." << ((*it_map).second.normals_pointcloud)->points.size() << " " << (scene_normals)->points.size() << std::endl;
                cg_algorithm_->setInputAndSceneNormals((*it_map).second.normals_pointcloud, scene_normals);
            }
            //we need to pass the keypoints_pointcloud and the specific object hypothesis
            specificCG(processed, keypoints_pointcloud, it_map->second);
            cg_algorithm_->setModelSceneCorrespondences ((*it_map).second.correspondences_to_inputcloud);
            cg_algorithm_->cluster (corresp_clusters);

            std::cout << "Instances:" << corresp_clusters.size () << " Total correspondences:" << (*it_map).second.correspondences_to_inputcloud->size () << " " << it_map->first << std::endl;
            std::vector<bool> good_indices_for_hypothesis (corresp_clusters.size (), true);

            if (threshold_accept_model_hypothesis_ < 1.f)
            {
                //sort the hypotheses for each model according to their correspondences and take those that are threshold_accept_model_hypothesis_ over the max cardinality
                int max_cardinality = -1;
                for (size_t i = 0; i < corresp_clusters.size (); i++)
                {
                    //std::cout <<  (corresp_clusters[i]).size() << " -- " << (*(*it_map).second.correspondences_to_inputcloud).size() << std::endl;
                    if (max_cardinality < static_cast<int> (corresp_clusters[i].size ()))
                        max_cardinality = static_cast<int> (corresp_clusters[i].size ());
                }

                for (size_t i = 0; i < corresp_clusters.size (); i++)
                {
                    if (static_cast<float> ((corresp_clusters[i]).size ()) < (threshold_accept_model_hypothesis_ * static_cast<float> (max_cardinality)))
                        good_indices_for_hypothesis[i] = false;
                }
            }

            size_t kept = 0;
            for (size_t i = 0; i < corresp_clusters.size (); i++)
            {
                if (!good_indices_for_hypothesis[i])
                    continue;

                //drawCorrespondences (processed, it_map->second, keypoints_pointcloud, corresp_clusters[i]);

                Eigen::Matrix4f best_trans;
                typename pcl::registration::TransformationEstimationSVD < PointInT, PointInT > t_est;
                t_est.estimateRigidTransformation (*(*it_map).second.correspondences_pointcloud, *keypoints_pointcloud, corresp_clusters[i], best_trans);

                models_->push_back ((*it_map).second.model_);
                transforms_->push_back (best_trans);

                kept++;
            }

            std::cout << "kept " << kept << " out of " << corresp_clusters.size () << std::endl;
        }
        clearSpecificCG();

        std::cout << "Number of hypotheses:" << models_->size() << std::endl;

        //Prepare scene and model clouds for the pose refinement step
        if (ICP_iterations_ > 0 || hv_algorithm_)
            source_->voxelizeAllModels (VOXEL_SIZE_ICP_);

        if (ICP_iterations_ > 0)
            poseRefinement();

        if (hv_algorithm_ && (models_->size () > 0))
            hypothesisVerification();
    }
}

template<template<class > class Distance, typename PointInT, typename FeatureT>
void
v4r::LocalRecognitionPipeline<Distance, PointInT, FeatureT>::getView (const ModelT & model, size_t view_id, PointInTPtr & view)
{
    view.reset (new pcl::PointCloud<PointInT>);
    std::stringstream view_fn; view_fn << training_out_dir_ << "/" << model.class_ << "/" << model.id_ << "/" << descr_name_ + "/view_" << setfill('0') << setw(view_id_length_) << view_id << ".pcd";
    pcl::io::loadPCDFile (view_fn.str (), *view);
}

template<template<class > class Distance, typename PointInT, typename FeatureT>
void
v4r::LocalRecognitionPipeline<Distance, PointInT, FeatureT>::getNormals (const ModelT & model, size_t view_id,
                                                                         pcl::PointCloud<pcl::Normal>::Ptr & normals_cloud)
{
    if (use_cache_)
    {
        typedef std::pair<std::string, size_t> mv_pair;
        mv_pair pair_model_view = std::make_pair (model.id_, view_id);

        std::map<mv_pair, pcl::PointCloud<pcl::Normal>::Ptr, std::less<mv_pair>,
                Eigen::aligned_allocator<std::pair<mv_pair, pcl::PointCloud<pcl::Normal>::Ptr > > >::iterator it =
                normals_cache_.find (pair_model_view);

        if (it != normals_cache_.end ())
        {
            normals_cloud = it->second;
            return;
        }

    }

    std::stringstream normals_fn; normals_fn << training_out_dir_ << "/" << model.class_ << "/" << model.id_ << "/" << descr_name_ + "/keypoint_normals_" << setfill('0') << setw(view_id_length_) << view_id << ".pcd";
    if (!v4r::io::existsFile(normals_fn.str()))
        throw std::runtime_error("Normal file does not exist");
    pcl::io::loadPCDFile (normals_fn.str(), *normals_cloud);
}


template<template<class > class Distance, typename PointInT, typename FeatureT>
Eigen::Matrix4f
v4r::LocalRecognitionPipeline<Distance, PointInT, FeatureT>::getPose (const ModelT & model, size_t view_id)
{
    Eigen::Matrix4f pose_matrix;
    if (use_cache_)
    {
        typedef std::pair<std::string, size_t> mv_pair;
        mv_pair pair_model_view = std::make_pair (model.id_, view_id);

        std::map<mv_pair, Eigen::Matrix4f, std::less<mv_pair>, Eigen::aligned_allocator<std::pair<mv_pair, Eigen::Matrix4f> > >::iterator it =
                poses_cache_.find (pair_model_view);

        if (it != poses_cache_.end ())
        {
            pose_matrix = it->second;
        }
    }
    else
    {
        std::stringstream pose_fn; pose_fn << training_in_dir_ << "/" << model.class_ << "/" << model.id_ << "/pose_" << setfill('0') << setw(view_id_length_) << view_id << ".txt";

        if (!v4r::io::existsFile(pose_fn.str()))
            throw std::runtime_error("Pose file does not exist!");

        v4r::io::readMatrixFromFile( pose_fn.str(), pose_matrix);
    }
    return pose_matrix;
}

template<template<class > class Distance, typename PointInT, typename FeatureT>
PointInT
v4r::LocalRecognitionPipeline<Distance, PointInT, FeatureT>::getKeypoint (const ModelT & model,
                                                                           size_t view_id,
                                                                           size_t keypoint_id)
{
    PointInT kp;
    if (use_cache_)
    {
        std::pair<std::string, size_t> pair_model_view = std::make_pair (model.id_, view_id);
        typename std::map<std::pair<std::string, size_t>, PointInTPtr>::iterator it = keypoints_cache_.find (pair_model_view);

        if (it != keypoints_cache_.end ())
        {
            kp = it->second->points[keypoint_id];
        }
    }
    else
    {
        std::stringstream kp_fn; kp_fn << training_out_dir_ << "/" << model.class_ << "/" << model.id_ << "/" << descr_name_ + "/keypoints_" << setfill('0') << setw(view_id_length_) << view_id << ".pcd";
        if(!v4r::io::existsFile(kp_fn.str()))
            throw std::runtime_error("Keypoint file does not exist");

        pcl::PointCloud<PointInT> keypoint_cloud;
        pcl::io::loadPCDFile (kp_fn.str(), keypoint_cloud);
        kp = keypoint_cloud[keypoint_id];
    }

    return kp;
}
