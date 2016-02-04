#include <v4r/common/miscellaneous.h>
#include <v4r/common/normals.h>
#include <v4r/io/eigen.h>
#include <v4r/recognition/local_recognizer.h>

#include <pcl/registration/transformation_estimation_svd.h>
//#include <pcl/visualization/pcl_visualizer.h>

#include <glog/logging.h>
#include <sstream>

namespace v4r
{

template<typename PointT>
void
LocalRecognitionPipeline<PointT>::loadFeaturesAndCreateFLANN ()
{
    std::vector<ModelTPtr> models = source_->getModels();
    flann_models_.clear();
    std::vector<std::vector<float> > descriptors;

    for (size_t m_id = 0; m_id < models.size (); m_id++)
    {
        ModelTPtr m = models[m_id];
        const std::string out_train_path = models_dir_  + "/" + m->class_ + "/" + m->id_ + "/" + descr_name_;
        const std::string in_train_path = models_dir_  + "/" + m->class_ + "/" + m->id_ + "/views/";

        for(size_t v_id=0; v_id< m->view_filenames_.size(); v_id++)
        {
            std::string signature_basename (m->view_filenames_[v_id]);
            boost::replace_last(signature_basename, source_->getViewPrefix(), "/descriptors_");
            boost::replace_last(signature_basename, ".pcd", ".desc");

            std::ifstream f (out_train_path + signature_basename, std::ifstream::binary);
            if(!f.is_open()) {
                std::cerr << "Could not find signature file " << out_train_path << signature_basename << std::endl;
                continue;
            }
            int nrows, ncols;
            f.read((char*) &nrows, sizeof(nrows));
            f.read((char*) &ncols, sizeof(ncols));
            std::vector<std::vector<float> > signature (nrows, std::vector<float>(ncols));
            for(size_t sig_id=0; sig_id<nrows; sig_id++)
                f.read((char*) &signature[sig_id][0], sizeof(signature[sig_id][0])*signature[sig_id].size());
            f.close();

            flann_model descr_model;
            descr_model.model = m;
            descr_model.view_id = m->view_filenames_[v_id];

            size_t kp_id_offset = 0;

            if (param_.use_cache_) //load model data (keypoints, pose and normals for each training view) and save them to cache
            {
                std::string pose_basename (m->view_filenames_[v_id]);
                boost::replace_last(pose_basename, source_->getViewPrefix(), "/pose_");
                boost::replace_last(pose_basename, ".pcd", ".txt");

                Eigen::Matrix4f pose_matrix = io::readMatrixFromFile( in_train_path + pose_basename);

                std::string keypoint_basename (m->view_filenames_[v_id]);
                boost::replace_last(keypoint_basename, source_->getViewPrefix(), + "/keypoints_");
                typename pcl::PointCloud<PointT>::Ptr keypoints (new pcl::PointCloud<PointT> ());
                pcl::io::loadPCDFile (out_train_path + keypoint_basename, *keypoints);

                std::string kp_normals_basename (m->view_filenames_[v_id]);
                boost::replace_last(kp_normals_basename, source_->getViewPrefix(), "/keypoint_normals_");
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

//                size_t existing_kp = m->kp_info_.size();
//                m->kp_info_.resize(existing_kp + keypoints->points.size());
            }

            for (size_t dd = 0; dd < signature.size (); dd++)
            {
                descr_model.keypoint_id = kp_id_offset + dd;
                descriptors.push_back(signature[dd]);
                flann_models_.push_back (descr_model);
            }
        }
    }
    std::cout << "Total number of " << estimator_->getFeatureDescriptorName() << " features within the model database: " << flann_models_.size () << std::endl;

    flann_data_.reset (new flann::Matrix<float>(new float[descriptors.size () * descriptors[0].size()], descriptors.size (), descriptors[0].size()));
    for (size_t i = 0; i < flann_data_->rows; i++) {
      for (size_t j = 0; j < flann_data_->cols; j++) {
        flann_data_->ptr()[i * flann_data_->cols + j] = descriptors[i][j];
      }
    }

    if(param_.distance_metric_==2)
    {
        flann_index_l2_.reset( new flann::Index<flann::L2<float> > (*flann_data_, flann::KDTreeIndexParams (4)));
        flann_index_l2_->buildIndex();
    }
    else
    {
        flann_index_l1_.reset( new flann::Index<flann::L1<float> > (*flann_data_, flann::KDTreeIndexParams (4)));
        flann_index_l1_->buildIndex();
    }

}


template<typename PointT>
bool
LocalRecognitionPipeline<PointT>::initialize (bool force_retrain)
{
    if(!estimator_)
        throw std::runtime_error("Keypoint extractor with feature estimator is not set!");

    descr_name_ = estimator_->getFeatureDescriptorName();

    std::vector<ModelTPtr> models = source_->getModels();

    std::cout << "Models size:" << models.size () << std::endl;

    if (force_retrain)
    {
        for (size_t i = 0; i < models.size (); i++)
            source_->removeDescDirectory (*models[i], models_dir_, descr_name_);
    }

    for (ModelTPtr &m : models)
    {
        const std::string dir = models_dir_ + "/" + m->class_ + "/" + m->id_ + "/" + descr_name_;

        if (!io::existsFolder(dir))
        {
            std::cout << "Model " << m->class_ << " " << m->id_ << " not trained. Training " << estimator_->getFeatureDescriptorName() << " on " << m->view_filenames_.size () << " views..." << std::endl;

            if(!source_->getLoadIntoMemory())
                source_->loadInMemorySpecificModel(*m);

            for (size_t v = 0; v < m->view_filenames_.size(); v++)
            {
                std::vector<std::vector<float> > all_signatures;
                std::vector<std::vector<float> > object_signatures;
                typename pcl::PointCloud<PointT> all_keypoints;
                typename pcl::PointCloud<PointT> object_keypoints;
                pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

                computeNormals<PointT>(m->views_[v], normals, param_.normal_computation_method_);

                std::vector<int> all_kp_indices, obj_kp_indices;
                estimator_->setNormals(normals);
                typename pcl::PointCloud<PointT> foo;
                bool success = estimator_->estimate (*m->views_[v], foo, all_keypoints, all_signatures);
                (void) success;
                estimator_->getKeypointIndices(all_kp_indices);

                // remove signatures and keypoints which do not belong to object
                std::vector<bool> obj_mask = createMaskFromIndices(m->indices_[v].indices, m->views_[v]->points.size());
                obj_kp_indices.resize( all_kp_indices.size() );
                object_signatures.resize( all_kp_indices.size() ) ;
                size_t kept=0;
                for (size_t kp_id = 0; kp_id < all_kp_indices.size(); kp_id++)
                {
                    int idx = all_kp_indices[kp_id];
                    if ( obj_mask[idx] )
                    {
                        obj_kp_indices[kept] = idx;
                        object_signatures[kept] = all_signatures[kp_id];
                        kept++;
                    }
                }
                object_signatures.resize( kept );
                obj_kp_indices.resize( kept );

                pcl::copyPointCloud( *m->views_[v], obj_kp_indices, object_keypoints);

                if (object_keypoints.points.size()) //save descriptors and keypoints to disk
                {
                    io::createDirIfNotExist(dir);
                    std::string descriptor_basename (m->view_filenames_[v]);
                    boost::replace_last(descriptor_basename, source_->getViewPrefix(), "/descriptors_");
                    boost::replace_last(descriptor_basename, ".pcd", ".desc");
                    std::ofstream f(dir + descriptor_basename, std::ofstream::binary );
                    int rows = object_signatures.size();
                    int cols = object_signatures[0].size();
                    f.write((const char*)(&rows), sizeof(rows));
                    f.write((const char*)(&cols), sizeof(cols));
                    for(size_t sig_id=0; sig_id<object_signatures.size(); sig_id++)
                        f.write(reinterpret_cast<const char*>(&object_signatures[sig_id][0]), sizeof(object_signatures[sig_id][0]) * object_signatures[sig_id].size());
                    f.close();

                    std::string keypoint_basename (m->view_filenames_[v]);
                    boost::replace_last(keypoint_basename, source_->getViewPrefix(), "/keypoints_");
                    pcl::io::savePCDFileBinary (dir + keypoint_basename, object_keypoints);

                    std::string kp_normals_basename (m->view_filenames_[v]);
                    boost::replace_last(kp_normals_basename, source_->getViewPrefix(), "/keypoint_normals_");
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
            std::cout << "Model " << m->class_ << " " << m->id_ << " already trained using " << estimator_->getFeatureDescriptorName() << "." << std::endl;
            m->views_.clear(); //there is no need to keep the views in memory once the model has been trained
        }
    }

    loadFeaturesAndCreateFLANN ();

    return true;
}

template<typename PointT>
void
LocalRecognitionPipeline<PointT>::recognize ()
{
    models_.clear();
    transforms_.clear();
    scene_keypoints_.reset(new pcl::PointCloud<PointT>);
    obj_hypotheses_.clear();

    if (feat_kp_set_from_outside_)
    {
        pcl::copyPointCloud(*scene_, scene_kp_indices_, *scene_keypoints_);
        LOG(INFO) << "Signatures and Keypoints set from outside.";
        feat_kp_set_from_outside_ = false;
    }
    else
    {
        estimator_->setNormals(scene_normals_);
        typename pcl::PointCloud<PointT> processed_foo;
        estimator_->estimate (*scene_, processed_foo, *scene_keypoints_, signatures_);
        estimator_->getKeypointIndices(scene_kp_indices_);
    }

    if (scene_keypoints_->points.size() != signatures_.size())
        throw std::runtime_error("Size of keypoint cloud is not equal to number of signatures!");

    int size_feat = signatures_[0].size();

    flann::Matrix<float> distances (new float[param_.knn_], 1, param_.knn_);
    flann::Matrix<int> indices (new int[param_.knn_], 1, param_.knn_);
    flann::Matrix<float> query_desc (new float[size_feat], 1, size_feat);

    for (size_t idx = 0; idx < signatures_.size (); idx++)
    {
        memcpy (&query_desc.ptr()[0], &signatures_[idx][0], size_feat * sizeof(float));

        if(param_.distance_metric_==2)
            flann_index_l2_->knnSearch (query_desc, indices, distances, param_.knn_, flann::SearchParams (param_.kdtree_splits_));
        else
            flann_index_l1_->knnSearch (query_desc, indices, distances, param_.knn_, flann::SearchParams (param_.kdtree_splits_));

        if(distances[0][0] > param_.max_descriptor_distance_)
            continue;

        for (size_t i = 0; i < param_.knn_; i++)
        {
            const flann_model &f = flann_models_[ indices[0][i] ];
            float m_dist = param_.correspondence_distance_weight_ * distances[0][i];

            typename symHyp::iterator it_map;
            if ((it_map = obj_hypotheses_.find (f.model->id_)) != obj_hypotheses_.end ())
            {
                ObjectHypothesis<PointT> &oh = it_map->second;
                pcl::Correspondence c ( (int)f.keypoint_id, (int)idx, m_dist);
                oh.model_scene_corresp_.push_back(c);
                oh.indices_to_flann_models_.push_back( indices[0][i] );
            }
            else //create object hypothesis
            {
                ObjectHypothesis<PointT> oh;
                oh.model_ = f.model;
                oh.model_scene_corresp_.reserve (signatures_.size () * param_.knn_);
                oh.indices_to_flann_models_.reserve(signatures_.size () * param_.knn_);
                oh.model_scene_corresp_.push_back( pcl::Correspondence ((int)f.keypoint_id, (int)idx, m_dist) );
                oh.indices_to_flann_models_.push_back( indices[0][i] );
                obj_hypotheses_[oh.model_->id_] = oh;
            }
        }
    }

    delete[] indices.ptr ();
    delete[] distances.ptr ();
    delete[] query_desc.ptr ();

    typename symHyp::iterator it_map;
    for (it_map = obj_hypotheses_.begin(); it_map != obj_hypotheses_.end (); it_map++)
        it_map->second.model_scene_corresp_.shrink_to_fit();   // free memory

    if(cg_algorithm_ && !param_.save_hypotheses_)    // correspondence grouping is not done outside
    {
        correspondenceGrouping();

        //Prepare scene and model clouds for the pose refinement step
        if ( hv_algorithm_ )
            source_->voxelizeAllModels (param_.voxel_size_icp_);

        if ( hv_algorithm_ && models_.size() )
            hypothesisVerification();

        signatures_.clear();
        scene_keypoints_.reset();
        scene_kp_indices_.clear();
    }
}


template<typename PointT>
void
LocalRecognitionPipeline<PointT>::correspondenceGrouping ()
{
    if(cg_algorithm_->getRequiresNormals() && (!scene_normals_ || scene_normals_->points.size() != scene_->points.size()))
        computeNormals<PointT>(scene_, scene_normals_, param_.normal_computation_method_);

    typename std::map<std::string, ObjectHypothesis<PointT> >::iterator it;
    for (it = obj_hypotheses_.begin (); it != obj_hypotheses_.end (); it++)
    {
        ObjectHypothesis<PointT> &oh = it->second;

        if(oh.model_scene_corresp_.size() < 3)
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

        std::cout << "Merged " << corresp_clusters.size() << " clusters into " << new_transforms.size() << " clusters. Total correspondences: " << oh.model_scene_corresp_.size () << " " << it->first << std::endl;

        //        oh.visualize(*scene_);

        size_t existing_hypotheses = models_.size();
        models_.resize( existing_hypotheses + new_transforms.size(), oh.model_  );
        transforms_.insert(transforms_.end(), new_transforms.begin(), new_transforms.end());
    }
}

}
