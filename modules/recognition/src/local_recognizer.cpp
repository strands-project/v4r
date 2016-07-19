#include <v4r/common/organized_edge_detection.h>
#include <v4r/common/convertCloud.h>
#include <v4r/common/miscellaneous.h>
#include <v4r/common/normals.h>
#include <v4r/features/types.h>
#include <v4r/io/eigen.h>
#include <v4r/segmentation/dominant_plane_segmenter.h>
#include <v4r/segmentation/multiplane_segmenter.h>
#include <v4r/recognition/local_recognizer.h>

#include <pcl/features/boundary.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/registration/transformation_estimation_svd.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include <glog/logging.h>
#include <sstream>
#include <omp.h>

namespace v4r
{

template<typename PointT>
void
LocalRecognitionPipeline<PointT>::visualizeKeypoints() const
{
    if(!vis_)
        vis_.reset( new pcl::visualization::PCLVisualizer("keypoints"));

    std::stringstream title; title << keypoint_indices_.size() << " " << estimator_->getFeatureDescriptorName() << " keypoints";
    vis_->setWindowName(title.str());
    vis_->removeAllPointClouds();
    vis_->removeAllShapes();
    vis_->addPointCloud(scene_, "scene");

    if(!keypoint_indices_.empty())
    {
        pcl::PointCloud<PointT> colored_kps;
        pcl::PointCloud<pcl::Normal> kp_normals;
        pcl::PointCloud<PointT> colored_kps_unfiltered;
        pcl::PointCloud<pcl::Normal> kp_unfiltered_normals;
        pcl::copyPointCloud(*scene_, keypoint_indices_, colored_kps);
        pcl::copyPointCloud(*scene_normals_, keypoint_indices_, kp_normals);
        pcl::copyPointCloud(*scene_, keypoint_indices_unfiltered_, colored_kps_unfiltered);
        pcl::copyPointCloud(*scene_normals_, keypoint_indices_unfiltered_, kp_unfiltered_normals);
        for(PointT &p : colored_kps.points)
        {
            p.r=255.f;
            p.g=0.f;
            p.b=0.f;
        }
        for(PointT &p : colored_kps_unfiltered.points)
        {
            p.r=0.f;
            p.g=255.f;
            p.b=0.f;
        }

//        vis_->addPointCloudNormals<PointT, pcl::Normal>(colored_kps_unfiltered.makeShared(), kp_unfiltered_normals.makeShared(), 10, 0.05, "kp_normals_unfiltered");
        vis_->addPointCloud(colored_kps_unfiltered.makeShared(), "kps_unfiltered");
        vis_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "kps_unfiltered");

//        vis_->addPointCloudNormals<PointT, pcl::Normal>(colored_kps.makeShared(), kp_normals.makeShared(), 10, 0.05, "normals_model");
        vis_->addPointCloud(colored_kps.makeShared(), "kps");
        vis_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "kps");

    }
    vis_->setBackgroundColor(1,1,1);
    vis_->resetCamera();
    vis_->spin();
}

template<typename PointT>
void
LocalRecognitionPipeline<PointT>::loadFeaturesFromDisk ()
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
            for(int sig_id=0; sig_id<nrows; sig_id++)
                f.read((char*) &signature[sig_id][0], sizeof(signature[sig_id][0])*signature[sig_id].size());
            f.close();

            flann_model descr_model;
            descr_model.model = m;
            descr_model.view_id = m->view_filenames_[v_id];

            size_t kp_id_offset = 0;
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
void
LocalRecognitionPipeline<PointT>::filterKeypoints (bool filter_signatures)
{
    if (keypoint_indices_.empty() )
        return;

    std::vector<bool> kp_is_kept(keypoint_indices_.size(), true);

    if(param_.visualize_keypoints_)
        keypoint_indices_unfiltered_ = keypoint_indices_;

    typename pcl::search::KdTree<PointT>::Ptr tree;
    pcl::PointCloud<pcl::Normal>::Ptr normals_for_planarity_check;
    if(param_.filter_planar_)
    {
        pcl::ScopeTime tt("Computing planar keypoints");
        normals_for_planarity_check.reset ( new pcl::PointCloud<pcl::Normal> );

        if(!tree)
            tree.reset(new pcl::search::KdTree<PointT>);
        pcl::NormalEstimationOMP<PointT, pcl::Normal> normalEstimation;
        normalEstimation.setInputCloud(scene_);
        boost::shared_ptr< std::vector<int> > IndicesPtr (new std::vector<int>);
        *IndicesPtr = keypoint_indices_;
        normalEstimation.setIndices(IndicesPtr);
        normalEstimation.setRadiusSearch(param_.planar_support_radius_);
        normalEstimation.setSearchMethod(tree);
        normalEstimation.compute(*normals_for_planarity_check);

        for(size_t i=0; i<keypoint_indices_.size(); i++)
        {
            if(normals_for_planarity_check->points[i].curvature < param_.threshold_planar_)
                kp_is_kept[i] = false;
        }
    }

    if (param_.filter_border_pts_)
    {
        pcl::ScopeTime tt("Computing boundary points");
        CHECK(scene_->isOrganized());
        //compute depth discontinuity edges
        OrganizedEdgeBase<PointT, pcl::Label> oed;
        oed.setDepthDisconThreshold (0.05f); //at 1m, adapted linearly with depth
        oed.setMaxSearchNeighbors(100);
        oed.setEdgeType (  OrganizedEdgeBase<PointT,           pcl::Label>::EDGELABEL_OCCLUDING
                         | OrganizedEdgeBase<pcl::PointXYZRGB, pcl::Label>::EDGELABEL_OCCLUDED
                         | OrganizedEdgeBase<pcl::PointXYZRGB, pcl::Label>::EDGELABEL_NAN_BOUNDARY
                         );
        oed.setInputCloud (scene_);

        pcl::PointCloud<pcl::Label> labels;
        std::vector<pcl::PointIndices> edge_indices;
        oed.compute (labels, edge_indices);

        // count indices to allocate memory beforehand
        size_t kept=0;
        for (size_t j = 0; j < edge_indices.size (); j++)
            kept += edge_indices[j].indices.size ();

        std::vector<int> discontinuity_edges;
        discontinuity_edges.resize(kept);

        kept=0;
        for (size_t j = 0; j < edge_indices.size (); j++)
        {
            for (size_t i = 0; i < edge_indices[j].indices.size (); i++)
                discontinuity_edges[kept++] = edge_indices[j].indices[i];
        }

        cv::Mat boundary_mask = cv::Mat_<unsigned char>::zeros(scene_->height, scene_->width);
        for(size_t i=0; i<discontinuity_edges.size(); i++)
        {
            int idx = discontinuity_edges[i];
            int u = idx%scene_->width;
            int v = idx/scene_->width;

            boundary_mask.at<unsigned char>(v,u) = 255;
        }


        cv::Mat element = cv::getStructuringElement( cv::MORPH_ELLIPSE,
                                                     cv::Size( 2*param_.boundary_width_ + 1, 2*param_.boundary_width_+1 ),
                                                     cv::Point( param_.boundary_width_, param_.boundary_width_ ) );
        cv::Mat boundary_mask_dilated;
        cv::dilate( boundary_mask, boundary_mask_dilated, element );

        kept=0;
        for(size_t i=0; i<keypoint_indices_.size(); i++)
        {
            int idx = keypoint_indices_[i];
            int u = idx%scene_->width;
            int v = idx/scene_->width;

            if ( boundary_mask_dilated.at<unsigned char>(v,u) )
                kp_is_kept[i] = false;
        }
    }

    size_t kept=0;
    for(size_t i=0; i<keypoint_indices_.size(); i++)
    {
        if(kp_is_kept[i])
        {
            keypoint_indices_[kept] = keypoint_indices_[i];
            if( filter_signatures )
            {
                scene_signatures_[kept] = scene_signatures_[i];
            }
            kept++;
        }
    }
    keypoint_indices_.resize(kept);
    if( filter_signatures )
    {
        scene_signatures_.resize(kept);
    }

    indices_.clear();
}

template<typename PointT>
void
LocalRecognitionPipeline<PointT>::extractKeypoints ()
{
    pcl::ScopeTime t("Extracting all keypoints with filtering");

    keypoint_indices_.clear();

    std::vector<bool> obj_mask;
    if(indices_.empty())
        obj_mask.resize( scene_->points.size(), true);
    else
        obj_mask = createMaskFromIndices(indices_, scene_->points.size());

    for (size_t i = 0; i < keypoint_extractor_.size (); i++)
    {
        KeypointExtractor<PointT> &ke = *keypoint_extractor_[i];
        const std::string time_txt = "Extracting " + ke.getKeypointExtractorName() + " keypoints";
        pcl::ScopeTime tt(time_txt.c_str());

        ke.setInputCloud (scene_);
        if (ke.needNormals ())
            ke.setNormals (scene_normals_);

        pcl::PointCloud<PointT> detected_keypoints;
        ke.compute (detected_keypoints);

        std::vector<int> kp_indices = ke.getKeypointIndices();

        // only keep keypoints which are finite (with finite normals), are closer than the maximum allowed distance,
        // belong to the Region of Interest and are not planar (if planarity filter is on)
        size_t kept=0;
        for(size_t kp_id=0; kp_id<kp_indices.size(); kp_id++)
        {
            int idx = kp_indices[kp_id];
            if(     pcl::isFinite(scene_->points[idx]) &&
                    (!ke.needNormals() || pcl::isFinite(scene_normals_->points[idx]))
                    && scene_->points[idx].z < param_.max_distance_
                    && obj_mask[idx])
            {
                kp_indices[kept] = idx;
                kept++;
            }
        }
        kp_indices.resize(kept);
        keypoint_indices_.insert(keypoint_indices_.end(), kp_indices.begin(), kp_indices.end());
    }

    indices_.clear();
}

template<typename PointT>
bool
LocalRecognitionPipeline<PointT>::initialize (bool force_retrain)
{
    CHECK(estimator_);
    CHECK(source_);
    feat_kp_set_from_outside_ = false;
    initialization_phase_ = true;
    descr_name_ = estimator_->getFeatureDescriptorName();
    size_t descr_dims = estimator_->getFeatureDimensions();
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

        std::cout << "Loading " << estimator_->getFeatureDescriptorName() << " for model " << m->class_ << " " << m->id_ <<  " ("<< m->view_filenames_.size () << " views)" << std::endl;

        io::createDirIfNotExist(dir);

        if(!source_->getLoadIntoMemory())
            source_->loadInMemorySpecificModel(*m);

        for (size_t v = 0; v < m->view_filenames_.size(); v++)
        {
            // check if descriptor file exists. If not, train this view
            std::string descriptor_basename (m->view_filenames_[v]);
            boost::replace_last(descriptor_basename, source_->getViewPrefix(), "/descriptors_");
            boost::replace_last(descriptor_basename, ".pcd", ".desc");

            if( !io::existsFile(dir+descriptor_basename) )
            {
                std::stringstream foo; foo << "Training " << estimator_->getFeatureDescriptorName() << " on view " << m->class_ << "/" << m->id_ << "/" << m->view_filenames_[v];

                pcl::ScopeTime t(foo.str().c_str());
                scene_ = m->views_[v];
                scene_normals_.reset(new pcl::PointCloud<pcl::Normal>);
                {
                    pcl::ScopeTime tt("Computing scene normals");
                    computeNormals<PointT>(scene_, scene_normals_, param_.normal_computation_method_);
                }
                indices_ = m->indices_[v];

                if(!computeFeatures())
                    continue;

                CHECK(scene_signatures_.size() == keypoint_indices_.size());

                std::ofstream f(dir + descriptor_basename, std::ofstream::binary );
                int rows = scene_signatures_.size();
                int cols = scene_signatures_[0].size();
                f.write((const char*)(&rows), sizeof(rows));
                f.write((const char*)(&cols), sizeof(cols));
                for(size_t sig_id=0; sig_id<scene_signatures_.size(); sig_id++)
                    f.write(reinterpret_cast<const char*>(&scene_signatures_[sig_id][0]), sizeof(scene_signatures_[sig_id][0]) * scene_signatures_[sig_id].size());
                f.close();

                std::string keypoint_basename (m->view_filenames_[v]);
                boost::replace_last(keypoint_basename, source_->getViewPrefix(), "/keypoints_");
                pcl::PointCloud<PointT> scene_keypoints;
                pcl::copyPointCloud(*scene_, keypoint_indices_, scene_keypoints);
                pcl::io::savePCDFileBinary (dir + keypoint_basename, scene_keypoints);

                std::string kp_normals_basename (m->view_filenames_[v]);
                boost::replace_last(kp_normals_basename, source_->getViewPrefix(), "/keypoint_normals_");
                pcl::PointCloud<pcl::Normal> normals_keypoints;
                pcl::copyPointCloud(*scene_normals_, keypoint_indices_, normals_keypoints);
                pcl::io::savePCDFileBinary (dir + kp_normals_basename, normals_keypoints);
            }
        }

        if(!source_->getLoadIntoMemory())
        {
            m->views_.clear();
        }
    }

    loadFeaturesFromDisk();
//    if(param_.use_codebook_)
//    {
//        computeFeatureProbabilities();
//        visualizeModelProbabilities();
//        filterModelKeypointsBasedOnPriorProbability();
//        visualizeModelProbabilities();
//        computeCodebook();
//    }
    initialization_phase_ = false;
    indices_.clear();
    return true;
}

template<typename PointT>
void
LocalRecognitionPipeline<PointT>::featureMatching()
{
    CHECK (scene_signatures_.size () == keypoint_indices_.size() );

    std::cout << "computing " << scene_signatures_.size () << " matches." << std::endl;

    int size_feat = scene_signatures_[0].size();

    flann::Matrix<float> distances (new float[param_.knn_], 1, param_.knn_);
    flann::Matrix<int> indices (new int[param_.knn_], 1, param_.knn_);
    flann::Matrix<float> query_desc (new float[size_feat], 1, size_feat);

    for (size_t idx = 0; idx < scene_signatures_.size (); idx++)
    {
        memcpy (&query_desc.ptr()[0], &scene_signatures_[idx][0], size_feat * sizeof(float));

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
            if ((it_map = local_obj_hypotheses_.find (f.model->id_)) != local_obj_hypotheses_.end ())
            {
                LocalObjectHypothesis<PointT> &loh = it_map->second;
                pcl::Correspondence c ( (int)f.keypoint_id, (int)idx, m_dist);
                loh.model_scene_corresp_.push_back(c);
                loh.indices_to_flann_models_.push_back( indices[0][i] );
            }
            else //create object hypothesis
            {
                LocalObjectHypothesis<PointT> loh;
                loh.model_ = f.model;
                loh.model_scene_corresp_.reserve (scene_signatures_.size () * param_.knn_);
                loh.indices_to_flann_models_.reserve(scene_signatures_.size () * param_.knn_);
                loh.model_scene_corresp_.push_back( pcl::Correspondence ((int)f.keypoint_id, (int)idx, m_dist) );
                loh.indices_to_flann_models_.push_back( indices[0][i] );
                local_obj_hypotheses_[loh.model_->id_] = loh;
            }
        }
    }

    delete[] indices.ptr ();
    delete[] distances.ptr ();
    delete[] query_desc.ptr ();

    typename symHyp::iterator it_map;
    for (it_map = local_obj_hypotheses_.begin(); it_map != local_obj_hypotheses_.end (); it_map++)
        it_map->second.model_scene_corresp_.shrink_to_fit();   // free memory
}

template<typename PointT>
void
LocalRecognitionPipeline<PointT>::featureEncoding()
{
    pcl::ScopeTime t("Feature Encoding");
    if (feat_kp_set_from_outside_)
    {
        LOG(INFO) << "Signatures and Keypoints set from outside.";
        feat_kp_set_from_outside_ = false;
    }
    else
    {
        estimator_->setInputCloud(scene_);
        estimator_->setNormals(scene_normals_);

        if(estimator_->getFeatureType() == FeatureType::SIFT_GPU || estimator_->getFeatureType() == FeatureType::SIFT_OPENCV) // for SIFT we do not need to extract keypoints explicitly
            estimator_->setIndices(indices_);
        else
            estimator_->setIndices(keypoint_indices_);

        estimator_->compute (scene_signatures_);

        if(keypoint_indices_.empty())
            keypoint_indices_ = estimator_->getKeypointIndices();
    }


    CHECK ( keypoint_indices_.size() == scene_signatures_.size() );


    // remove signatures (with corresponding keypoints) with nan elements
    size_t kept=0;
    for(size_t sig_id=0; sig_id<scene_signatures_.size(); sig_id++)
    {
        bool keep_this = true;
        for(size_t dim=0; dim< scene_signatures_[sig_id].size(); dim++)
        {
            if( std::isnan(scene_signatures_[sig_id][dim]) || !std::isfinite(scene_signatures_[sig_id][dim]) )
            {
                keep_this = false;
                break;
            }
        }

        if(keep_this)
        {
            scene_signatures_[kept] = scene_signatures_[sig_id];
            keypoint_indices_[kept] = keypoint_indices_[sig_id];
            kept++;
        }
    }
    keypoint_indices_.resize(kept);
    scene_signatures_.resize(kept);

    if(keypoint_indices_.empty())
        return;
}

template<typename PointT>
bool
LocalRecognitionPipeline<PointT>::computeFeatures()
{
    scene_signatures_.resize(0);
    keypoint_indices_.clear();

    CHECK (scene_);

    if (feat_kp_set_from_outside_)
    {
        LOG(INFO) << "Signatures and Keypoints set from outside.";
        feat_kp_set_from_outside_ = false;
    }

    if(param_.filter_points_above_plane_ && !initialization_phase_)
    {
//        pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
//        pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
//        pcl::SACSegmentation<PointT> seg;
//        seg.setOptimizeCoefficients (true);
//        seg.setModelType (pcl::SACMODEL_PLANE);
//        seg.setMethodType (pcl::SAC_RANSAC);
//        seg.setDistanceThreshold (0.01);
//        seg.setInputCloud (scene_);
//        seg.segment (*inliers, *coefficients);

//        Eigen::Vector4f table_plane (coefficients->values[0],
//                coefficients->values[1],
//                coefficients->values[2],
//                coefficients->values[3]);
//        table_plane.normalize();


//        typename DominantPlaneSegmenter<PointT>::Parameter seg_param;
//        seg_param.compute_table_plane_only_ = true;
//        DominantPlaneSegmenter<PointT> seg ( seg_param );
        typename MultiplaneSegmenter<PointT>::Parameter seg_param;
        seg_param.min_cluster_size_ = 10000;
        seg_param.sensor_noise_max_ = 0.015f;
        MultiplaneSegmenter<PointT> seg (seg_param);
        seg.setInputCloud(scene_);
        seg.setNormalsCloud(scene_normals_);
        seg.segment();
        Eigen::Vector4f table_plane = seg.getTablePlane();

        // flip table plane vector towards viewpoint
        Eigen::Vector3f vp;
        vp(0)=vp(1)=0.f; vp(2) = 1;
        Eigen::Vector3f table_vec = table_plane.head(3);
        if(vp.dot(table_vec)>0)
            table_plane *= -1.f;

        if ( 0 ) ///TODO: CHECK IF A DOMINANT PLANE WAS FOUND!! ( inliers->indices.empty() )
        {
            std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
        }
        else
        {
            typename pcl::PointCloud<PointT>::Ptr filtered_scene (new pcl::PointCloud<PointT>(*scene_));
            for (size_t j = 0; j < filtered_scene->points.size (); j++)
            {
                const Eigen::Vector4f xyz_p = filtered_scene->points[j].getVector4fMap ();

                if ( !pcl::isFinite( filtered_scene->points[j] ) )
                    continue;

                float val = xyz_p.dot(table_plane);

                if (val < 0.01f)
                {
                    filtered_scene->points[j].x = std::numeric_limits<float>::quiet_NaN ();
                    filtered_scene->points[j].y = std::numeric_limits<float>::quiet_NaN ();
                    filtered_scene->points[j].z = std::numeric_limits<float>::quiet_NaN ();
                }
            }

            if((int)filtered_scene->points.size() > param_.min_plane_size_)
                scene_ = filtered_scene;
            else
                std::cerr << "Could not find a proper dominant plane!" << std::endl;
        }
    }

    if(estimator_->getFeatureType() == FeatureType::SIFT_GPU || estimator_->getFeatureType() == FeatureType::SIFT_OPENCV) // for SIFT we do not need to extract keypoints explicitly
    {
        featureEncoding();
        filterKeypoints(true);
    }
    else
    {
        extractKeypoints();

        if(keypoint_indices_.empty())
            return false;

        filterKeypoints();
        featureEncoding();
    }

    if(keypoint_indices_.empty())
        return false;

    return true;
}

template<typename PointT>
void
LocalRecognitionPipeline<PointT>::recognize ()
{
    local_obj_hypotheses_.clear();
    keypoint_indices_.clear();

    if(!computeFeatures())
    {
        indices_.clear();
        return;
    }

    std::cout << "Number of " << estimator_->getFeatureDescriptorName() << " features: " << keypoint_indices_.size() << std::endl;

//    if(param_.use_codebook_)
//        filterKeypoints();

    if(param_.visualize_keypoints_)
        visualizeKeypoints();

    featureMatching();
    indices_.clear();
}

//template class V4R_EXPORTS LocalRecognitionPipeline<pcl::PointXYZ>;
template class V4R_EXPORTS LocalRecognitionPipeline<pcl::PointXYZRGB>;
}


