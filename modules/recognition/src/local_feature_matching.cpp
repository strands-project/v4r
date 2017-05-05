#include <pcl_1_8/features/organized_edge_detection.h>
#include <v4r/common/miscellaneous.h>
#include <v4r/io/eigen.h>
#include <v4r/io/filesystem.h>
#include <v4r/recognition/local_feature_matching.h>

#include <boost/filesystem.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include <pcl/common/time.h>
#include <pcl/common/transforms.h>
#include <pcl/features/boundary.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <opencv2/opencv.hpp>

#include <sstream>
#include <omp.h>

namespace v4r
{

template<typename PointT>
void
LocalFeatureMatcher<PointT>::visualizeKeypoints(const std::vector<KeypointIndex> &kp_indices,
                                                const std::vector<KeypointIndex> &unfiltered_kp_indices) const
{
    static pcl::visualization::PCLVisualizer::Ptr vis;

    if(!vis)
        vis.reset( new pcl::visualization::PCLVisualizer("keypoints"));

    std::stringstream title; title << kp_indices.size() << " keypoints";
    vis->setWindowName(title.str());
    vis->removeAllPointClouds();
    vis->removeAllShapes();
    vis->addPointCloud(scene_, "scene");

    if(!kp_indices.empty())
    {
        pcl::PointCloud<PointT> colored_kps;
        pcl::PointCloud<pcl::Normal> kp_normals;
        pcl::PointCloud<PointT> colored_kps_unfiltered;
        pcl::PointCloud<pcl::Normal> kp_unfiltered_normals;
        pcl::copyPointCloud(*scene_, kp_indices, colored_kps);
        pcl::copyPointCloud(*scene_normals_, kp_indices, kp_normals);
        pcl::copyPointCloud(*scene_, unfiltered_kp_indices, colored_kps_unfiltered);
        pcl::copyPointCloud(*scene_normals_, unfiltered_kp_indices, kp_unfiltered_normals);
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
        vis->addPointCloud(colored_kps_unfiltered.makeShared(), "kps_unfiltered");
        vis->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "kps_unfiltered");

    //        vis_->addPointCloudNormals<PointT, pcl::Normal>(colored_kps.makeShared(), kp_normals.makeShared(), 10, 0.05, "normals_model");
        vis->addPointCloud(colored_kps.makeShared(), "kps");
        vis->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "kps");

    }
    std::stringstream txt_kp; txt_kp << "Filtered keypoints (" << kp_indices.size() << ")";
    std::stringstream txt_kp_rejected; txt_kp_rejected << "Rejected keypoints (" << kp_indices.size() << ")";

    if( !vis_param_->no_text_ )
    {
        vis->addText(txt_kp.str(), 10, 10, 12, 1., 0, 0, "filtered keypoints");
        vis->addText(txt_kp_rejected.str(), 10, 20, 12, 0, 1., 0, "rejected keypoints");
    }

    vis->setBackgroundColor(1,1,1);
    vis->resetCamera();
    vis->spin();
}

template<typename PointT>
std::vector<int>
LocalFeatureMatcher<PointT>::getInlier (const std::vector<KeypointIndex> &input_keypoints) const
{
    if (input_keypoints.empty() )
        return std::vector<int>();

    boost::dynamic_bitset<> kp_is_kept(input_keypoints.size());
    kp_is_kept.set();

//    if(visualize_keypoints_)
//        keypoint_indices_unfiltered_ = input_keypoints;

    if(param_.filter_planar_)
    {
        pcl::ScopeTime tt("Computing planar keypoints");
        typename pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
        pcl::NormalEstimationOMP<PointT, pcl::Normal> normalEstimation;
        normalEstimation.setInputCloud(scene_);
        boost::shared_ptr< std::vector<int> > IndicesPtr (new std::vector<int>);
        *IndicesPtr = input_keypoints;
        normalEstimation.setIndices(IndicesPtr);
        normalEstimation.setRadiusSearch(param_.planar_support_radius_);
        normalEstimation.setSearchMethod(tree);
        pcl::PointCloud<pcl::Normal>::Ptr normals_for_planarity_check ( new pcl::PointCloud<pcl::Normal> );
        normalEstimation.compute(*normals_for_planarity_check);

        for(size_t i=0; i<input_keypoints.size(); i++)
        {
            if(normals_for_planarity_check->points[i].curvature < param_.threshold_planar_)
                kp_is_kept.reset(i);
        }
    }

    if (param_.filter_border_pts_)
    {
        pcl::ScopeTime t("Computing boundary points");
        if(scene_->isOrganized())
        {
            //compute depth discontinuity edges
            pcl_1_8::OrganizedEdgeBase<PointT, pcl::Label> oed;
            oed.setDepthDisconThreshold (0.05f); //at 1m, adapted linearly with depth
            oed.setMaxSearchNeighbors(100);
            oed.setEdgeType (  pcl_1_8::OrganizedEdgeBase<PointT, pcl::Label>::EDGELABEL_OCCLUDING
                             | pcl_1_8::OrganizedEdgeBase<PointT, pcl::Label>::EDGELABEL_OCCLUDED
                             | pcl_1_8::OrganizedEdgeBase<PointT, pcl::Label>::EDGELABEL_NAN_BOUNDARY
                             );
            oed.setInputCloud (scene_);

            pcl::PointCloud<pcl::Label> labels;
            std::vector<pcl::PointIndices> edge_indices;
            oed.compute (labels, edge_indices);

            // count indices to allocate memory beforehand
            size_t kept=0;
            for (size_t j = 0; j < edge_indices.size (); j++)
                kept += edge_indices[j].indices.size ();

            std::vector<int> discontinuity_edges (kept);

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
            for(size_t i=0; i<input_keypoints.size(); i++)
            {
                int idx = input_keypoints[i];
                int u = idx%scene_->width;
                int v = idx/scene_->width;

                if ( boundary_mask_dilated.at<unsigned char>(v,u) )
                    kp_is_kept.reset(i);
            }
        }
        else
           LOG(ERROR) << "Input scene is not organized so cannot extract edge points.";
    }

    return createIndicesFromMask<int>( kp_is_kept);
}

template<typename PointT>
std::vector<int>
LocalFeatureMatcher<PointT>::extractKeypoints (const std::vector<int> &region_of_interest)
{
    if(keypoint_extractor_.empty())
    {
        LOG(INFO) << "No keypoint extractor given. Using all points as point of interest.";
        return std::vector<int>();
    }

    pcl::ScopeTime t("Extracting all keypoints with filtering");
    boost::dynamic_bitset<> obj_mask;
    boost::dynamic_bitset<> kp_mask ( scene_->points.size(), 0);

    if( region_of_interest.empty() )    // if empty take whole cloud
    {
        obj_mask.resize( scene_->points.size(), 0);
        obj_mask.set();
    }
    else
        obj_mask = createMaskFromIndices(region_of_interest, scene_->points.size());

    bool estimator_need_normals = false;
    for(const typename LocalEstimator<PointT>::ConstPtr &est : estimators_)
    {
        if(est->needNormals())
        {
            estimator_need_normals = true;
            break;
        }
    }

    for (typename KeypointExtractor<PointT>::Ptr ke : keypoint_extractor_)
    {
        ke->setInputCloud (scene_);
        ke->setNormals (scene_normals_);
        ke->compute ();

        const std::vector<int> kp_indices = ke->getKeypointIndices();

        // only keep keypoints which are finite (with finite normals), are closer than the maximum allowed distance,
        // belong to the Region of Interest and are not planar (if planarity filter is on)
        for(int idx : kp_indices)
        {
            if(     obj_mask[idx] && pcl::isFinite( scene_->points[idx] ) &&
                    ( !estimator_need_normals || pcl::isFinite(scene_normals_->points[idx]))
                    && scene_->points[idx].getVector3fMap().norm() < param_.max_keypoint_distance_z_ )
            {
                kp_mask.set( idx );
            }
        }
    }
    return createIndicesFromMask<int>(kp_mask);
}

template<typename PointT>
void
LocalFeatureMatcher<PointT>::initialize (const std::string &trained_dir, bool retrain)
{
    CHECK ( m_db_ );
    validate();
    lomdbs_.resize( estimators_.size() );

    std::vector<typename Model<PointT>::ConstPtr> models = m_db_->getModels ();

    for (size_t est_id=0; est_id < estimators_.size(); est_id++)
    {
        LocalObjectModelDatabase::Ptr lomdb( new LocalObjectModelDatabase );
        std::vector<FeatureDescriptor> all_signatures; ///< all signatures extracted from all objects in the model database

        typename LocalEstimator<PointT>::Ptr &est = estimators_[est_id];

        for( typename Model<PointT>::ConstPtr m : models)
        {
            bf::path trained_path_feat = trained_dir; // directory where feature descriptors and keypoints are stored
            trained_path_feat /= m->id_;
            trained_path_feat /= est->getFeatureDescriptorName() + est->getUniqueId();

            std::vector<FeatureDescriptor> model_signatures;
            pcl::PointCloud<pcl::PointXYZ>::Ptr model_keypoints (new pcl::PointCloud<pcl::PointXYZ>);
            pcl::PointCloud<pcl::Normal>::Ptr model_kp_normals (new pcl::PointCloud<pcl::Normal>);

            bf::path kp_path = trained_path_feat;
            kp_path /= "keypoints.pcd";
            bf::path kp_normals_path = trained_path_feat;
            kp_normals_path /= "keypoint_normals.pcd";
            bf::path signatures_path = trained_path_feat;
            signatures_path /= "signatures.dat";

            if( !retrain && io::existsFile( kp_path) && io::existsFile( kp_normals_path ) && io::existsFile( signatures_path ) )
            {
                pcl::io::loadPCDFile( kp_path.string(), *model_keypoints );
                pcl::io::loadPCDFile( kp_normals_path.string(), *model_kp_normals );
                ifstream is(signatures_path.string(), ios::binary);
                boost::archive::binary_iarchive iar(is);
                iar >> model_signatures;
                is.close();
            }
            else
            {
                const auto training_views = m->getTrainingViews();
                std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > existing_poses;

                if(param_.train_on_individual_views_)
                {
                    for(const auto &tv : training_views)
                    {
                        std::string txt = "Training " + est->getFeatureDescriptorName() + " (with id \"" + est->getUniqueId() + "\") on view " + m->class_ + "/" + m->id_ + "/" + tv->filename_;
                        pcl::ScopeTime t( txt.c_str() );

                        std::vector<int> obj_indices;

                        Eigen::Matrix4f pose;
                        if(tv->cloud_)   // point cloud and all relevant information is already in memory (fast but needs a much memory when a lot of training views/objects)
                        {
                            scene_ = tv->cloud_;
                            scene_normals_ = tv->normals_;
                            obj_indices = tv->indices_;
                            pose = tv->pose_;
                        }
                        else
                        {
                            typename pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
                            pcl::io::loadPCDFile(tv->filename_, *cloud);

                            try
                            {
                                pose = io::readMatrixFromFile(tv->pose_filename_);
                            }
                            catch (const std::runtime_error &e)
                            {
                                LOG(ERROR) << "Could not read pose from file " << tv->pose_filename_ << "! Setting it to identity" << std::endl;
                                pose = Eigen::Matrix4f::Identity();
                            }

                            // read object mask from file
                            obj_indices.clear();
                            if ( !io::existsFile( tv->indices_filename_ ) )
                            {
                                LOG(WARNING) << "No object indices " << tv->indices_filename_ << " found for object " << m->class_ <<
                                             "/" << m->id_ << " / " << tv->filename_ << "! Taking whole cloud as object of interest!" << std::endl;
                            }
                            else
                            {
                                std::ifstream mi_f ( tv->indices_filename_ );
                                int idx;
                                while ( mi_f >> idx )
                                   obj_indices.push_back(idx);
                                mi_f.close();

                                boost::dynamic_bitset<> obj_mask = createMaskFromIndices( obj_indices, cloud->points.size() );
                                for(size_t px=0; px<cloud->points.size(); px++)
                                {
                                    if( !obj_mask[px] )
                                    {
                                        PointT &p = cloud->points[px];
                                        p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
                                    }
                                }
                            }

                            scene_ = cloud;

                            if ( 1 ) // always needs normals since we never know if correspondence grouping does! ..... this->needNormals() )
                            {
                                normal_estimator_->setInputCloud( cloud );
                                pcl::PointCloud<pcl::Normal>::Ptr normals;
                                normals = normal_estimator_->compute();
                                scene_normals_ = normals;
                            }
                        }


                        bool similar_pose_exists = false;
                        for(const Eigen::Matrix4f &ep : existing_poses)
                        {
                            Eigen::Vector3f v1 = pose.block<3,1>(0,0);
                            Eigen::Vector3f v2 = ep.block<3,1>(0,0);
                            v1.normalize();
                            v2.normalize();
                            float dotp = v1.dot(v2);
                            const Eigen::Vector3f crossp = v1.cross(v2);

                            float rel_angle_deg = acos(dotp) * 180.f / M_PI;
                            if (crossp(2) < 0)
                                rel_angle_deg = 360.f - rel_angle_deg;


                            if (rel_angle_deg < param_.required_viewpoint_change_deg_)
                            {
                                similar_pose_exists = true;
                                break;
                            }
                        }
                        if(!similar_pose_exists)
                        {
                            std::vector<int> filtered_kp_indices;

                            if( have_sift_estimator_ ) // for SIFT we do not need to extract keypoints explicitly
                                filtered_kp_indices = obj_indices;
                            else
                            {
                                const std::vector<KeypointIndex> keypoint_indices = extractKeypoints( obj_indices );
                                std::vector<int> inlier = getInlier(keypoint_indices);
                                filtered_kp_indices = filterVector<KeypointIndex> (keypoint_indices, inlier);

                                if( visualize_keypoints_)
                                    visualizeKeypoints(filtered_kp_indices, keypoint_indices);
                            }

                            std::vector<FeatureDescriptor> signatures_view;
                            featureEncoding( *est, filtered_kp_indices, filtered_kp_indices, signatures_view);

                            if( have_sift_estimator_ ) // for SIFT we do not need to extract keypoints explicitly
                            {
                                std::vector<int> inlier = getInlier(filtered_kp_indices);
                                filtered_kp_indices = filterVector<KeypointIndex> (filtered_kp_indices, inlier);
                                signatures_view = filterVector<FeatureDescriptor> (signatures_view, inlier);
                            }

                            if( filtered_kp_indices.empty() )
                                continue;

                            existing_poses.push_back(pose);
                            LOG(INFO) << "Adding " << signatures_view.size() << " " << est->getFeatureDescriptorName() <<
                                         "with id \"" << est->getUniqueId() << "\" descriptors to the model database. " << std::endl;

                            CHECK(signatures_view.size() == filtered_kp_indices.size());

                            pcl::PointCloud<pcl::PointXYZ> model_keypoints_tmp;
                            pcl::PointCloud<pcl::Normal> model_keypoint_normals_tmp;
                            pcl::copyPointCloud( *scene_, filtered_kp_indices, model_keypoints_tmp );
                            pcl::copyPointCloud( *scene_normals_, filtered_kp_indices, model_keypoint_normals_tmp );
                            pcl::transformPointCloud(model_keypoints_tmp, model_keypoints_tmp, pose);
                            v4r::transformNormals(model_keypoint_normals_tmp, model_keypoint_normals_tmp, pose);
                            *model_keypoints += model_keypoints_tmp;
                            *model_kp_normals += model_keypoint_normals_tmp;
                            model_signatures.insert(model_signatures.end(), signatures_view.begin(), signatures_view.end());

                            indices_.clear();
                        }
                        else
                            LOG(INFO) << "Ignoring view " << tv->filename_ << " because a similar camera pose exists.";
                    }
                }
                else
                {
                    scene_ = m->getAssembled(1);
                    scene_normals_ = m->getNormalsAssembled(1);

                    const std::vector<KeypointIndex> keypoint_indices = extractKeypoints( );
                    std::vector<int> inlier = getInlier(keypoint_indices);
                    std::vector<KeypointIndex> filtered_kp_indices = filterVector<KeypointIndex> (keypoint_indices, inlier);

                    if( visualize_keypoints_ )
                        visualizeKeypoints(filtered_kp_indices, keypoint_indices);

                    std::vector<FeatureDescriptor> signatures;
                    featureEncoding( *est, filtered_kp_indices, filtered_kp_indices, signatures);

                    if( filtered_kp_indices.empty() )
                        continue;

                    LOG(INFO) << "Adding " << signatures.size() << " " << est->getFeatureDescriptorName() <<
                                 " (with id \"" << est->getUniqueId() << "\") descriptors to the model database. ";

                    CHECK(signatures.size() == filtered_kp_indices.size());

                    pcl::PointCloud<pcl::PointXYZ> model_keypoints_tmp;
                    pcl::PointCloud<pcl::Normal> model_keypoint_normals_tmp;
                    pcl::copyPointCloud( *scene_, filtered_kp_indices, model_keypoints_tmp );
                    pcl::copyPointCloud( *scene_normals_, filtered_kp_indices, model_keypoint_normals_tmp );
                    *model_keypoints += model_keypoints_tmp;
                    *model_kp_normals += model_keypoint_normals_tmp;
                    model_signatures.insert(model_signatures.end(), signatures.begin(), signatures.end());
                }

                io::createDirForFileIfNotExist( kp_path.string() );
                pcl::io::savePCDFileBinaryCompressed ( kp_path.string(), *model_keypoints);
                pcl::io::savePCDFileBinaryCompressed ( kp_normals_path.string(), *model_kp_normals);
                ofstream os( signatures_path.string() , ios::binary);
                boost::archive::binary_oarchive oar(os);
                oar << model_signatures;
                os.close();
            }

    //        assert(lom->keypoints_->points.size() == model_signatures.size());

            all_signatures.insert( all_signatures.end(), model_signatures.begin(), model_signatures.end() );

            std::vector<LocalObjectModelDatabase::flann_model> flann_models_tmp ( model_signatures.size() );
            for (size_t f=0; f<model_signatures.size(); f++)
            {
                flann_models_tmp[f].model_id_ = m->id_;
                flann_models_tmp[f].keypoint_id_ = f;
            }
            lomdb->flann_models_.insert ( lomdb->flann_models_.end(), flann_models_tmp.begin(), flann_models_tmp.end() );

            LocalObjectModel::Ptr lom (new LocalObjectModel );
            lom->keypoints_ = model_keypoints;
            lom->kp_normals_ = model_kp_normals;
            lomdb->l_obj_models_[m->id_] = lom;
        }

        CHECK( lomdb->flann_models_.size() == all_signatures.size() );

        lomdb->flann_data_.reset ( new flann::Matrix<float> (
                                new float[ all_signatures.size() * all_signatures[0].size()],
                all_signatures.size(), all_signatures[0].size()));

        for (size_t i = 0; i < lomdb->flann_data_->rows; i++)
            for (size_t j = 0; j < lomdb->flann_data_->cols; j++)
                lomdb->flann_data_->ptr()[i * lomdb->flann_data_->cols + j] = all_signatures[i][j];

        LOG(INFO) << "Building the kdtree index for " << lomdb->flann_data_->rows << " elements.";

        if(param_.distance_metric_==2)
        {
            lomdb->flann_index_l2_.reset( new ::flann::Index<::flann::L2<float> > (*(lomdb->flann_data_), ::flann::KDTreeIndexParams (param_.kdtree_num_trees_)));
    //        lomdb_->flann_index_l2_.reset( new flann::Index<flann::L2<float> > (*(lomdb_->flann_data_), flann::LinearIndexParams()));
            lomdb->flann_index_l2_->buildIndex();
        }
        else if(param_.distance_metric_==3)
        {
            lomdb->flann_index_chisquare_.reset( new ::flann::Index<::flann::ChiSquareDistance<float> > (*(lomdb->flann_data_), ::flann::KDTreeIndexParams (param_.kdtree_num_trees_)));
            lomdb->flann_index_chisquare_->buildIndex();
        }
        else if(param_.distance_metric_==4)
        {
            lomdb->flann_index_hellinger_.reset( new ::flann::Index<::flann::HellingerDistance<float> > (*(lomdb->flann_data_), ::flann::KDTreeIndexParams (param_.kdtree_num_trees_)));
            lomdb->flann_index_hellinger_->buildIndex();
        }
        else
        {
            lomdb->flann_index_l1_.reset( new ::flann::Index<::flann::L1<float> > (*(lomdb->flann_data_), ::flann::KDTreeIndexParams (param_.kdtree_num_trees_)));
    //        lomdb_->flann_index_l1_.reset( new flann::Index<flann::L1<float> > (*(lomdb_->flann_data_), flann::LinearIndexParams()));
            lomdb->flann_index_l1_->buildIndex();
        }
        lomdbs_[est_id] = lomdb;
    }

    mergeKeypointsFromMultipleEstimators();
    indices_.clear();
}

template<typename PointT>
void
LocalFeatureMatcher<PointT>::mergeKeypointsFromMultipleEstimators()
{
    model_keypoints_.clear();
    model_kp_idx_range_start_.resize( estimators_.size() );

    for(size_t est_id=0; est_id<estimators_.size(); est_id++)
    {
        LocalObjectModelDatabase::ConstPtr lomdb_tmp = lomdbs_[est_id];

        for ( const auto & lo : lomdb_tmp->l_obj_models_ )
        {
            const std::string &model_id = lo.first;
            const LocalObjectModel &lom = *(lo.second);

            std::map<std::string, typename LocalObjectModel::ConstPtr>::const_iterator
                    it_loh = model_keypoints_.find(model_id);

            if ( it_loh != model_keypoints_.end () ) // append keypoints to existing ones
            {
                model_kp_idx_range_start_[est_id][model_id] = it_loh->second->keypoints_->points.size();
                *(it_loh->second->keypoints_) += *(lom.keypoints_);
                *(it_loh->second->kp_normals_) += *(lom.kp_normals_);
            }
            else    // keypoints do not exist yet for this model
            {
                model_kp_idx_range_start_[est_id][model_id] = 0;
                LocalObjectModel::Ptr lom_copy (new LocalObjectModel);
                *(lom_copy->keypoints_) = *(lom.keypoints_);
                *(lom_copy->kp_normals_) = *(lom.kp_normals_);
                model_keypoints_[model_id] = lom_copy;
            }
        }
    }
}


template<typename PointT>
void
LocalFeatureMatcher<PointT>::featureMatching(const std::vector<KeypointIndex> &kp_indices,
                                             const std::vector<FeatureDescriptor> &signatures,
                                             const LocalObjectModelDatabase::ConstPtr &lomdb,
                                             size_t model_keypoint_offset)
{
    CHECK (signatures.size () == kp_indices.size() );

    LOG(INFO) << "computing " << signatures.size () << " matches.";

    int size_feat = signatures[0].size();

    ::flann::Matrix<float> distances (new float[param_.knn_], 1, param_.knn_);
    ::flann::Matrix<int> indices (new int[param_.knn_], 1, param_.knn_);
    ::flann::Matrix<float> query_desc (new float[size_feat], 1, size_feat);

    for (size_t idx = 0; idx < signatures.size (); idx++)
    {
        memcpy (&query_desc.ptr()[0], &signatures[idx][0], size_feat * sizeof(float));

        if(param_.distance_metric_==2)
            lomdb->flann_index_l2_->knnSearch (query_desc, indices, distances, param_.knn_, ::flann::SearchParams (param_.kdtree_splits_));
        else if(param_.distance_metric_==3)
            lomdb->flann_index_chisquare_->knnSearch (query_desc, indices, distances, param_.knn_, ::flann::SearchParams (param_.kdtree_splits_));
        else if(param_.distance_metric_==4)
            lomdb->flann_index_hellinger_->knnSearch (query_desc, indices, distances, param_.knn_, ::flann::SearchParams (param_.kdtree_splits_));
        else
            lomdb->flann_index_l1_->knnSearch (query_desc, indices, distances, param_.knn_, ::flann::SearchParams (param_.kdtree_splits_));

        if(distances[0][0] > param_.max_descriptor_distance_)
            continue;

        for (size_t i = 0; i < param_.knn_; i++)
        {
            const typename LocalObjectModelDatabase::flann_model &f = lomdb->flann_models_[ indices[0][i] ];
            float m_dist = param_.correspondence_distance_weight_ * distances[0][i];

            typename std::map<std::string, LocalObjectModel::ConstPtr >::const_iterator it = model_keypoints_.find( f.model_id_);
//            const LocalObjectModel &m_kps = *(it->second);

            KeypointIndex m_idx = f.keypoint_id_ + model_keypoint_offset;
            KeypointIndex s_idx = kp_indices[idx];
//            CHECK ( m_idx < m_kps.keypoints_->points.size() );
//            CHECK ( kp_indices[idx] < scene_->points.size() );

            typename std::map<std::string, LocalObjectHypothesis<PointT> >::iterator it_c = corrs_.find ( f.model_id_ );
            if ( it_c != corrs_.end () )
            { // append correspondences to existing ones
                pcl::CorrespondencesPtr &corrs = it_c->second.model_scene_corresp_;
                corrs->push_back( pcl::Correspondence ( m_idx, s_idx, m_dist ) );
            }
            else //create object hypothesis
            {
                LocalObjectHypothesis<PointT> new_loh;
                new_loh.model_scene_corresp_.reset (new pcl::Correspondences);
                new_loh.model_scene_corresp_->push_back( pcl::Correspondence ( m_idx, s_idx, m_dist ) );
                new_loh.model_id_ = f.model_id_;
                corrs_[ f.model_id_ ] = new_loh;
            }
        }
    }

    delete[] indices.ptr ();
    delete[] distances.ptr ();
    delete[] query_desc.ptr ();
}

template<typename PointT>
void
LocalFeatureMatcher<PointT>::featureEncoding(LocalEstimator<PointT> &est,
                                             const std::vector<KeypointIndex> &keypoint_indices,
                                             std::vector<KeypointIndex> &filtered_keypoint_indices,
                                             std::vector<FeatureDescriptor> &signatures )
{
    {
        pcl::ScopeTime t("Feature Encoding");
        est.setInputCloud(scene_);
        est.setNormals(scene_normals_);
        est.setIndices(keypoint_indices);
        est.compute (signatures);
        filtered_keypoint_indices = est.getKeypointIndices();
    }

    CHECK ( filtered_keypoint_indices.size() == signatures.size() );

    // remove signatures (with corresponding keypoints) with nan elements
    size_t kept=0;
    for(size_t sig_id=0; sig_id<signatures.size(); sig_id++)
    {
        bool keep_this = true;
        for(size_t dim=0; dim< signatures[sig_id].size(); dim++)
        {
            if( std::isnan(signatures[sig_id][dim]) || !std::isfinite(signatures[sig_id][dim]) )
            {
                keep_this = false;
                LOG(ERROR) << "DOES THIS REALLY HAPPEN?";
                break;
            }
        }

        if(keep_this)
        {
            signatures[kept] = signatures[sig_id];
            filtered_keypoint_indices[kept] = filtered_keypoint_indices[sig_id];
            kept++;
        }
    }
    filtered_keypoint_indices.resize(kept);
    signatures.resize(kept);
}

template<typename PointT>
void
LocalFeatureMatcher<PointT>::recognize ()
{
    corrs_.clear();
    keypoint_indices_.clear();

    const std::vector<KeypointIndex> keypoint_indices = extractKeypoints( indices_ );
    std::vector<KeypointIndex> filtered_kp_indices;

    if( !have_sift_estimator_) // for SIFT we do not need to extract keypoints explicitly
    {
        std::vector<int> inlier = getInlier(keypoint_indices);
        filtered_kp_indices = filterVector<KeypointIndex> (keypoint_indices, inlier);
    }

    for(size_t est_id=0; est_id < estimators_.size(); est_id++)
    {
        typename LocalEstimator<PointT>::Ptr &est = estimators_[est_id];

        std::vector<KeypointIndex> filtered_kp_indices_tmp;
        std::vector<FeatureDescriptor> signatures_tmp;
        featureEncoding( *est, filtered_kp_indices, filtered_kp_indices_tmp, signatures_tmp);

        if( have_sift_estimator_ ) // for SIFT we do not need to filter keypoints after detection (which includes kp extraction)
        {
            std::vector<int> inlier = getInlier(filtered_kp_indices_tmp);
            filtered_kp_indices_tmp = filterVector<KeypointIndex> (filtered_kp_indices_tmp, inlier);
            signatures_tmp = filterVector<FeatureDescriptor> (signatures_tmp, inlier);
        }

        if( filtered_kp_indices_tmp.empty() )
            continue;

        LOG(INFO) << "Number of " << est->getFeatureDescriptorName() << " features: " << filtered_kp_indices_tmp.size();

        if(visualize_keypoints_)
            visualizeKeypoints(filtered_kp_indices_tmp, keypoint_indices);

        featureMatching(filtered_kp_indices_tmp, signatures_tmp, lomdbs_[est_id]);
        indices_.clear();
    }
    indices_.clear();
}

//template class V4R_EXPORTS LocalFeatureMatcher<pcl::PointXYZ>;
template class V4R_EXPORTS LocalFeatureMatcher<pcl::PointXYZRGB>;
}


