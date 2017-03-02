#include <pcl_1_8/features/organized_edge_detection.h>
#include <v4r/common/miscellaneous.h>
#include <v4r/features/types.h>
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

#include <opencv2/opencv.hpp>

#include <glog/logging.h>
#include <sstream>
#include <omp.h>

namespace v4r
{

template<typename PointT>
void
LocalFeatureMatcher<PointT>::visualizeKeypoints() const
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
LocalFeatureMatcher<PointT>::filterKeypoints (bool filter_signatures)
{
    if (keypoint_indices_.empty() )
        return;

    boost::dynamic_bitset<> kp_is_kept(keypoint_indices_.size());
    kp_is_kept.set();

    if(visualize_keypoints_)
        keypoint_indices_unfiltered_ = keypoint_indices_;

    if(param_.filter_planar_)
    {
        pcl::ScopeTime tt("Computing planar keypoints");
        typename pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
        pcl::NormalEstimationOMP<PointT, pcl::Normal> normalEstimation;
        normalEstimation.setInputCloud(scene_);
        boost::shared_ptr< std::vector<int> > IndicesPtr (new std::vector<int>);
        *IndicesPtr = keypoint_indices_;
        normalEstimation.setIndices(IndicesPtr);
        normalEstimation.setRadiusSearch(param_.planar_support_radius_);
        normalEstimation.setSearchMethod(tree);
        pcl::PointCloud<pcl::Normal>::Ptr normals_for_planarity_check ( new pcl::PointCloud<pcl::Normal> );
        normalEstimation.compute(*normals_for_planarity_check);

        for(size_t i=0; i<keypoint_indices_.size(); i++)
        {
            if(normals_for_planarity_check->points[i].curvature < param_.threshold_planar_)
                kp_is_kept.reset(i);
        }
    }

    if (param_.filter_border_pts_)
    {
        pcl::ScopeTime t("Computing boundary points");
        CHECK(scene_->isOrganized());
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
        for(size_t i=0; i<keypoint_indices_.size(); i++)
        {
            int idx = keypoint_indices_[i];
            int u = idx%scene_->width;
            int v = idx/scene_->width;

            if ( boundary_mask_dilated.at<unsigned char>(v,u) )
                kp_is_kept.reset(i);
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
LocalFeatureMatcher<PointT>::extractKeypoints ()
{
    pcl::ScopeTime t("Extracting all keypoints with filtering");
    boost::dynamic_bitset<> obj_mask;
    boost::dynamic_bitset<> kp_mask ( scene_->points.size(), 0);

    if( indices_.empty() )
    {
        obj_mask.resize( scene_->points.size(), 0);
        obj_mask.set();
    }
    else
        obj_mask = createMaskFromIndices(indices_, scene_->points.size());

    for (typename KeypointExtractor<PointT>::Ptr ke : keypoint_extractor_)
    {
        ke->setInputCloud (scene_);

        if (ke->needNormals ())
            ke->setNormals (scene_normals_);

        pcl::PointCloud<PointT> detected_keypoints;
        ke->compute (detected_keypoints);

        std::vector<int> kp_indices = ke->getKeypointIndices();

        // only keep keypoints which are finite (with finite normals), are closer than the maximum allowed distance,
        // belong to the Region of Interest and are not planar (if planarity filter is on)
        for(int idx : kp_indices)
        {
            if(     obj_mask[idx] && pcl::isFinite( scene_->points[idx] ) &&
                    (!ke->needNormals() || pcl::isFinite(scene_normals_->points[idx]))
                    && scene_->points[idx].z < param_.max_keypoint_distance_z_ )
            {
                kp_mask.set( idx );
            }
        }
    }
    keypoint_indices_ = createIndicesFromMask<int>(kp_mask);
    indices_.clear();
}

template<typename PointT>
void
LocalFeatureMatcher<PointT>::initialize (const std::string &trained_dir, bool retrain)
{
    CHECK ( m_db_ );
    lomdb_.reset( new LocalObjectModelDatabase );

    std::vector<typename Model<PointT>::ConstPtr> models = m_db_->getModels ();
    std::vector<std::vector<float> > all_signatures; ///< all signatures extracted from all objects in the model database

    for( typename Model<PointT>::ConstPtr m : models)
    {
        bf::path trained_path_feat = trained_dir; // directory where feature descriptors and keypoints are stored
        trained_path_feat /= m->id_;
        trained_path_feat /= getFeatureName();

        std::vector<std::vector<float> > model_signatures;
        pcl::PointCloud<pcl::PointXYZ>::Ptr model_keypoints (new pcl::PointCloud<pcl::PointXYZ>);

        bf::path kp_path = trained_path_feat;
        kp_path /= "keypoints.pcd";
        bf::path signatures_path = trained_path_feat;
        signatures_path /= "signatures.dat";

        if( !retrain && io::existsFile( kp_path.string() ) && io::existsFile( signatures_path.string() ) )
        {
            pcl::io::loadPCDFile( kp_path.string(), *model_keypoints );
            ifstream is(signatures_path.string(), ios::binary);
            boost::archive::binary_iarchive iar(is);
            iar >> model_signatures;
            is.close();
        }
        else
        {
            const auto training_views = m->getTrainingViews();
            std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > existing_poses;

            for(const auto &tv : training_views)
            {
                std::string txt = "Training " + estimator_->getFeatureDescriptorName() + " on view " + m->class_ + "/" + m->id_ + "/" + tv->filename_;
                pcl::ScopeTime t( txt.c_str() );

                Eigen::Matrix4f pose;
                if(tv->cloud_)   // point cloud and all relevant information is already in memory (fast but needs a much memory when a lot of training views/objects)
                {
                    scene_ = tv->cloud_;
                    scene_normals_ = tv->normals_;
                    indices_ = tv->indices_;
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
                        std::cerr << "Could not read pose from file " << tv->pose_filename_ << "!" << std::endl;
                        pose = Eigen::Matrix4f::Identity();
                    }

                    scene_ = cloud;

                    if ( this->needNormals() )
                    {
                        scene_normals_.reset( new pcl::PointCloud<pcl::Normal> );
                        pcl::IntegralImageNormalEstimation<PointT, pcl::Normal> ne;
                        ne.setNormalEstimationMethod (ne.AVERAGE_3D_GRADIENT);
                        ne.setMaxDepthChangeFactor(0.02f);
                        ne.setNormalSmoothingSize(10.0f);
                        ne.setInputCloud(scene_);
                        pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
                        ne.compute(*normals);
                        scene_normals_ = normals;
                    }

                    // read object mask from file
                    indices_.clear();
                    std::ifstream mi_f ( tv->indices_filename_ );
                    int idx;
                    while ( mi_f >> idx )
                       indices_.push_back(idx);
                    mi_f.close();
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
                    if( ! computeFeatures() )
                        continue;

                    existing_poses.push_back(pose);
                    std::cout << "Adding " << scene_signatures_.size() << " " << this->getFeatureName()<< " descriptors to the model database. " << std::endl;

                    assert(scene_signatures_.size() == keypoint_indices_.size());

                    pcl::PointCloud<pcl::PointXYZ> model_keypoints_tmp;
                    pcl::copyPointCloud( *scene_, keypoint_indices_, model_keypoints_tmp );
                    pcl::transformPointCloud(model_keypoints_tmp, model_keypoints_tmp, pose);
                    *model_keypoints += model_keypoints_tmp;
                    model_signatures.insert(model_signatures.end(), scene_signatures_.begin(), scene_signatures_.end());

                    indices_.clear();
                }
                else
                    std::cout << "Ignoring view " << tv->filename_ << " because a similar camera pose exists." << std::endl;
            }

            io::createDirForFileIfNotExist( kp_path.string() );
            pcl::io::savePCDFileBinaryCompressed ( kp_path.string(), *model_keypoints);
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
        lomdb_->flann_models_.insert ( lomdb_->flann_models_.end(), flann_models_tmp.begin(), flann_models_tmp.end() );

        LocalObjectModel::Ptr lom (new LocalObjectModel );
        lom->keypoints_ = model_keypoints;
        lomdb_->l_obj_models_[m->id_] = lom;
    }

    lomdb_->flann_data_.reset ( new flann::Matrix<float> (
                            new float[ all_signatures.size() * all_signatures[0].size()],
            all_signatures.size(), all_signatures[0].size()));

    for (size_t i = 0; i < lomdb_->flann_data_->rows; i++)
        for (size_t j = 0; j < lomdb_->flann_data_->cols; j++)
            lomdb_->flann_data_->ptr()[i * lomdb_->flann_data_->cols + j] = all_signatures[i][j];

    std::cout << "Building the kdtree index for " << lomdb_->flann_data_->rows << " elements." << std::endl;

    if(param_.distance_metric_==2)
    {
        lomdb_->flann_index_l2_.reset( new ::flann::Index<::flann::L2<float> > (*(lomdb_->flann_data_), ::flann::KDTreeIndexParams (param_.kdtree_num_trees_)));
//        lomdb_->flann_index_l2_.reset( new flann::Index<flann::L2<float> > (*(lomdb_->flann_data_), flann::LinearIndexParams()));
        lomdb_->flann_index_l2_->buildIndex();
    }
    else
    {
        lomdb_->flann_index_l1_.reset( new ::flann::Index<::flann::L1<float> > (*(lomdb_->flann_data_), ::flann::KDTreeIndexParams (param_.kdtree_num_trees_)));
//        lomdb_->flann_index_l1_.reset( new flann::Index<flann::L1<float> > (*(lomdb_->flann_data_), flann::LinearIndexParams()));
        lomdb_->flann_index_l1_->buildIndex();
    }

    indices_.clear();
}

template<typename PointT>
void
LocalFeatureMatcher<PointT>::featureMatching()
{
    CHECK (scene_signatures_.size () == keypoint_indices_.size() );

    std::cout << "computing " << scene_signatures_.size () << " matches." << std::endl;

    int size_feat = scene_signatures_[0].size();

    ::flann::Matrix<float> distances (new float[param_.knn_], 1, param_.knn_);
    ::flann::Matrix<int> indices (new int[param_.knn_], 1, param_.knn_);
    ::flann::Matrix<float> query_desc (new float[size_feat], 1, size_feat);

    for (size_t idx = 0; idx < scene_signatures_.size (); idx++)
    {
        memcpy (&query_desc.ptr()[0], &scene_signatures_[idx][0], size_feat * sizeof(float));

        if(param_.distance_metric_==2)
            lomdb_->flann_index_l2_->knnSearch (query_desc, indices, distances, param_.knn_, ::flann::SearchParams (param_.kdtree_splits_));
        else
            lomdb_->flann_index_l1_->knnSearch (query_desc, indices, distances, param_.knn_, ::flann::SearchParams (param_.kdtree_splits_));

        if(distances[0][0] > param_.max_descriptor_distance_)
            continue;

        for (size_t i = 0; i < param_.knn_; i++)
        {
            const typename LocalObjectModelDatabase::flann_model &f = lomdb_->flann_models_[ indices[0][i] ];
            float m_dist = param_.correspondence_distance_weight_ * distances[0][i];

            auto it_c = corrs_.find ( f.model_id_ );
            if ( it_c != corrs_.end () )
            { // append correspondences to existing ones
                pcl::CorrespondencesPtr &corrs = it_c->second.model_scene_corresp_;
                corrs->push_back( pcl::Correspondence ( (int)f.keypoint_id_, keypoint_indices_[idx], m_dist ) );
            }
            else //create object hypothesis
            {
                LocalObjectHypothesis<PointT> new_loh;
                new_loh.model_scene_corresp_.reset (new pcl::Correspondences);
                new_loh.model_scene_corresp_->push_back( pcl::Correspondence ( (int)f.keypoint_id_, keypoint_indices_[idx], m_dist ) );
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
LocalFeatureMatcher<PointT>::featureEncoding()
{
    {
        pcl::ScopeTime t("Feature Encoding");
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
LocalFeatureMatcher<PointT>::computeFeatures()
{
    scene_signatures_.resize(0);
    keypoint_indices_.clear();

    CHECK (scene_);

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
LocalFeatureMatcher<PointT>::recognize ()
{
    corrs_.clear();
    keypoint_indices_.clear();

    if(!computeFeatures())
    {
        indices_.clear();
        return;
    }

    std::cout << "Number of " << estimator_->getFeatureDescriptorName() << " features: " << keypoint_indices_.size() << std::endl;

    if(visualize_keypoints_)
        visualizeKeypoints();

    featureMatching();
    indices_.clear();
}

//template class V4R_EXPORTS LocalFeatureMatcher<pcl::PointXYZ>;
template class V4R_EXPORTS LocalFeatureMatcher<pcl::PointXYZRGB>;
}


