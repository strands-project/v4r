#include <v4r/common/normals.h>
#include <v4r/io/eigen.h>
#include <v4r/recognition/model.h>
#include <v4r/recognition/global_recognizer.h>

#include <pcl/common/angles.h>
#include <pcl/PointIndices.h>
#include <pcl/registration/icp.h>
#include <glog/logging.h>
#include <sstream>
#include <omp.h>

namespace v4r
{
template<typename PointT>
void
GlobalRecognizer<PointT>::loadFeaturesFromDisk ()
{
    std::vector<ModelTPtr> models = source_->getModels();
    flann_models_.clear();
    flann_model descr_model;

    const std::string descr_name = estimator_->getFeatureDescriptorName();

    for (size_t m_id = 0; m_id < models.size (); m_id++)
    {
        descr_model.model = models[m_id];
        ModelT &m = *models[m_id];
        const std::string out_train_path = models_dir_  + "/" + m.class_ + "/" + m.id_ + "/" + descr_name;
        const std::string in_train_path = models_dir_  + "/" + m.class_ + "/" + m.id_ + "/views/";

        Eigen::MatrixXf model_signatures;
        m.poses_.resize(m.view_filenames_.size());
        m.view_centroid_.resize(m.view_filenames_.size(), 3);
        m.eigen_pose_alignment_.resize(m.view_filenames_.size(), Eigen::Matrix4f::Identity());
        m.elongations_.resize(m.view_filenames_.size(), 3);

        for(size_t v_id=0; v_id< m.view_filenames_.size(); v_id++)
        {
            descr_model.view_id = v_id;

            // read signature
            std::string signature_basename (m.view_filenames_[v_id]);
            boost::replace_last(signature_basename, ".pcd", ".desc");
            const std::string signature_fn = out_train_path + "/" + signature_basename;
            if(!v4r::io::existsFile(signature_fn))
                continue;

            Eigen::MatrixXf view_signatures;
            Eigen::read_binary(signature_fn.c_str(), view_signatures);

            if(view_signatures.rows()==0)
                continue;

            for(size_t i=0; i<view_signatures.rows(); i++)
            {
                flann_models_.push_back(descr_model); // TODO: pre-allocate memory.
            }

            all_model_signatures_.conservativeResize(all_model_signatures_.rows() + view_signatures.rows(), view_signatures.cols());
            all_model_signatures_.bottomRows( view_signatures.rows() ) = view_signatures;
            all_trained_model_label_.conservativeResize(all_trained_model_label_.rows() + view_signatures.rows());
            all_trained_model_label_.tail( view_signatures.rows() ) = m_id * Eigen::VectorXi::Ones( view_signatures.rows() );

            model_signatures.conservativeResize( model_signatures.rows() + view_signatures.rows(), view_signatures.cols());
            model_signatures.bottomRows( view_signatures.rows() ) = view_signatures;


            // read camera poses
            std::string pose_basename (m.view_filenames_[v_id]);
            boost::replace_first(pose_basename, "cloud_", "pose_");
            boost::replace_last(pose_basename, ".pcd", ".txt");
            const std::string pose_fn = in_train_path + "/" + pose_basename;
            if(v4r::io::existsFile(pose_fn))
            {
                m.poses_[v_id] = io::readMatrixFromFile(pose_fn);
            }

            // read covariance based pose estimates
            std::string cov_pose_basename (m.view_filenames_[v_id]);
            boost::replace_last(cov_pose_basename, ".pcd", ".cov_pose");
            const std::string cov_pose_fn = out_train_path + "/" + cov_pose_basename;
            if(v4r::io::existsFile(cov_pose_fn))
            {
                m.eigen_pose_alignment_[v_id] = io::readMatrixFromFile(cov_pose_fn);
            }
            else
                std::cerr << "Could not find covariance based pose estimation file " << cov_pose_fn << std::endl;

            // read elongation
            std::string elongations_basename (m.view_filenames_[v_id]);
            boost::replace_last(elongations_basename, ".pcd", ".elongations");
            const std::string elongations_fn = out_train_path + "/" + elongations_basename;
            if(v4r::io::existsFile(elongations_fn))
            {
                std::ifstream f(elongations_fn.c_str());
                f >> m.elongations_(v_id, 0) >> m.elongations_(v_id, 1) >> m.elongations_(v_id, 2);
                f.close();
            }
            else
                std::cerr << "Could not find elongation file " << elongations_fn << std::endl;


        }
        m.signatures_[descr_name] = model_signatures;
    }\
}

template<typename PointT>
bool
GlobalRecognizer<PointT>::featureEncoding(Eigen::MatrixXf &signatures)
{
    CHECK(estimator_);


#ifdef BLA
    typename pcl::PointCloud<PointT>::Ptr scene_roi (new pcl::PointCloud<PointT>);
    pcl::PointCloud<pcl::Normal>::Ptr scene_normals_roi (new pcl::PointCloud<pcl::Normal>);
    if( !indices_.empty())
    {
        pcl::copyPointCloud(*scene_, indices_, *scene_roi);
        estimator_->setInputCloud(scene_roi);

        if(scene_normals_)
        {
            pcl::copyPointCloud(*scene_normals_, indices_, *scene_normals_roi);
            estimator_->setNormals(scene_normals_roi);
        }
    }
    else
    {
        estimator_->setInputCloud(scene_);
        estimator_->setNormals(scene_normals_);
    }
#else
    estimator_->setInputCloud(scene_);
    estimator_->setNormals(scene_normals_);
    if(!indices_.empty())
        estimator_->setIndices(indices_);
#endif

    return estimator_->compute(signatures);
}

template<typename PointT>
bool
GlobalRecognizer<PointT>::initialize(bool force_retrain)
{
    CHECK(estimator_);
    CHECK(source_);
    const std::string descr_name = estimator_->getFeatureDescriptorName();
    std::vector<ModelTPtr> models = source_->getModels();

    std::cout << "Models size:" << models.size () << std::endl;

    if (force_retrain)
    {
        for (size_t i = 0; i < models.size (); i++)
            source_->removeDescDirectory (*models[i], models_dir_, descr_name);
    }

    for (ModelTPtr &m : models)
    {
        const std::string dir = models_dir_ + "/" + m->class_ + "/" + m->id_ + "/" + descr_name;

        bool view_is_already_trained = false;
        if ( io::existsFolder(dir) )   // check if training directory exists and the number of descriptors is equal to the number of views
        {
            std::vector<std::string> descriptor_files = io::getFilesInDirectory(dir, ".*.desc", false);
            if(descriptor_files.size()== m->view_filenames_.size())
                view_is_already_trained = true;
        }

        if ( !view_is_already_trained )
        {
            std::cout << "Model " << m->class_ << " " << m->id_ << " not trained. Training " << descr_name << " on " << m->view_filenames_.size () << " views..." << std::endl;
            io::createDirIfNotExist(dir);

            if(!source_->getLoadIntoMemory())
                source_->loadInMemorySpecificModel(*m);

            LOG(INFO) << "Computing signatures for " << m->class_ << " for id " <<  m->id_ << " with " << m->views_.size() << " views.";

            for (size_t view_id = 0; view_id < m->view_filenames_.size(); view_id++)
            {
                std::stringstream foo; foo << "processing view " << view_id;
                scene_ = m->views_[view_id];
                if ( this->needNormals() && !scene_normals_ )
                {
                    scene_normals_.reset(new pcl::PointCloud<pcl::Normal>);
                    {
                        pcl::ScopeTime tt("Computing scene normals");
                        computeNormals<PointT>(scene_, scene_normals_, 2);
                    }
                }
                indices_ = m->indices_[view_id];
                Eigen::MatrixXf signature_tmp;
                if (!featureEncoding(signature_tmp))
                    continue;

                // write signatures and keypoints (with normals) to files
                std::string signature_basename (m->view_filenames_[view_id]);
                boost::replace_last(signature_basename, ".pcd", ".desc");
                Eigen::write_binary(dir+"/"+signature_basename, signature_tmp);


                // write centroid to disk
                Eigen::Vector4f centroid;
                pcl::compute3DCentroid(*m->views_[view_id], centroid);
                std::vector<float> centroid_v(3);
                centroid_v[0] = centroid[0];
                centroid_v[1] = centroid[1];
                centroid_v[2] = centroid[2];

                std::string centroid_basename (m->view_filenames_[view_id]);
                boost::replace_last(centroid_basename, ".pcd", ".centroid");
                io::writeVectorToFile (dir+"/"+centroid_basename, centroid_v);


                // write entropy to disk ///NOTE: This is not implemented right now
                std::stringstream path_entropy;
                path_entropy << dir << "/entropy_" << view_id << ".txt";
                io::writeFloatToFile (path_entropy.str (), m->self_occlusions_[view_id]);


                // write estimated objects orientation(*) to disk (*per view; based on covariance matrix)
                EIGEN_ALIGN16 Eigen::Matrix3f covariance_matrix;
                Eigen::Vector4f centroid_scene_cluster;
                EIGEN_ALIGN16 Eigen::Vector3f eigenValues;
                EIGEN_ALIGN16 Eigen::Matrix3f eigenVectors, eigenBasis;
                computeMeanAndCovarianceMatrix (*scene_, indices_, covariance_matrix, centroid_scene_cluster);
                pcl::eigen33 (covariance_matrix, eigenVectors, eigenValues);

                // create orthonormal rotation matrix from eigenvectors
                eigenBasis.col(0) = eigenVectors.col(0).normalized();
                float dotp12 = eigenVectors.col(1).dot(eigenBasis.col(0));
                Eigen::Vector3f eig2 = eigenVectors.col(1) - dotp12 * eigenBasis.col(0);
                eigenBasis.col(1) = eig2.normalized();
                Eigen::Vector3f eig3 = eigenBasis.col(0).cross ( eigenBasis.col(1) );
                eigenBasis.col(2) = eig3.normalized();

                // transform cluster into origin and align with eigenvectors
                Eigen::Matrix4f tf_rot = Eigen::Matrix4f::Identity();
                Eigen::Matrix4f tf_trans = Eigen::Matrix4f::Identity();
                tf_trans.block<3,1>(0,3) = -centroid_scene_cluster.topRows(3);
                tf_rot.block<3,3>(0,0) = eigenBasis.transpose();

                std::string covariance_pose (m->view_filenames_[view_id]);
                boost::replace_last(covariance_pose, ".pcd", ".cov_pose");
                io::writeMatrixToFile( dir+"/"+covariance_pose, tf_rot*tf_trans);

                // compute max elongations
                typename pcl::PointCloud<PointT>::Ptr eigenvec_aligned(new pcl::PointCloud<PointT>);
                pcl::copyPointCloud(*scene_, indices_, *eigenvec_aligned);
                pcl::transformPointCloud(*eigenvec_aligned, *eigenvec_aligned, tf_rot*tf_trans);

                float xmin,ymin,xmax,ymax,zmin,zmax;
                xmin = ymin = xmax = ymax = zmin = zmax = 0.f;
                for(size_t pt=0; pt<eigenvec_aligned->points.size(); pt++)
                {
                    const PointT &p = eigenvec_aligned->points[pt];
                    if(p.x < xmin)
                        xmin = p.x;
                    if(p.x > xmax)
                        xmax = p.x;
                    if(p.y < ymin)
                        ymin = p.y;
                    if(p.y > ymax)
                        ymax = p.y;
                    if(p.z < zmin)
                        zmin = p.z;
                    if(p.z > zmax)
                        zmax = p.z;
                }

                std::string elongationsbn (m->view_filenames_[view_id]);
                boost::replace_last(elongationsbn, ".pcd", ".elongations");
                std::ofstream f(dir+"/"+elongationsbn);
                f<< xmax - xmin << " " << ymax - ymin << " " << zmax - zmin;
                f.close();
            }

            if(!source_->getLoadIntoMemory())
                m->views_.clear();
        }
        else
            LOG(INFO) << "Model " << m->class_ << " with id " <<  m->id_ << " (" << m->views_.size() << " views) has already been trained.";
    }

    loadFeaturesFromDisk();
    classifier_->train(all_model_signatures_, all_trained_model_label_);
    all_model_signatures_.resize(0,0);  // not needed here any more
    all_trained_model_label_.resize(0);
    return true;
}


template<typename PointT>
void
GlobalRecognizer<PointT>::computeEigenBasis()
{
    centroids_.resize( clusters_.size() );
    eigen_basis_.resize( clusters_.size() );
    elongations_.resize( clusters_.size() );

    for(size_t i=0; i<clusters_.size(); i++)
    {
        EIGEN_ALIGN16 Eigen::Matrix3f covariance_matrix;
        EIGEN_ALIGN16 Eigen::Vector3f eigenValues;
        EIGEN_ALIGN16 Eigen::Matrix3f eigenVectors;
        computeMeanAndCovarianceMatrix (*scene_, clusters_[i], covariance_matrix, centroids_[i]);
        pcl::eigen33 (covariance_matrix, eigenVectors, eigenValues);

        Eigen::Matrix3f &eigenBasis = eigen_basis_[i];

        // create orthonormal rotation matrix from eigenvectors
        eigenBasis.col(0) = eigenVectors.col(0).normalized();
        float dotp12 = eigenVectors.col(1).dot(eigenBasis.col(0));
        Eigen::Vector3f eig2 = eigenVectors.col(1) - dotp12 * eigenBasis.col(0);
        eigenBasis.col(1) = eig2.normalized();
        Eigen::Vector3f eig3 = eigenBasis.col(0).cross ( eigenBasis.col(1) );
        eigenBasis.col(2) = eig3.normalized();

        // transform cluster into origin and align with eigenvectors
        Eigen::Matrix4f tf_rot_inv = Eigen::Matrix4f::Identity();
        tf_rot_inv.block<3,3>(0,0) = eigenBasis.transpose();
        Eigen::Matrix4f tf_trans_inv = Eigen::Matrix4f::Identity();
        tf_trans_inv.block<3,1>(0,3) = -centroids_[i].topRows(3);

        Eigen::Matrix4f tf_trans = Eigen::Matrix4f::Identity();
        tf_trans.block<3,1>(0,3) = centroids_[i].topRows(3);
        Eigen::Matrix4f tf_rot = tf_rot_inv.inverse();

        // compute max elongations
        typename pcl::PointCloud<PointT>::Ptr eigenvec_aligned(new pcl::PointCloud<PointT>);
        pcl::copyPointCloud(*scene_, clusters_[i], *eigenvec_aligned);
        pcl::transformPointCloud(*eigenvec_aligned, *eigenvec_aligned, tf_rot_inv*tf_trans_inv);

        float xmin,ymin,xmax,ymax,zmin,zmax;
        xmin = ymin = xmax = ymax = zmin = zmax = 0.f;
        for(size_t pt=0; pt<eigenvec_aligned->points.size(); pt++)
        {
            const PointT &p = eigenvec_aligned->points[pt];
            if(p.x < xmin)
                xmin = p.x;
            if(p.x > xmax)
                xmax = p.x;
            if(p.y < ymin)
                ymin = p.y;
            if(p.y > ymax)
                ymax = p.y;
            if(p.z < zmin)
                zmin = p.z;
            if(p.z > zmax)
                zmax = p.z;
        }

        elongations_[i](0) = xmax - xmin;
        elongations_[i](1) = ymax - ymin;
        elongations_[i](2) = zmax - zmin;
    }
}

template<typename PointT>
void
GlobalRecognizer<PointT>::featureMatching(const Eigen::MatrixXf &query_sig,
                                          int cluster_id,
                                          std::vector<ModelTPtr> &matched_models,
                                          std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &transforms,
                                          std::vector<float> &distance)
{
    if(query_sig.cols() == 0 || query_sig.rows() == 0)
        return;

    std::vector<ModelTPtr> models = source_->getModels();

    Eigen::MatrixXi knn_indices;
    Eigen::MatrixXf knn_distances;
    Eigen::MatrixXi predicted_label;
    classifier_->predict(query_sig, predicted_label);

    if(classifier_->getType() == ClassifierType::KNN)
        classifier_->getTrainingSampleIDSforPredictions(knn_indices, knn_distances);

    Eigen::Matrix4f tf_trans = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f tf_rot = Eigen::Matrix4f::Identity();
    Eigen::Vector4f centroid;

    if(param_.use_table_plane_for_alignment_)   // we do not need to know the closest training view
    {
        float dist = centroids_[cluster_id].dot( table_plane_ );
        centroid = centroids_[cluster_id] - dist * table_plane_;

        // create some arbitrary coordinate system on table plane (s.t. normal corresponds to z axis, and others are orthonormal)
        Eigen::Vector3f vec_z = table_plane_.topRows(3);
        vec_z.normalize();

        Eigen::Vector3f dummy; ///NOTE we just need to find any other point on the plane except centroid to create a coordinate system (hopefully this one is not close to zero)
        dummy(0) = 1; dummy(1) = 0; dummy(2) = 0;
        Eigen::Vector3f vec_x = vec_z.cross(dummy);
        vec_x.normalize();

        Eigen::Vector3f vec_y = vec_z.cross(vec_x);
        vec_y.normalize();

        Eigen::Matrix3f rotation_basis;
        rotation_basis.col(0) = vec_x;
        rotation_basis.col(1) = vec_y;
        rotation_basis.col(2) = vec_z;

        Eigen::Matrix4f tf_rot_inv = Eigen::Matrix4f::Identity();
        tf_rot_inv.block<3,3>(0,0) = rotation_basis.transpose();
        tf_rot = tf_rot_inv.inverse();
    }
    else    // use eigenvectors for alignment
    {
        centroid = centroids_[cluster_id];
    }
    tf_trans.block<3,1>(0,3) = centroid.topRows(3);

    // pre-allocate memory
    size_t max_hypotheses = predicted_label.rows() * predicted_label.cols();

    if(param_.use_table_plane_for_alignment_)
        max_hypotheses *= (int)(360.f / param_.z_angle_sampling_density_degree_);
    else
        max_hypotheses *= 4;

    matched_models.resize( max_hypotheses  );
    distance.resize( max_hypotheses );
    transforms.resize( max_hypotheses );

    size_t kept=0;
    for(int query_id=0; query_id<predicted_label.rows(); query_id++)
    {
        for (size_t k = 0; k < predicted_label.cols(); k++)
        {
            ModelTPtr m;
            if(param_.use_table_plane_for_alignment_)
            {
                m = models[ predicted_label(query_id, k) ];

                const Eigen::Vector3f &elongations_model = m->elongations_.colwise().maxCoeff();    // as we don't know the view, we just take the maximum extent of each axis over all training views
                const Eigen::Vector3f &elongations = elongations_[cluster_id];
                if( param_.check_elongations_ &&
                   (elongations(2)/elongations_model(2) < param_.min_elongation_ratio_||
                    elongations(2)/elongations_model(2) > param_.max_elongation_ratio_||
                    elongations(1)/elongations_model(1) < param_.min_elongation_ratio_||
                    elongations(1)/elongations_model(1) > param_.max_elongation_ratio_) )
                    continue;

                for(double rot_i=0.f; rot_i<360.f; rot_i+=param_.z_angle_sampling_density_degree_)
                {
                    matched_models[kept] = m;

                    double rot_rad = pcl::deg2rad(rot_i);
                    Eigen::Matrix4f rot_tmp = Eigen::Matrix4f::Identity();
                    rot_tmp(0,0) =  cos(rot_rad);
                    rot_tmp(0,1) = -sin(rot_rad);
                    rot_tmp(1,0) =  sin(rot_rad);
                    rot_tmp(1,1) =  cos(rot_rad);
                    transforms[kept] = tf_trans * tf_rot  * rot_tmp;

                    distance[kept] =  0.f;
                    kept++;
                }
            }
            else
            {
                const flann_model &f = flann_models_ [ knn_indices( query_id, k ) ];
                m = f.model;

                const Eigen::Vector3f &elongations_model = m->elongations_.row( f.view_id);
                const Eigen::Vector3f &elongations = elongations_[cluster_id];
                if( param_.check_elongations_ &&
                   (elongations(2)/elongations_model(2) < param_.min_elongation_ratio_||
                    elongations(2)/elongations_model(2) > param_.max_elongation_ratio_||
                    elongations(1)/elongations_model(1) < param_.min_elongation_ratio_||
                    elongations(1)/elongations_model(1) > param_.max_elongation_ratio_) )
                    continue;

                // there are four possibilites (due to sign ambiguity of eigenvector)
                Eigen::Matrix3f eigenBasis, sign_operator;
                Eigen::Matrix3f identity = Eigen::Matrix3f::Identity();

                // once take eigen vector as they are computed
                sign_operator = identity;
                eigenBasis = eigen_basis_[cluster_id] * sign_operator;
                Eigen::Matrix4f tf_rot_inv = Eigen::Matrix4f::Identity();
                tf_rot_inv.block<3,3>(0,0) = eigenBasis.transpose();
                tf_rot = tf_rot_inv.inverse();
                matched_models[kept] = f.model;
                Eigen::Matrix4f tf_m_inv = m->poses_[f.view_id].inverse();
                transforms[kept] = tf_trans * tf_rot * f.model->eigen_pose_alignment_[f.view_id] * tf_m_inv;
                distance[kept] = knn_distances( query_id, k );
                kept++;

                // now take the first one negative
                sign_operator = identity;
                sign_operator(0,0) = -1;
                sign_operator(2,2) = -1;   // due to right-hand rule
                eigenBasis = eigen_basis_[cluster_id] * sign_operator;
                tf_rot_inv = Eigen::Matrix4f::Identity();
                tf_rot_inv.block<3,3>(0,0) = eigenBasis.transpose();
                tf_rot = tf_rot_inv.inverse();
                matched_models[kept] = f.model;
                transforms[kept] = tf_trans * tf_rot * f.model->eigen_pose_alignment_[f.view_id] * tf_m_inv;
                distance[kept] = knn_distances( query_id, k );
                kept++;


                // now take the second one negative
                sign_operator = identity;
                sign_operator(1,1) = -1;
                sign_operator(2,2) = -1;   // due to right-hand rule
                eigenBasis = eigen_basis_[cluster_id] * sign_operator;
                tf_rot_inv = Eigen::Matrix4f::Identity();
                tf_rot_inv.block<3,3>(0,0) = eigenBasis.transpose();
                tf_rot = tf_rot_inv.inverse();
                matched_models[kept] = f.model;
                transforms[kept] = tf_trans * tf_rot * f.model->eigen_pose_alignment_[f.view_id] * tf_m_inv;
                distance[kept] = knn_distances( query_id, k );
                kept++;


                // and last take first and second one negative
                sign_operator = identity;
                sign_operator(0,0) = -1;
                sign_operator(1,1) = -1;
                eigenBasis = eigen_basis_[cluster_id] * sign_operator;
                tf_rot_inv = Eigen::Matrix4f::Identity();
                tf_rot_inv.block<3,3>(0,0) = eigenBasis.transpose();
                tf_rot = tf_rot_inv.inverse();
                matched_models[kept] = f.model;
                transforms[kept] = tf_trans * tf_rot * f.model->eigen_pose_alignment_[f.view_id] * tf_m_inv;
                distance[kept] = knn_distances( query_id, k );
                kept++;
            }
        }
    }

    matched_models.resize( kept );
    distance.resize( kept );
    transforms.resize( kept );
}

template<typename PointT>
void
GlobalRecognizer<PointT>::poseRefinement()
{

#pragma omp paralell for schedule(dynamic)
    for(size_t i=0; i<models_.size(); i++)
    {
        ModelT &m = *models_[i];
        Eigen::Matrix4f &tf = transforms_[i];

        typename pcl::PointCloud<PointT>::Ptr model_aligned ( new pcl::PointCloud<PointT>() );
        typename pcl::PointCloud<PointT>::ConstPtr model_cloud = m.getAssembled( 3 );
        pcl::transformPointCloud( *model_cloud, *model_aligned, tf);

        pcl::IterativeClosestPoint<PointT, PointT> icp;
        icp.setInputSource(model_aligned);
        icp.setInputTarget(scene_);
        icp.setMaximumIterations(param_.icp_iterations_);
        pcl::PointCloud<PointT> aligned_visible_model;
        icp.align(aligned_visible_model);
        if(icp.hasConverged())
            tf = icp.getFinalTransformation() * tf;
        else
            std::cout << "ICP did not converge." << std::endl;
    }
}

template<typename PointT>
void
GlobalRecognizer<PointT>::recognize()
{
    CHECK(seg_);
    models_.clear();
    transforms_.clear();

    seg_->setInputCloud(scene_);
    seg_->setNormalsCloud(scene_normals_);
    seg_->segment();
    seg_->getSegmentIndices(clusters_);
    table_plane_ = seg_->getTablePlane();

    computeEigenBasis();

    size_t feat_dimensions = estimator_->getFeatureDimensions();
//    signatures_.resize(clusters_.size(), feat_dimensions);

    for(size_t i=0; i<clusters_.size(); i++)
    {
        indices_ = clusters_[i].indices;
        Eigen::MatrixXf signature_tmp;
        featureEncoding( signature_tmp );

        std::vector<ModelTPtr> models_tmp;
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transforms_tmp;
        std::vector<float> distance;
        featureMatching( signature_tmp, i, models_tmp, transforms_tmp, distance);

        models_.insert(models_.end(), models_tmp.begin(), models_tmp.end());
        transforms_.insert(transforms_.end(), transforms_tmp.begin(), transforms_tmp.end());


        if (param_.visualize_clusters_)
        {
            models_per_cluster_.push_back(models_tmp);
            dist_models_per_cluster_.push_back(distance);
        }
//        signatures_.row(i) = signature_tmp;
    }
//    std::ofstream f("/tmp/query_sig.txt");
//    f<<signatures_<<std::endl;
//    f.close();
    if (param_.icp_iterations_)
        poseRefinement();

    if (param_.visualize_clusters_)
    {
        visualize();
        models_per_cluster_.clear();
        dist_models_per_cluster_.clear();
    }

    indices_.clear();
}


template<typename PointT>
void
GlobalRecognizer<PointT>::visualize()
{
    if(!vis_)
    {
        vis_.reset ( new pcl::visualization::PCLVisualizer("Global recognition results") );
        vis_->createViewPort(0,0,0.5,1,vp1_);
        vis_->createViewPort(0.5,0,1,1,vp2_);
        vis_->setBackgroundColor(1,1,1,vp1_);
        vis_->setBackgroundColor(1,1,1,vp2_);
    }
    vis_->removeAllPointClouds();
    vis_->removeAllShapes();
    vis_->addPointCloud(scene_, "cloud", vp1_);


    typename pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud (new pcl::PointCloud<pcl::PointXYZRGB>());

    Eigen::Matrix3Xf rgb_cluster_colors(3, clusters_.size());
    for(size_t i=0; i < clusters_.size(); i++)
    {
        rgb_cluster_colors(0, i) = rand()%255;
        rgb_cluster_colors(1, i) = rand()%255;
        rgb_cluster_colors(2, i) = rand()%255;
    }

    for(size_t i=0; i < clusters_.size(); i++)
    {
        pcl::PointCloud<pcl::PointXYZRGB> cluster;
        pcl::copyPointCloud(*scene_, clusters_[i], cluster);
        for(size_t pt_id=0; pt_id<cluster.points.size(); pt_id++)
        {
            cluster.points[pt_id].r = rgb_cluster_colors(0, i);
            cluster.points[pt_id].g = rgb_cluster_colors(1, i);
            cluster.points[pt_id].b = rgb_cluster_colors(2, i);
        }
        *colored_cloud += cluster;
    }

    size_t disp_id=0;
    for(size_t i=0; i < models_per_cluster_.size(); i++)
    {
        for(size_t k=0; k<models_per_cluster_[i].size(); k++)
        {
            const ModelT &m = *models_per_cluster_[i][k];
            std::stringstream model_id; model_id << m.id_ << ": " << dist_models_per_cluster_[i][k];
            std::stringstream unique_id; unique_id << i << "_" << k;
//            vis_->addText(model_id.str(), 12, 12 + 12*disp_id, 10,
//                          rgb_cluster_colors(0, i)/255.f,
//                          rgb_cluster_colors(1, i)/255.f,
//                          rgb_cluster_colors(2, i)/255.f,
//                          unique_id.str(), vp2_);
            disp_id++;
        }
    }
    vis_->addPointCloud(colored_cloud,"segments", vp2_);
    vis_->spin();
}

template class V4R_EXPORTS GlobalRecognizer<pcl::PointXYZ>;
template class V4R_EXPORTS GlobalRecognizer<pcl::PointXYZRGB>;

}

