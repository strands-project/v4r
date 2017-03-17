#include <v4r/io/eigen.h>
#include <v4r/io/filesystem.h>
#include <v4r/recognition/global_recognizer.h>
#include <v4r/segmentation/plane_utils.h>

#include <boost/filesystem.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include <pcl/common/angles.h>
#include <pcl/common/time.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/integral_image_normal.h>

#include <glog/logging.h>
#include <sstream>
#include <omp.h>

namespace v4r
{

template<typename PointT>
bool
GlobalRecognizer<PointT>::featureEncoding(Eigen::MatrixXf &signature)
{
    CHECK(estimator_);
    estimator_->setInputCloud(scene_);
    estimator_->setNormals(scene_normals_);

    if( !cluster_->indices_.empty() )
        estimator_->setIndices(cluster_->indices_);

    return estimator_->compute(signature);
}

template<typename PointT>
void
GlobalRecognizer<PointT>::initialize(const std::string &trained_dir, bool retrain)
{
    CHECK(estimator_);
    CHECK(m_db_);

    Eigen::MatrixXf all_model_signatures; ///< all signatures extracted from all objects in the model database
    Eigen::VectorXi all_trained_model_labels; ///< target label for each model signature

    std::vector<typename Model<PointT>::ConstPtr> models = m_db_->getModels ();

    LOG(INFO) << "Models size:" << models.size ();

    for ( const typename Model<PointT>::ConstPtr m : models )
    {
        std::string label_name;
        size_t target_id = 0;

        if ( param_.classify_instances_ )
            label_name = m->id_;
        else
            label_name = m->class_;

        bool label_exists = false;
        size_t lbl_id_tmp;
        for(lbl_id_tmp=0; lbl_id_tmp<id_to_model_name_.size(); lbl_id_tmp++)
        {
            if( id_to_model_name_[lbl_id_tmp].compare( label_name ) == 0 )
            {
                label_exists = true;
                break;
            }
        }
        if(label_exists)
            target_id = lbl_id_tmp;
        else
        {
            target_id = id_to_model_name_.size();
            id_to_model_name_.push_back( label_name );
        }


        GlobalObjectModel::Ptr gom (new GlobalObjectModel );

        bf::path trained_path_feat = trained_dir; // directory where feature descriptors and keypoints are stored
        trained_path_feat /= m->class_;
        trained_path_feat /= m->id_;
        trained_path_feat /= getFeatureName();

        bf::path signatures_path = trained_path_feat;
        signatures_path /= "signatures.dat";

        if( retrain || !io::existsFile( signatures_path.string() ) )
        {
            const auto training_views = m->getTrainingViews();

            for(const auto &tv : training_views)
            {
                std::string txt = "Training " + estimator_->getFeatureDescriptorName() + " on view " + m->class_ + "/" + m->id_ + "/" + tv->filename_;
                pcl::ScopeTime t( txt.c_str() );

                Eigen::Matrix4f pose;
                std::vector<int> indices;
                if(tv->cloud_)   // point cloud and all relevant information is already in memory (fast but needs a much memory when a lot of training views/objects)
                {
                    scene_ = tv->cloud_;
                    scene_normals_ = tv->normals_;
                    indices = tv->indices_;
                    pose = tv->pose_;
                }
                else
                {
                    typename pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
                    pcl::io::loadPCDFile(tv->filename_, *cloud);
                    scene_ = cloud;

                    // read pose from file (if exists)
                    try
                    {
                        pose = io::readMatrixFromFile(tv->pose_filename_);
                    }
                    catch (const std::runtime_error &e)
                    {
                        std::cerr << "Could not read pose from file " << tv->pose_filename_ << "!" << std::endl;
                        pose = Eigen::Matrix4f::Identity();
                    }

                    // read object mask from file
                    std::ifstream mi_f ( tv->indices_filename_ );
                    int idx;
                    while ( mi_f >> idx )
                       indices.push_back(idx);
                    mi_f.close();
                }

                if ( !scene_normals_ && this->needNormals() )
                {
                    scene_normals_.reset( new pcl::PointCloud<pcl::Normal> );
                    pcl::IntegralImageNormalEstimation<PointT, pcl::Normal> ne;
                    ne.setNormalEstimationMethod ( ne.COVARIANCE_MATRIX );
                    ne.setMaxDepthChangeFactor(0.02f);
                    ne.setNormalSmoothingSize(10.0f);
                    ne.setInputCloud(scene_);
                    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
                    ne.compute(*normals);
                    scene_normals_ = normals;
                }

                bool similar_pose_exists = false;
                for(const Eigen::Matrix4f &ep : gom->model_poses_)
                {
                    Eigen::Vector3f v1 = pose.block<3,1>(0,0);
                    Eigen::Vector3f v2 = ep.block<3,1>(0,0);
                    v1.normalize();
                    v2.normalize();
                    float dotp = v1.dot(v2);
                    const Eigen::Vector3f crossp = v1.cross(v2);

                    float rel_angle_deg = pcl::rad2deg( acos(dotp) );
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
                    cluster_.reset ( new Cluster(*scene_, indices) );

                    Eigen::MatrixXf signature_tmp;
                    if (!featureEncoding(signature_tmp))
                        continue;

                    CHECK (signature_tmp.rows() == 1);

                    gom->model_poses_.push_back(pose);
                    gom->model_signatures_.conservativeResize( gom->model_elongations_.rows() + signature_tmp.rows(), signature_tmp.cols());
                    gom->model_signatures_.bottomRows( signature_tmp.rows() ) = signature_tmp;
                    gom->model_elongations_.conservativeResize( gom->model_elongations_.rows() + 1, 3);
                    gom->model_centroids_.conservativeResize( gom->model_centroids_.rows() + 1, 4);

                    gom->eigen_based_pose_.push_back( cluster_->eigen_pose_alignment_ );
                    gom->model_centroids_.bottomRows(1) = cluster_->centroid_.transpose();
                    gom->model_elongations_.bottomRows(1) = cluster_->elongation_.transpose();
                }
                else
                    LOG(INFO) << "Ignoring view " << tv->filename_ << " because a similar camera pose exists.";


                cluster_.reset();
                scene_.reset();
                scene_normals_.reset();
            }

            CHECK( (gom->model_elongations_.rows() == gom->model_centroids_.rows()) &&
                   (gom->model_elongations_.rows() == gom->model_signatures_.rows())     );

            Eigen::VectorXf view_centroid_to_3d_model_centroid (gom->model_centroids_.rows());
            for(int view_id=0; view_id < gom->model_centroids_.rows(); view_id++)
            {
                const Eigen::Vector4f view_centroid = gom->model_centroids_.row(view_id).transpose();
                const Eigen::Vector4f view_centroid_aligned = gom->model_poses_[view_id] * view_centroid;

                view_centroid_to_3d_model_centroid(view_id) = (view_centroid_aligned - m->centroid_).head(3).norm();
            }
            gom->mean_distance_view_centroid_to_3d_model_centroid_ = view_centroid_to_3d_model_centroid.mean();

            io::createDirForFileIfNotExist( signatures_path.string() );
            ofstream os( signatures_path.string() , ios::binary);
            boost::archive::binary_oarchive oar(os);
            oar << gom;
            os.close();
        }

        ifstream is(signatures_path.string(), ios::binary);
        boost::archive::binary_iarchive iar(is);
        iar >> gom;
        is.close();

        gomdb_.global_models_[ m->id_ ] = gom;

        if( gom->model_signatures_.rows() > 0 )
        {
            all_model_signatures.conservativeResize(all_model_signatures.rows() + gom->model_signatures_.rows(), gom->model_signatures_.cols());
            all_model_signatures.bottomRows( gom->model_signatures_.rows() ) = gom->model_signatures_;
            all_trained_model_labels.conservativeResize(all_trained_model_labels.rows() + gom->model_signatures_.rows());
            all_trained_model_labels.tail( gom->model_signatures_.rows() ) = target_id * Eigen::VectorXi::Ones( gom->model_signatures_.rows() );

            std::vector<GlobalObjectModelDatabase::flann_model> flann_models_tmp ( gom->model_signatures_.rows());
            for(size_t fm_id=0; fm_id<flann_models_tmp.size(); fm_id++)
            {
                GlobalObjectModelDatabase::flann_model &f = flann_models_tmp[fm_id];
                f.instance_name_ = m->id_;
                f.class_name_ = m->class_;
                f.view_id_ = fm_id;
            }
            gomdb_.flann_models_.insert( gomdb_.flann_models_.end(),  flann_models_tmp.begin(), flann_models_tmp.end() );
        }
    }

    classifier_->train(all_model_signatures, all_trained_model_labels);
    all_model_signatures.resize(0,0);  // not needed here any more
    all_trained_model_labels.resize(0);
}


template<typename PointT>
void
GlobalRecognizer<PointT>::featureMatching( const Eigen::MatrixXf &query_sig )
{
    if(query_sig.cols() == 0 || query_sig.rows() == 0)
        return;

    Eigen::MatrixXi predicted_label;
    classifier_->predict(query_sig, predicted_label);

    Eigen::Matrix4f tf_trans = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f tf_rot = Eigen::Matrix4f::Identity();
    Eigen::Vector4f centroid;

    if( !param_.estimate_pose_ )
    {
        obj_hyps_filtered_.resize( predicted_label.rows() * predicted_label.cols() );
        for(int query_id=0; query_id<predicted_label.rows(); query_id++)
        {
            for (int k = 0; k < predicted_label.cols(); k++)
            {
                int lbl = predicted_label(query_id, k);
                const std::string &model_name = id_to_model_name_[lbl];
                const std::string &class_name = "";

                obj_hyps_filtered_[query_id*predicted_label.cols()+k].reset( new ObjectHypothesis<PointT>);
                obj_hyps_filtered_[query_id*predicted_label.cols()+k]->model_id_ = model_name;
                obj_hyps_filtered_[query_id*predicted_label.cols()+k]->class_id_ = class_name;
            }
        }
    }
    else
    {
        CHECK( classifier_->getType() == ClassifierType::KNN || param_.use_table_plane_for_alignment_);
        CHECK( !param_.use_table_plane_for_alignment_ ||
               (param_.use_table_plane_for_alignment_ && cluster_->isTablePlaneSet() ) ) << "Selected to use table plane for pose alignment but table plane has not been set! " << std::endl;

        if( param_.use_table_plane_for_alignment_ )   // we do not need to know the closest training view
        {
            float dist = dist2plane(cluster_->centroid_.head(3), cluster_->table_plane_);
            centroid = cluster_->centroid_.head(3) - dist * cluster_->table_plane_.head(3);

            // create some arbitrary coordinate system on table plane (s.t. normal corresponds to z axis, and others are orthonormal)
            Eigen::Vector3f vec_z = cluster_->table_plane_.topRows(3);
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
            centroid = cluster_->centroid_;
        }
        tf_trans.block<3,1>(0,3) = centroid.topRows(3);

        // pre-allocate memory
        size_t max_hypotheses = predicted_label.rows()*predicted_label.cols();

        if(param_.use_table_plane_for_alignment_)
            max_hypotheses *= (int)(360.f / param_.z_angle_sampling_density_degree_);
        else
            max_hypotheses *= 4;

        obj_hyps_filtered_.resize( max_hypotheses );

        for(size_t i=0; i<obj_hyps_filtered_.size(); i++)
            obj_hyps_filtered_[i].reset( new ObjectHypothesis<PointT> );

        size_t kept=0;
        for(int query_id=0; query_id<predicted_label.rows(); query_id++)
        {
            for (int k = 0; k < predicted_label.cols(); k++)
            {
                int lbl = predicted_label(query_id, k);

                std::string model_name="", class_name="";
                if ( param_.classify_instances_ )
                    model_name = id_to_model_name_[lbl];
                else
                    class_name = id_to_model_name_[lbl];

                if(param_.use_table_plane_for_alignment_ || classifier_->getType() != ClassifierType::KNN)
                {   // do brute force sampling along azimuth angle
                    GlobalObjectModel::ConstPtr gom = gomdb_.global_models_[model_name];
                    const Eigen::Vector3f &elongations_model = gom->model_elongations_.colwise().maxCoeff();    // as we don't know the view, we just take the maximum extent of each axis over all training views

                    if(keep_all_hypotheses_)
                    {
                        for(float rot_i=0.f; rot_i<360.f; rot_i+=param_.z_angle_sampling_density_degree_)
                        {
                            float rot_rad = pcl::deg2rad(rot_i);
                            Eigen::Matrix4f rot_tmp = Eigen::Matrix4f::Identity();
                            rot_tmp(0,0) =  cos(rot_rad);
                            rot_tmp(0,1) = -sin(rot_rad);
                            rot_tmp(1,0) =  sin(rot_rad);
                            rot_tmp(1,1) =  cos(rot_rad);

                            typename ObjectHypothesis<PointT>::Ptr h( new ObjectHypothesis<PointT>);
                            h->transform_ = tf_trans * tf_rot  * rot_tmp;
                            h->confidence_ =  0.f;
                            h->model_id_ = model_name;
                            h->class_id_ = class_name;
                            all_obj_hyps_.push_back(h);
                        }
                    }

                    if( param_.check_elongations_ &&
                       ( cluster_->elongation_(2)/elongations_model(2) < param_.min_elongation_ratio_||
                         cluster_->elongation_(2)/elongations_model(2) > param_.max_elongation_ratio_||
                         cluster_->elongation_(1)/elongations_model(1) < param_.min_elongation_ratio_||
                         cluster_->elongation_(1)/elongations_model(1) > param_.max_elongation_ratio_) )
                    {
                        continue;
                    }

                    for(float rot_i=0.f; rot_i<360.f; rot_i+=param_.z_angle_sampling_density_degree_)
                    {
                        float rot_rad = pcl::deg2rad(rot_i);
                        Eigen::Matrix4f rot_tmp = Eigen::Matrix4f::Identity();
                        rot_tmp(0,0) =  cos(rot_rad);
                        rot_tmp(0,1) = -sin(rot_rad);
                        rot_tmp(1,0) =  sin(rot_rad);
                        rot_tmp(1,1) =  cos(rot_rad);

                        obj_hyps_filtered_[kept]->transform_ = tf_trans * tf_rot  * rot_tmp;
                        obj_hyps_filtered_[kept]->confidence_ =  0.f;
                        obj_hyps_filtered_[kept]->model_id_ = model_name;
                        obj_hyps_filtered_[kept]->class_id_ = class_name;
                        kept++;
                    }
                }
                else    // align principal axis with the ones from closest view in training set
                {
                    Eigen::MatrixXi knn_indices;
                    Eigen::MatrixXf knn_distances;
                    classifier_->getTrainingSampleIDSforPredictions(knn_indices, knn_distances);
                    const GlobalObjectModelDatabase::flann_model &f = gomdb_.flann_models_ [ knn_indices( query_id, k ) ];
                    const size_t view_id = f.view_id_;
                    auto it = gomdb_.global_models_.find( f.instance_name_ );
                    CHECK( it != gomdb_.global_models_.end() ) << "could not find model " << f.instance_name_ << ". There was something wrong with the model initialiazation. Maybe retraining the database helps.";
                    GlobalObjectModel::ConstPtr gom = it->second;

                    const Eigen::Vector3f &model_elongations = gom->model_elongations_.row( view_id );
                    if( param_.check_elongations_ &&
                       ( cluster_->elongation_(2)/model_elongations(2) < param_.min_elongation_ratio_||
                         cluster_->elongation_(2)/model_elongations(2) > param_.max_elongation_ratio_||
                         cluster_->elongation_(1)/model_elongations(1) < param_.min_elongation_ratio_||
                         cluster_->elongation_(1)/model_elongations(1) > param_.max_elongation_ratio_) )
                        continue;

                    // there are four possibilites (due to sign ambiguity of eigenvector)
                    Eigen::Matrix3f eigenBasis, sign_operator;
                    Eigen::Matrix3f identity = Eigen::Matrix3f::Identity();

                    // once take eigen vector as they are computed
                    sign_operator = identity;
                    eigenBasis = cluster_->eigen_basis_ * sign_operator;
                    Eigen::Matrix4f tf_rot_inv = Eigen::Matrix4f::Identity();
                    tf_rot_inv.block<3,3>(0,0) = eigenBasis.transpose();
                    tf_rot = tf_rot_inv.inverse();
                    Eigen::Matrix4f tf_m_inv = gom->model_poses_[ view_id ].inverse();
                    obj_hyps_filtered_[kept]->transform_ = tf_trans * tf_rot * gom->eigen_based_pose_[ view_id ] * tf_m_inv;
                    obj_hyps_filtered_[kept]->confidence_ = knn_distances( query_id, k );
                    obj_hyps_filtered_[kept]->model_id_ = model_name;
                    obj_hyps_filtered_[kept]->class_id_ = class_name;
                    kept++;

                    // now take the first one negative
                    sign_operator = identity;
                    sign_operator(0,0) = -1;
                    sign_operator(2,2) = -1;   // due to right-hand rule
                    eigenBasis = cluster_->eigen_basis_ * sign_operator;
                    tf_rot_inv = Eigen::Matrix4f::Identity();
                    tf_rot_inv.block<3,3>(0,0) = eigenBasis.transpose();
                    tf_rot = tf_rot_inv.inverse();
                    obj_hyps_filtered_[kept]->transform_ = tf_trans * tf_rot * gom->eigen_based_pose_[ view_id ] * tf_m_inv;
                    obj_hyps_filtered_[kept]->confidence_ = knn_distances( query_id, k );
                    obj_hyps_filtered_[kept]->model_id_ = model_name;
                    obj_hyps_filtered_[kept]->class_id_ = class_name;
                    kept++;


                    // now take the second one negative
                    sign_operator = identity;
                    sign_operator(1,1) = -1;
                    sign_operator(2,2) = -1;   // due to right-hand rule
                    eigenBasis = cluster_->eigen_basis_ * sign_operator;
                    tf_rot_inv = Eigen::Matrix4f::Identity();
                    tf_rot_inv.block<3,3>(0,0) = eigenBasis.transpose();
                    tf_rot = tf_rot_inv.inverse();
                    obj_hyps_filtered_[kept]->transform_ = tf_trans * tf_rot * gom->eigen_based_pose_[ view_id ] * tf_m_inv;
                    obj_hyps_filtered_[kept]->confidence_ = knn_distances( query_id, k );
                    obj_hyps_filtered_[kept]->model_id_ = model_name;
                    obj_hyps_filtered_[kept]->class_id_ = class_name;
                    kept++;


                    // and last take first and second one negative
                    sign_operator = identity;
                    sign_operator(0,0) = -1;
                    sign_operator(1,1) = -1;
                    eigenBasis = cluster_->eigen_basis_ * sign_operator;
                    tf_rot_inv = Eigen::Matrix4f::Identity();
                    tf_rot_inv.block<3,3>(0,0) = eigenBasis.transpose();
                    tf_rot = tf_rot_inv.inverse();
                    obj_hyps_filtered_[kept]->transform_ = tf_trans * tf_rot * gom->eigen_based_pose_[ view_id ] * tf_m_inv;
                    obj_hyps_filtered_[kept]->confidence_ = knn_distances( query_id, k );
                    obj_hyps_filtered_[kept]->model_id_ = model_name;
                    obj_hyps_filtered_[kept]->class_id_ = class_name;
                    kept++;
                }
            }
        }

        obj_hyps_filtered_.resize( kept );
    }
}

template<typename PointT>
void
GlobalRecognizer<PointT>::recognize()
{
    CHECK( !param_.estimate_pose_ || (param_.estimate_pose_ && cluster_) ) << "Cluster that needs to be classified is not set!";

    obj_hyps_filtered_.clear();
    all_obj_hyps_.clear();
    Eigen::MatrixXf signature_tmp;
    featureEncoding( signature_tmp );
    featureMatching( signature_tmp );
    cluster_.reset();
}

template class V4R_EXPORTS GlobalRecognizer<pcl::PointXYZ>;
template class V4R_EXPORTS GlobalRecognizer<pcl::PointXYZRGB>;

}

