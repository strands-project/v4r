#include <v4r/common/miscellaneous.h>
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

//#define _VISUALIZE_

namespace v4r
{


template<typename PointT>
void
GlobalRecognizer<PointT>::validate() const
{
    CHECK ( m_db_ );
    CHECK ( estimator_ );
    CHECK ( estimator_->getFeatureType() != FeatureType::OURCVFH || classifier_->getType() == ClassifierType::KNN ) << "OurCVFH needs to obtain best training view - therefore only available in combination with nearest neighbor search";
    CHECK ( classifier_->getType() == ClassifierType::KNN || param_.use_table_plane_for_alignment_ || !param_.estimate_pose_);
}

template<typename PointT>
void
GlobalRecognizer<PointT>::initialize(const std::string &trained_dir, bool retrain)
{
    validate();

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

#ifdef _VISUALIZE_
        pcl::visualization::PCLVisualizer vis;
        int vp1, vp2;
        vis.createViewPort(0,0,.5,1,vp1);
        vis.createViewPort(0.5,0,1,1,vp2);
        typename pcl::PointCloud<PointT>::ConstPtr model_cloud = m->getAssembled(3);
        vis.addPointCloud(model_cloud, "model", vp1);
#endif


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
                    normal_estimator_->setInputCloud( scene_ );
                    pcl::PointCloud<pcl::Normal>::Ptr normals = normal_estimator_->compute();
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

                    estimator_->setInputCloud(scene_);
                    estimator_->setNormals(scene_normals_);

                    if( !cluster_->indices_.empty() )
                        estimator_->setIndices(cluster_->indices_);

                    Eigen::MatrixXf signature_tmp;
                    estimator_->compute(signature_tmp);

                    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > model_poses_tmp (signature_tmp.rows(), pose);
                    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > eigen_based_pose_tmp (signature_tmp.rows(), cluster_->eigen_pose_alignment_);

                    gom->model_poses_.insert( gom->model_poses_.end(), model_poses_tmp.begin(), model_poses_tmp.end() );
                    gom->eigen_based_pose_.insert( gom->eigen_based_pose_.end(), eigen_based_pose_tmp.begin(), eigen_based_pose_tmp.end() );

                    gom->model_elongations_.conservativeResize( gom->model_elongations_.rows() + signature_tmp.rows(), 3);
                    gom->model_elongations_.bottomRows( signature_tmp.rows() ) = cluster_->elongation_.transpose().replicate(signature_tmp.rows(), 1);

                    gom->model_centroids_.conservativeResize( gom->model_centroids_.rows() + signature_tmp.rows(), 4);
                    gom->model_centroids_.bottomRows( signature_tmp.rows() ) = cluster_->centroid_.transpose().replicate(signature_tmp.rows(), 1);

                    gom->model_signatures_.conservativeResize( gom->model_signatures_.rows() + signature_tmp.rows(), signature_tmp.cols());
                    gom->model_signatures_.bottomRows( signature_tmp.rows() ) = signature_tmp;


                    // for OUR-CVFH
                    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > descriptor_transforms = estimator_->getTransforms( );gom->descriptor_transforms_.insert(gom->descriptor_transforms_.end(), descriptor_transforms.begin(), descriptor_transforms.end() );
#ifdef _VISUALIZE_
                    for(const Eigen::Matrix4f &tf : descriptor_transforms)
                    {
                        Eigen::Matrix4f tf_inv = tf.inverse();
                        {
                        Eigen::Matrix4f transf = pose * tf_inv;
                        Eigen::Matrix3f rot_tmp  = transf.block<3,3>(0,0);
                        Eigen::Vector3f trans_tmp = transf.block<3,1>(0,3);
                        Eigen::Affine3f affine_trans;
                        affine_trans.fromPositionOrientationScale(trans_tmp, rot_tmp, Eigen::Vector3f::Ones());
                        std::stringstream co_id; co_id << tf;
                        vis.addCoordinateSystem(vis_param_->coordinate_axis_scale_, affine_trans, co_id.str(), vp1);
                        }
                        {
                        Eigen::Matrix4f transf = tf_inv;
                        Eigen::Matrix3f rot_tmp  = transf.block<3,3>(0,0);
                        Eigen::Vector3f trans_tmp = transf.block<3,1>(0,3);
                        Eigen::Affine3f affine_trans;
                        affine_trans.fromPositionOrientationScale(trans_tmp, rot_tmp, Eigen::Vector3f::Ones());
                        std::stringstream co_id; co_id << tf<<"vp2";
                        vis.addCoordinateSystem(vis_param_->coordinate_axis_scale_, affine_trans, co_id.str(), vp2);
                        }
                        vis.spin();
                    }
#endif
                }
                else
                    LOG(INFO) << "Ignoring view " << tv->filename_ << " because a similar camera pose exists.";


                cluster_.reset();
                scene_.reset();
                scene_normals_.reset();
            }


#ifdef _VISUALIZE_
                    vis.removeAllPointClouds(vp2);
                    vis.addPointCloud(scene_, "training_view", vp2);
            vis.spin();
#endif

            CHECK( (gom->model_elongations_.rows() == gom->model_centroids_.rows()) &&
                   (gom->model_elongations_.rows() == gom->model_signatures_.rows())     );


            // compute the average discrepancy between the centroid computed on the whole 3D model to the centroid computed on individual 2.5D training views.
            // Can be used later to compensate object's pose translation component.
            Eigen::VectorXf view_centroid_to_3d_model_centroid (gom->model_centroids_.rows());
            for(int view_id=0; view_id < gom->model_centroids_.rows(); view_id++)
            {
                const Eigen::Vector4f &view_centroid = gom->model_centroids_.row(view_id).transpose();
                const Eigen::Vector4f &view_centroid_aligned = gom->model_poses_[view_id] * view_centroid;

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

        if( gom->model_signatures_.rows() )
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
GlobalRecognizer<PointT>::featureEncodingAndMatching(  )
{
    Eigen::MatrixXf query_sig;

    estimator_->setInputCloud(scene_);
    estimator_->setNormals(scene_normals_);

    if( !cluster_->indices_.empty() )
        estimator_->setIndices(cluster_->indices_);

    estimator_->compute(query_sig);
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > descriptor_transforms = estimator_->getTransforms( );

    if( query_sig.cols() == 0 || query_sig.rows() == 0)
    {
        LOG(ERROR) << "No signature computed for input cluster!";
        return;
    }

    Eigen::MatrixXi predicted_label;
    classifier_->predict(query_sig, predicted_label);

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

                typename ObjectHypothesis<PointT>::Ptr oh( new ObjectHypothesis<PointT>);
                oh->model_id_ = model_name;
                oh->class_id_ = class_name;
                obj_hyps_filtered_[query_id*predicted_label.cols()+k] = oh;
            }
        }
        return;
    }
    else if( !descriptor_transforms.empty() ) // this will be true for OURCVFH - we can estimate the object pose from the computed SGURF (semi-global unique reference frame)
    {
        Eigen::MatrixXi knn_indices;
        Eigen::MatrixXf knn_distances;
        classifier_->getTrainingSampleIDSforPredictions(knn_indices, knn_distances);

        obj_hyps_filtered_.resize( predicted_label.rows() * predicted_label.cols() );
        for(int query_id=0; query_id<predicted_label.rows(); query_id++)
        {
            for (int k = 0; k < predicted_label.cols(); k++)
            {
                const GlobalObjectModelDatabase::flann_model &f = gomdb_.flann_models_ [ knn_indices( query_id, k ) ];
                const size_t view_id = f.view_id_;
                auto it = gomdb_.global_models_.find( f.instance_name_ );
                CHECK( it != gomdb_.global_models_.end() ) << "could not find model " << f.instance_name_ << ". There was something wrong with the model initialiazation. Maybe retraining the database helps.";

                const GlobalObjectModel::ConstPtr &gom = it->second;

                typename ObjectHypothesis<PointT>::Ptr oh( new ObjectHypothesis<PointT>);
                oh->model_id_ = f.instance_name_;
                oh->class_id_ = f.class_name_;
                oh->transform_ = 1.f * descriptor_transforms[query_id].inverse() * gom->descriptor_transforms_[view_id] * gom->model_poses_[view_id].inverse();
                obj_hyps_filtered_[query_id*predicted_label.cols()+k] = oh;

#ifdef _VISUALIZE_
                pcl::visualization::PCLVisualizer vis;
                int vp1, vp2, vp3, vp4, vp5, vp6;
                vis.createViewPort(0,0,0.33,0.5,vp1);
                vis.createViewPort(0.33,0,0.66,0.5,vp2);
                vis.createViewPort(0.66,0,1,0.5,vp3);
                vis.createViewPort(0,0.5,0.33,1,vp4);
                vis.createViewPort(0.33,0.5,0.66,1,vp5);
                vis.createViewPort(0.66,0.5,1,1,vp6);
                vis.setBackgroundColor(vis_param_->bg_color_[0], vis_param_->bg_color_[1], vis_param_->bg_color_[2], vp1);
                vis.setBackgroundColor(vis_param_->bg_color_[0], vis_param_->bg_color_[1], vis_param_->bg_color_[2], vp2);
                vis.setBackgroundColor(vis_param_->bg_color_[0], vis_param_->bg_color_[1], vis_param_->bg_color_[2], vp3);
                vis.setBackgroundColor(vis_param_->bg_color_[0], vis_param_->bg_color_[1], vis_param_->bg_color_[2], vp4);
                vis.setBackgroundColor(vis_param_->bg_color_[0], vis_param_->bg_color_[1], vis_param_->bg_color_[2], vp5);
                vis.setBackgroundColor(vis_param_->bg_color_[0], vis_param_->bg_color_[1], vis_param_->bg_color_[2], vp6);
                typename pcl::PointCloud<PointT>::Ptr cluster (new pcl::PointCloud<PointT>);
                typename pcl::PointCloud<PointT>::Ptr cluster_aligned (new pcl::PointCloud<PointT>);

                typename pcl::PointCloud<PointT>::Ptr model_view (new pcl::PointCloud<PointT>);
                typename pcl::PointCloud<PointT>::Ptr model_aligned (new pcl::PointCloud<PointT>);
                pcl::copyPointCloud(*scene_, cluster_->indices_, *cluster);
                vis.addPointCloud(cluster, "cluster", vp1);
                vis.addCoordinateSystem(vis_param_->coordinate_axis_scale_, "cluster_co", vp1);
                pcl::transformPointCloud(*cluster, *cluster_aligned, descriptor_transforms[query_id]);
                vis.addCoordinateSystem(vis_param_->coordinate_axis_scale_, "cluster_aligned_co", vp2);
                vis.addPointCloud(cluster_aligned, "cluster_aligned", vp2);

                bool found_model;
                typename Model<PointT>::ConstPtr m = m_db_->getModelById(f.class_name_, f.instance_name_, found_model);
                typename pcl::PointCloud<PointT>::ConstPtr model_cloud = m->getAssembled(3);
                vis.addPointCloud(model_cloud, "model", vp4);
                vis.addCoordinateSystem(vis_param_->coordinate_axis_scale_, "model", vp4);
                vis.addCoordinateSystem(vis_param_->coordinate_axis_scale_, "model_view", vp5);
                vis.addCoordinateSystem(vis_param_->coordinate_axis_scale_, "model_aligned", vp6);
                pcl::transformPointCloud(*model_cloud, *model_view, gom->model_poses_[view_id].inverse());
                vis.addPointCloud(model_view, "model_view", vp5);
                pcl::transformPointCloud(*model_view, *model_aligned, gom->descriptor_transforms_[view_id]);
                vis.addPointCloud(model_aligned, "model_aligned", vp6);
                vis.spin();
#endif
            }
        }
        return;
    }
    else    // estimate pose using some prior assumptions
    {
        CHECK( !param_.use_table_plane_for_alignment_ ||
               (param_.use_table_plane_for_alignment_ && cluster_->isTablePlaneSet() ) ) << "Selected to use table plane for pose alignment but table plane has not been set! " << std::endl;

        Eigen::Matrix4f tf_rot = Eigen::Matrix4f::Identity();

        if( param_.use_table_plane_for_alignment_ )   // rotate cluster such that the surface normal of the planar support is aligned with the z-axis. We do not need to know the closest training view.
        {
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

                GlobalObjectModel::ConstPtr gom = gomdb_.global_models_[model_name];
                const Eigen::Vector3f &elongations_model = gom->model_elongations_.colwise().maxCoeff();    // as we don't know the view, we just take the maximum extent of each axis over all training views
//                const Eigen::Vector3f &elongations_model = gom->model_elongations_.row( view_id );

                if( param_.check_elongations_ &&
                   ( cluster_->elongation_(2)/elongations_model(2) < param_.min_elongation_ratio_||
                     cluster_->elongation_(2)/elongations_model(2) > param_.max_elongation_ratio_||
                     cluster_->elongation_(1)/elongations_model(1) < param_.min_elongation_ratio_||
                     cluster_->elongation_(1)/elongations_model(1) > param_.max_elongation_ratio_) )
                {
                    continue;
                }

                if(param_.use_table_plane_for_alignment_ || classifier_->getType() != ClassifierType::KNN)
                {   // do brute force sampling along azimuth angle

                    bool found_model;
                    typename Model<PointT>::ConstPtr m = m_db_->getModelById (class_name, model_name, found_model);

                    // align model centroid with origin
                    Eigen::Matrix4f tf_om_shift2origin = Eigen::Matrix4f::Identity();
                    tf_om_shift2origin.block<3,1>(0,3) = -m->centroid_.head(3);

                    // move model such that no point is below z=0
                    Eigen::Matrix4f tf_om_shift2origin2 = Eigen::Matrix4f::Identity();
                    tf_om_shift2origin2(2,3) = -(m->minPoint_(2)-m->centroid_(2));

                    // align origin with downprojected cluster centroid
                    float centroid_correction = gom->mean_distance_view_centroid_to_3d_model_centroid_;

                    Eigen::Vector3f centroid_normalized = cluster_->centroid_.head(3).normalized();
                    Eigen::Vector3f centroid_corrected = cluster_->centroid_.head(3) + centroid_correction * centroid_normalized;

                    Eigen::Vector3f closest_pt_to_cluster_center = getClosestPointOnPlane(centroid_corrected, cluster_->table_plane_);
                    Eigen::Matrix4f tf_cluster_shift = Eigen::Matrix4f::Identity();
//                    tf11.block<3,1>(0,3) = -cluster_->centroid_.head(3);
                    tf_cluster_shift.block<3,1>(0,3) = -closest_pt_to_cluster_center;

                    // align table plane surface normal with model coordinate's z-axis
                    Eigen::Matrix4f tf_cluster_rot = Eigen::Matrix4f::Identity();
                    tf_cluster_rot.block<3,3>(0,0) = computeRotationMatrixToAlignVectors(cluster_->table_plane_.head(3), Eigen::Vector3f::UnitZ()); //Finv * G * F;

                    Eigen::Matrix4f align_cluster = tf_cluster_rot * tf_cluster_shift;

#ifdef _VISUALIZE_
                    pcl::visualization::PCLVisualizer vis;
                    int vp1, vp2, vp3, vp4, vp5, vp6, vp7, vp8;
                    vis.createViewPort( 0, 0, 0.25, 0.5, vp1);
                    vis.createViewPort(0.25, 0, 0.5, 0.5, vp2);
                    vis.createViewPort(0.5, 0, 0.75, 0.5, vp3);
                    vis.createViewPort(0.75, 0, 1, 0.5, vp4);
                    vis.createViewPort( 0, 0.5, 0.25, 1, vp5);
                    vis.createViewPort(0.25, 0.5, 0.5, 1, vp6);
                    vis.createViewPort(0.5, 0.5, 0.75, 1, vp7);
                    vis.createViewPort(0.75, 0.5, 1, 1, vp8);
                    vis.setBackgroundColor(vis_param_->bg_color_[0], vis_param_->bg_color_[1], vis_param_->bg_color_[2], vp1);
                    vis.setBackgroundColor(vis_param_->bg_color_[0], vis_param_->bg_color_[1], vis_param_->bg_color_[2], vp2);
                    vis.setBackgroundColor(vis_param_->bg_color_[0], vis_param_->bg_color_[1], vis_param_->bg_color_[2], vp3);
                    vis.setBackgroundColor(vis_param_->bg_color_[0], vis_param_->bg_color_[1], vis_param_->bg_color_[2], vp4);
                    vis.setBackgroundColor(vis_param_->bg_color_[0], vis_param_->bg_color_[1], vis_param_->bg_color_[2], vp5);
                    vis.setBackgroundColor(vis_param_->bg_color_[0], vis_param_->bg_color_[1], vis_param_->bg_color_[2], vp6);
                    vis.setBackgroundColor(vis_param_->bg_color_[0], vis_param_->bg_color_[1], vis_param_->bg_color_[2], vp7);
                    vis.setBackgroundColor(vis_param_->bg_color_[0], vis_param_->bg_color_[1], vis_param_->bg_color_[2], vp8);


                    typename pcl::PointCloud<PointT>::Ptr model_shifted (new pcl::PointCloud<PointT>);
                    typename pcl::PointCloud<PointT>::ConstPtr model_cloud = m->getAssembled(5);

                    typename pcl::PointCloud<PointT>::Ptr model_shifted2 (new pcl::PointCloud<PointT>);
                    vis.addPointCloud(model_cloud, "original", vp1);
                    vis.addCoordinateSystem(vis_param_->coordinate_axis_scale_, "original2", vp1);
                    if( !vis_param_->no_text_)
                    {
                        vis.addText("object in model coordinate system", 10, 10, vis_param_->fontsize_, vis_param_->text_color_[0], vis_param_->text_color_[1], vis_param_->text_color_[2], "object in model coordinate system", vp1);
                        vis.addText("model coordinate system aligned with centroid", 10, 10, vis_param_->fontsize_, vis_param_->text_color_[0], vis_param_->text_color_[1], vis_param_->text_color_[2], "model coordinate system aligned with centroid", vp2);
                        vis.addText("model coordinate system aligned with downprojected centroid", 10, 10, vis_param_->fontsize_, vis_param_->text_color_[0], vis_param_->text_color_[1], vis_param_->text_color_[2], "model downprojected", vp3);
                        vis.addText("sampled rotation angles", 10, 10, vis_param_->fontsize_, vis_param_->text_color_[0], vis_param_->text_color_[1], vis_param_->text_color_[2], "sampled rotation angles", vp4);
                        vis.addText("original cluster", 10, 10, vis_param_->fontsize_, vis_param_->text_color_[0], vis_param_->text_color_[1], vis_param_->text_color_[2], "original cluster", vp5);
                        vis.addText("origin aligned with cluster centroid", 10, 10, vis_param_->fontsize_, vis_param_->text_color_[0], vis_param_->text_color_[1], vis_param_->text_color_[2], "cluster centroid aligned with origin", vp6);
                        vis.addText("origin aligned with downprojected cluster centroid", 10, 10, vis_param_->fontsize_, vis_param_->text_color_[0], vis_param_->text_color_[1], vis_param_->text_color_[2], "origin aligned with downprojected cluster centroid", vp7);
                        std::stringstream correction_txt;
                        correction_txt << "origin aligned with downprojected cluster centroid corrected by " << (int)(1000.f*centroid_correction) << " mm.";
                        vis.addText(correction_txt.str(), 10, 10, vis_param_->fontsize_, vis_param_->text_color_[0], vis_param_->text_color_[1], vis_param_->text_color_[2], "origin aligned with shifted downprojected cluster centroid", vp8);
                    }

                    pcl::transformPointCloud(*model_cloud, *model_shifted, tf_om_shift2origin);
                    vis.addPointCloud(model_shifted, "shifted", vp2);
                    vis.addCoordinateSystem(vis_param_->coordinate_axis_scale_, "shifted_co", vp2);

                    pcl::transformPointCloud(*model_shifted, *model_shifted2, tf_om_shift2origin2);
                    vis.addPointCloud(model_shifted2, "shifted2", vp3);
                    vis.addCoordinateSystem(vis_param_->coordinate_axis_scale_, "shifted2_co", vp3);

                    typename pcl::PointCloud<PointT>::Ptr cluster (new pcl::PointCloud<PointT>);
                    typename pcl::PointCloud<PointT>::Ptr cluster_shifted_and_aligned (new pcl::PointCloud<PointT>);
                    typename pcl::PointCloud<PointT>::Ptr cluster_shifted_and_aligned_wo_correction (new pcl::PointCloud<PointT>);
                    typename pcl::PointCloud<PointT>::Ptr cluster_shifter_and_aligned_not_downprojected_not_corrected (new pcl::PointCloud<PointT>);

                    pcl::copyPointCloud(*scene_, cluster_->indices_, *cluster);
                    vis.addPointCloud(cluster, "cluster", vp5);
                    vis.addCoordinateSystem(vis_param_->coordinate_axis_scale_, "cluster2", vp5);

                    Eigen::Matrix4f tf_cluster_wo_downprojection = Eigen::Matrix4f::Identity();
                    tf_cluster_wo_downprojection.block<3,1>(0,3) = -cluster_->centroid_.head(3);
                    pcl::transformPointCloud(*cluster, *cluster_shifter_and_aligned_not_downprojected_not_corrected, tf_cluster_rot * tf_cluster_wo_downprojection);
                    Eigen::Vector4f min3d_tmp, max3d_tmp;
                    pcl::getMinMax3D(*cluster_shifter_and_aligned_not_downprojected_not_corrected, min3d_tmp, max3d_tmp); ///TODO: Do this computation during initialization
                    Eigen::Matrix4f tf_cluster_shift_wo_down_wo_correction = Eigen::Matrix4f::Identity();
                    tf_cluster_shift_wo_down_wo_correction(2,3) = -min3d_tmp(2);
                    pcl::transformPointCloud(*cluster_shifter_and_aligned_not_downprojected_not_corrected, *cluster_shifter_and_aligned_not_downprojected_not_corrected, tf_cluster_shift_wo_down_wo_correction);
                    vis.addPointCloud(cluster_shifter_and_aligned_not_downprojected_not_corrected, "cluster_shifter_and_aligned_not_downprojected_not_corrected", vp6);
                    vis.addCoordinateSystem(vis_param_->coordinate_axis_scale_, "cluster_shifter_and_aligned_not_downprojected_not_corrected_co", vp6);

                    Eigen::Vector3f closest_pt_to_cluster_center_wo_correction = getClosestPointOnPlane(cluster_->centroid_.head(3), cluster_->table_plane_);
                    Eigen::Matrix4f tf_cluster_shift_wo_correction = Eigen::Matrix4f::Identity();
//                    tf11.block<3,1>(0,3) = -cluster_->centroid_.head(3);
                    tf_cluster_shift_wo_correction.block<3,1>(0,3) = -closest_pt_to_cluster_center_wo_correction;
                    pcl::transformPointCloud(*cluster, *cluster_shifted_and_aligned_wo_correction, tf_cluster_rot * tf_cluster_shift_wo_correction);
                    vis.addPointCloud(cluster_shifted_and_aligned_wo_correction, "cluster_shifted_and_aligned_wo_correction", vp7);
                    vis.addCoordinateSystem(vis_param_->coordinate_axis_scale_, "cluster_shifted_and_aligned_wo_correction_co", vp7);

                    pcl::transformPointCloud(*cluster, *cluster_shifted_and_aligned, tf_cluster_rot * tf_cluster_shift);
                    vis.addPointCloud(cluster_shifted_and_aligned, "cluster_shifted_and_aligned", vp8);
                    vis.addCoordinateSystem(vis_param_->coordinate_axis_scale_, "cluster_shifted_and_aligned_co", vp8);
#endif

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

#ifdef _VISUALIZE_
                            typename pcl::PointCloud<PointT>::Ptr model_shifted_and_rotated (new pcl::PointCloud<PointT>);
                            pcl::transformPointCloud(*model_shifted2, *model_shifted_and_rotated, rot_tmp);
                            std::stringstream tmp; tmp << "cluster_shifted_and_rotated_" << rot_i;
                            vis.addPointCloud(model_shifted_and_rotated, tmp.str(), vp4);
                            vis.addCoordinateSystem(vis_param_->coordinate_axis_scale_, tmp.str()+ "2", vp4);
#endif
                            Eigen::Matrix4f alignment_tf = align_cluster.inverse() * rot_tmp * tf_om_shift2origin2 * tf_om_shift2origin;

                            typename ObjectHypothesis<PointT>::Ptr h( new ObjectHypothesis<PointT>);
                            h->transform_ = alignment_tf; //tf_trans * tf_rot  * rot_tmp;
                            h->confidence_ =  0.f;
                            h->model_id_ = model_name;
                            h->class_id_ = class_name;
                            all_obj_hyps_.push_back(h);
                        }
                    }
#ifdef _VISUALIZE_
                    vis.spin();
#endif

                    for(float rot_i=0.f; rot_i<360.f; rot_i+=param_.z_angle_sampling_density_degree_)
                    {
                        float rot_rad = pcl::deg2rad(rot_i);
                        Eigen::Matrix4f rot_tmp = Eigen::Matrix4f::Identity();
                        rot_tmp(0,0) =  cos(rot_rad);
                        rot_tmp(0,1) = -sin(rot_rad);
                        rot_tmp(1,0) =  sin(rot_rad);
                        rot_tmp(1,1) =  cos(rot_rad);

                        Eigen::Matrix4f alignment_tf = align_cluster.inverse() * rot_tmp * tf_om_shift2origin2 * tf_om_shift2origin;

                        obj_hyps_filtered_[kept]->transform_ = alignment_tf;//tf_trans * tf_rot  * rot_tmp;
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

                    Eigen::Matrix4f tf_trans = Eigen::Matrix4f::Identity();
                    tf_trans.block<3,1>(0,3) = cluster_->centroid_.topRows(3);

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
    featureEncodingAndMatching(  );
    cluster_.reset();
}

template class V4R_EXPORTS GlobalRecognizer<pcl::PointXYZ>;
template class V4R_EXPORTS GlobalRecognizer<pcl::PointXYZRGB>;

}

