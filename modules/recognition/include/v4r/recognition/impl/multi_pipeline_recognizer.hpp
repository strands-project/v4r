#include <v4r/recognition/multi_pipeline_recognizer.h>
#include <v4r/recognition/local_recognizer.h>
#include <v4r/recognition/global_recognizer.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <v4r/common/normals.h>
#include <v4r/features/types.h>
#include <glog/logging.h>
#include <omp.h>

namespace v4r
{

template<typename PointT>
bool
MultiRecognitionPipeline<PointT>::initialize(bool force_retrain)
{
    for(auto &r:recognizers_)
        r->initialize(force_retrain);

    return true;
}

template<typename PointT>
bool
MultiRecognitionPipeline<PointT>::update()
{
    for(auto &r:recognizers_)
        r->initialize(false);

    return true;
}

template<typename PointT>
bool
MultiRecognitionPipeline<PointT>::retrain( const std::string &model_name)
{
    for(auto &r:recognizers_)
        r->initialize(true);

    return true;
}

template<typename PointT>
void
MultiRecognitionPipeline<PointT>::callIndiviualRecognizer(boost::shared_ptr<Recognizer<PointT> > &rec)
{
    std::map<std::string, LocalObjectHypothesis<PointT> > local_hypotheses;
    std::vector<ObjectHypothesesGroup<PointT> > global_hypotheses;
    pcl::PointCloud<PointT> scene_kps;
    pcl::PointCloud<pcl::Normal> scene_kp_normals;

    rec->setInputCloud(scene_);
    rec->setSceneNormals(scene_normals_);
    rec->recognize();
    typename LocalRecognitionPipeline<PointT>::Ptr local_rec = boost::dynamic_pointer_cast<LocalRecognitionPipeline<PointT> > (rec);
    typename GlobalRecognizer<PointT>::Ptr global_rec = boost::dynamic_pointer_cast<GlobalRecognizer<PointT> > (rec);

    ///TODO: If all local recognizers are ready, we don't have to wait for global recognizers to finish but can already start with correspondence grouping
    if(global_rec) // for global recognizers
    {
        global_hypotheses = global_rec->getObjectHypothesis();
    }
    else if(local_rec)  // for local recognizers
    {
        local_rec->getSavedHypotheses( local_hypotheses );

        std::vector<int> kp_indices;
        typename pcl::PointCloud<PointT>::Ptr kp_tmp = local_rec->getKeypointCloud();
        local_rec->getKeypointIndices(kp_indices);
        CHECK(kp_tmp->points.size() == kp_indices.size());
        pcl::copyPointCloud(*scene_normals_, kp_indices, scene_kp_normals);
        scene_kps = *kp_tmp;
    }

    mergeStuff(global_hypotheses, local_hypotheses, scene_kps, scene_kp_normals);
}

template<typename PointT>
void
MultiRecognitionPipeline<PointT>::mergeStuff( const std::vector<ObjectHypothesesGroup<PointT>> &global_hypotheses,
                                              std::map<std::string, LocalObjectHypothesis<PointT> > &oh_m,
                                              const pcl::PointCloud<PointT> &scene_kps,
                                              const pcl::PointCloud<pcl::Normal> &scene_kp_normals)
{
    omp_set_lock(&rec_lock_);
    obj_hypotheses_.insert(obj_hypotheses_.end(), global_hypotheses.begin(), global_hypotheses.end());
//            input_icp_indices.insert(input_icp_indices.end(), segmentation_indices_[c].indices.begin(), segmentation_indices_[c].indices.end());

    for (auto &oh : oh_m) {
        for (auto &corr : oh.second.model_scene_corresp_) {  // add appropriate offset to correspondence index of the scene cloud
            corr.index_match += scene_keypoints_->points.size();
        }

        auto it_mp_oh = local_obj_hypotheses_.find(oh.first);
        if(it_mp_oh == local_obj_hypotheses_.end())   // no feature correspondences exist yet
            local_obj_hypotheses_.insert(oh);//std::pair<std::string, ObjectHypothesis<PointT> >(id, it_tmp->second));
        else
            it_mp_oh->second.model_scene_corresp_.insert(  it_mp_oh->second.model_scene_corresp_.  end(),
                                                                   oh.second.model_scene_corresp_.begin(),
                                                                   oh.second.model_scene_corresp_.  end() );
    }

    *scene_keypoints_ += scene_kps;

    if(scene_normals_)
        *scene_kp_normals_ += scene_kp_normals;


    omp_unset_lock(&rec_lock_);
}

template<typename PointT>
void
MultiRecognitionPipeline<PointT>::recognize()
{
    local_obj_hypotheses_.clear();
    obj_hypotheses_.clear();
    scene_keypoints_.reset(new pcl::PointCloud<PointT>);
    scene_kp_normals_.reset(new pcl::PointCloud<pcl::Normal>);
    scene_normals_.reset(new pcl::PointCloud<pcl::Normal>);
    computeNormals<PointT>(scene_, scene_normals_, param_.normal_computation_method_);

    std::vector<typename boost::shared_ptr<Recognizer<PointT> > > recognizer_without_siftgpu;
    typename boost::shared_ptr<Recognizer<PointT> > rec_siftgpu;
    for(size_t r_id=0; r_id < recognizers_.size(); r_id++)
    {
        if(recognizers_[r_id]->getFeatureType() != SIFT_GPU)
            recognizer_without_siftgpu.push_back( recognizers_[r_id]);
        else
            rec_siftgpu = recognizers_[r_id];
    }

    omp_init_lock(&rec_lock_);
//#pragma omp parallel
    {
//#pragma omp master  // SIFT-GPU needs to be exexuted in master thread as SIFT-GPU creates an OpenGL context which never gets destroyed really and crashed if used from another thread
        if(rec_siftgpu)
            callIndiviualRecognizer(rec_siftgpu);

//#pragma omp for schedule(dynamic)
    for(size_t r_id=0; r_id < recognizer_without_siftgpu.size(); r_id++)
        callIndiviualRecognizer(recognizer_without_siftgpu[r_id]);

    }
    omp_destroy_lock(&rec_lock_);

    compress();

    if(cg_algorithm_)    // correspondence grouping is not done outside
    {
        correspondenceGrouping();

        if (hv_algorithm_) //Prepare scene and model clouds for the pose refinement step
            getDataSource()->voxelizeAllModels (param_.voxel_size_icp_);

        if ( hv_algorithm_ && obj_hypotheses_.size() )
            hypothesisVerification();

        scene_keypoints_.reset();
        scene_kp_normals_.reset();
    }
}

template<typename PointT>
void MultiRecognitionPipeline<PointT>::correspondenceGrouping ()
{
    pcl::ScopeTime t("Correspondence Grouping");

    std::vector<LocalObjectHypothesis<PointT> > lohs(local_obj_hypotheses_.size());

    size_t id=0;
    typename std::map<std::string, LocalObjectHypothesis<PointT> >::const_iterator it;
    for (it = local_obj_hypotheses_.begin (), id=0; it != local_obj_hypotheses_.end (); ++it)
        lohs[id++] = it->second;

#pragma omp parallel for schedule(dynamic)
    for (size_t i=0; i<lohs.size(); i++)
    {
        const LocalObjectHypothesis<PointT> &loh = lohs[i];

        if(loh.model_scene_corresp_.size() < 3)
            continue;

        GraphGeometricConsistencyGrouping<PointT, PointT> cg = *cg_algorithm_;

        std::vector < pcl::Correspondences > corresp_clusters;
        cg.setSceneCloud (scene_keypoints_);
        cg.setInputCloud (loh.model_->keypoints_);

//        oh.visualize(*scene_, *scene_keypoints_);

        if(cg.getRequiresNormals())
            cg.setInputAndSceneNormals(loh.model_->kp_normals_, scene_kp_normals_);

        //we need to pass the keypoints_pointcloud and the specific object hypothesis
        cg.setModelSceneCorrespondences (loh.model_scene_corresp_);
        cg.cluster (corresp_clusters);

        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > new_transforms (corresp_clusters.size());
        typename pcl::registration::TransformationEstimationSVD < PointT, PointT > t_est;

        for (size_t cluster_id = 0; cluster_id < corresp_clusters.size(); cluster_id++)
            t_est.estimateRigidTransformation (*loh.model_->keypoints_, *scene_keypoints_, corresp_clusters[cluster_id], new_transforms[cluster_id]);

        if(param_.merge_close_hypotheses_) {
            std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > merged_transforms (corresp_clusters.size());
            std::vector<bool> cluster_has_been_taken(corresp_clusters.size(), false);
            const double angle_thresh_rad = param_.merge_close_hypotheses_angle_ * M_PI / 180.f ;

            size_t kept=0;
            for (size_t tf_id = 0; tf_id < new_transforms.size(); tf_id++) {

                if (cluster_has_been_taken[tf_id])
                    continue;

                cluster_has_been_taken[tf_id] = true;
                const Eigen::Vector3f centroid1 = new_transforms[tf_id].block<3, 1> (0, 3);
                const Eigen::Matrix3f rot1 = new_transforms[tf_id].block<3, 3> (0, 0);

                pcl::Correspondences merged_corrs = corresp_clusters[tf_id];

                for(size_t j=tf_id+1; j < new_transforms.size(); j++) {
                    const Eigen::Vector3f centroid2 = new_transforms[j].block<3, 1> (0, 3);
                    const Eigen::Matrix3f rot2 = new_transforms[j].block<3, 3> (0, 0);
                    const Eigen::Matrix3f rot_diff = rot2 * rot1.transpose();

                    double rotx = std::abs( atan2(rot_diff(2,1), rot_diff(2,2)));
                    double roty = std::abs( atan2(-rot_diff(2,0), sqrt(rot_diff(2,1) * rot_diff(2,1) + rot_diff(2,2) * rot_diff(2,2))) );
                    double rotz = std::abs( atan2(rot_diff(1,0), rot_diff(0,0)) );
                    double dist = (centroid1 - centroid2).norm();

                    if ( (dist < param_.merge_close_hypotheses_dist_) && (rotx < angle_thresh_rad) && (roty < angle_thresh_rad) && (rotz < angle_thresh_rad) ) {
                        merged_corrs.insert( merged_corrs.end(), corresp_clusters[j].begin(), corresp_clusters[j].end() );
                        cluster_has_been_taken[j] = true;
                    }
                }

                t_est.estimateRigidTransformation (*loh.model_->keypoints_, *scene_keypoints_, merged_corrs, merged_transforms[kept]);
                kept++;
            }
            merged_transforms.resize(kept);

            #pragma omp critical
            {
                for(size_t jj=0; jj<merged_transforms.size(); jj++)
                {
                    typename ObjectHypothesis<PointT>::Ptr new_oh (new ObjectHypothesis<PointT>);
                    new_oh->model_ = loh.model_;
                    new_oh->transform_ = merged_transforms[jj];
                    new_oh->confidence_ = corresp_clusters.size();

                    ObjectHypothesesGroup<PointT> new_ohg;
                    new_ohg.global_hypotheses_ = false;
                    new_ohg.ohs_.push_back( new_oh );
                    obj_hypotheses_.push_back( new_ohg );
                }
                std::cout << "Merged " << corresp_clusters.size() << " clusters into " << kept << " clusters. Total correspondences: " << loh.model_scene_corresp_.size () << " " << loh.model_->id_ << std::endl;
            }
        }
        else {
            #pragma omp critical
            {
                for(size_t jj=0; jj<new_transforms.size(); jj++)
                {
                    typename ObjectHypothesis<PointT>::Ptr new_oh (new ObjectHypothesis<PointT>);
                    new_oh->model_ = loh.model_;
                    new_oh->transform_ = new_transforms[jj];
                    new_oh->confidence_ = corresp_clusters.size();

                    ObjectHypothesesGroup<PointT> new_ohg;
                    new_ohg.global_hypotheses_ = false;
                    new_ohg.ohs_.push_back( new_oh );
                    obj_hypotheses_.push_back( new_ohg );
                }
            }
        }
    }
}
}
