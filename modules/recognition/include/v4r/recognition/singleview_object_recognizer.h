#ifndef SINGLEVIEW_OBJECT_RECOGNIZER_H
#define SINGLEVIEW_OBJECT_RECOGNIZER_H

#ifndef EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#endif

#include <pcl/common/common.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <v4r/recognition/multi_pipeline_recognizer.h>
#include <v4r/features/sift_local_estimator.h>
#include <v4r/common/common_data_structures.h>
#include <v4r/common/miscellaneous.h>
#include <v4r/core/macros.h>

#include "segmenter.h"
#include "boost_graph_extension.h"


struct camPosConstraints
{
    bool
    operator() (const Eigen::Vector3f & pos) const
    {
        if (pos[2] > 0)
            return true;

        return false;
    }
    ;
};

namespace v4r
{
class V4R_EXPORTS Recognizer
{
protected:
    typedef pcl::PointXYZRGB PointT;
    typedef pcl::PointCloud<PointT> PointInT;
    typedef pcl::PointCloud<PointT>::Ptr PointInTPtr;
    typedef pcl::PointCloud<PointT>::ConstPtr ConstPointInTPtr;
    typedef v4r::rec_3d_framework::Model<PointT> ModelT;
    typedef boost::shared_ptr<ModelT> ModelTPtr;
    typedef flann::L1<float> DistT;
    typedef pcl::Histogram<128> FeatureT;


    boost::shared_ptr<v4r::rec_3d_framework::MultiRecognitionPipeline<PointT> > multi_recog_;
    std::string models_dir_;
    std::string training_dir_sift_;
    std::string training_dir_shot_;
    std::string sift_structure_;
    std::string training_dir_ourcvfh_;
    std::string idx_flann_fn_sift_;
    std::string idx_flann_fn_shot_;

    std::map<std::string, v4r::rec_3d_framework::ObjectHypothesis<PointT> > hypotheses_;
    boost::shared_ptr< pcl::PointCloud<PointT> > pKeypointsMultipipe_;
    pcl::PointIndices keypointIndices_;
    cv::Ptr<SiftGPU> sift_;
    pcl::PointCloud<PointT>::Ptr pInputCloud_;
    pcl::PointCloud<pcl::Normal>::Ptr pSceneNormals_;
    boost::shared_ptr < std::vector<ModelTPtr> > models_;
    std::vector<ModelTPtr> models_verified_;
    boost::shared_ptr < std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > > transforms_;
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transforms_verified_;
    std::vector<std::string> model_ids_verified_;
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr> aligned_models_;
    std::vector<pcl::PointCloud<pcl::Normal>::ConstPtr> aligned_normals_;
    std::vector<pcl::PointCloud<pcl::PointXYZL>::Ptr> aligned_smooth_faces_;
    std::vector<std::string> model_ids_;
    std::vector<v4r::PlaneModel<PointT> > planes_found_;
    std::vector<pcl::PointCloud<PointT>::Ptr> verified_planes_;

    boost::shared_ptr < v4r::CorrespondenceGrouping<PointT, PointT> > cast_cg_alg_;

//    boost::shared_ptr < faat_pcl::rec_3d_framework::ModelOnlySource<pcl::PointXYZRGBNormal, pcl::PointXYZRGB>
//            > model_only_source_;


#ifdef SOC_VISUALIZE
    boost::shared_ptr<pcl::visualization::PCLVisualizer> vis_;
    int v1_,v2_, v3_;
#endif


public:
    struct hv_params{
            double resolution_;
            double inlier_threshold_;
            double radius_clutter_;
            double regularizer_;
            double clutter_regularizer_;
            double occlusion_threshold_;
            int optimizer_type_;
            double color_sigma_l_;
            double color_sigma_ab_;
            bool use_supervoxels_;
            bool detect_clutter_;
            bool ignore_color_;
            double smooth_seg_params_eps_;
            double smooth_seg_params_curv_t_;
            double smooth_seg_params_dist_t_;
            int smooth_seg_params_min_points_;
            int z_buffer_self_occlusion_resolution_;
            bool use_replace_moves_;
            bool requires_normals_;
            double radius_normals_;
            bool initial_status_;
            double hyp_penalty_;
            double duplicity_cm_weight_;
            bool histogram_specification_;
    }hv_params_;

    struct cg_params{
        int cg_size_threshold_;
        double cg_size_;
        double ransac_threshold_;
        double dist_for_clutter_factor_;
        int max_taken_;
        double max_time_for_cliques_computation_;
        double dot_distance_;
        bool use_cg_graph_;
    }cg_params_;

    struct sv_params{
        bool add_planes_;
        int knn_shot_;
        int knn_sift_;
        bool do_sift_;
        bool do_shot_;
        bool do_ourcvfh_;
        int icp_iterations_;
        int icp_type_;
        double chop_at_z_;
        int normal_computation_method_;
    }sv_params_;

    Recognizer ()
    {
        sv_params_.do_sift_ = true;
        sv_params_.do_shot_ = false;
        sv_params_.do_ourcvfh_ = false;
        sv_params_.add_planes_ = true;
        sv_params_.knn_shot_ = 1;
        sv_params_.knn_sift_ = 5;
        sv_params_.normal_computation_method_ = 0;

        sv_params_.icp_iterations_ = 0;
        sv_params_.icp_type_ = 1;
        sv_params_.chop_at_z_ = 1.5f;

        hv_params_.resolution_ = 0.005f;
        hv_params_.inlier_threshold_ = 0.015;
        hv_params_.radius_clutter_ = 0.03;
        hv_params_.regularizer_ = 3;
        hv_params_.clutter_regularizer_ = 5;
        hv_params_.occlusion_threshold_ = 0.01;
        hv_params_.optimizer_type_ = 0;
        hv_params_.color_sigma_l_ = 0.5;
        hv_params_.color_sigma_ab_ = 0.5;
        hv_params_.use_supervoxels_ = false;
        hv_params_.detect_clutter_ = true;
        hv_params_.ignore_color_ = true;
        hv_params_.smooth_seg_params_eps_ = 0.1f;
        hv_params_.smooth_seg_params_curv_t_ = 0.04f;
        hv_params_.smooth_seg_params_dist_t_ = 0.01f;
        hv_params_.smooth_seg_params_min_points_ = 100;
        hv_params_.z_buffer_self_occlusion_resolution_ = 250;
        hv_params_.use_replace_moves_ = true;
        hv_params_.requires_normals_ = false;
        hv_params_.radius_normals_ = 0.02f;
        hv_params_.initial_status_ = false;
        hv_params_.hyp_penalty_ = 0.05f;
        hv_params_.duplicity_cm_weight_ = 0.f;
        hv_params_.histogram_specification_ = true;

        cg_params_.cg_size_threshold_ = 3;
        cg_params_.cg_size_ = 0.015;
        cg_params_.ransac_threshold_ = 0.015;
        cg_params_.dist_for_clutter_factor_ = 0;
        cg_params_.max_taken_ = 2;
        cg_params_.max_time_for_cliques_computation_ = 100;
        cg_params_.dot_distance_ = 0.2;
        cg_params_.use_cg_graph_ = true;

        pInputCloud_.reset(new pcl::PointCloud<PointT>);
        pSceneNormals_.reset(new pcl::PointCloud<pcl::Normal>);

//        model_only_source_.reset (new faat_pcl::rec_3d_framework::ModelOnlySource<pcl::PointXYZRGBNormal, pcl::PointXYZRGB>);


#ifdef SOC_VISUALIZE
        vis_.reset (new pcl::visualization::PCLVisualizer ("classifier visualization"));
        vis_->createViewPort(0,0,0.33,1.f, v1_);
        vis_->createViewPort(0.33,0,0.66,1.f, v2_);
        vis_->createViewPort(0.66,0,1,1.f, v3_);
#endif
    }

    bool recognize ();

    void initialize();

    void getVerifiedPlanes(std::vector<pcl::PointCloud<PointT>::Ptr> &planes) const
    {
        planes = verified_planes_;
    }

    void setTraining_dir_sift(const std::string &training_dir_sift)
    {
        training_dir_sift_ = training_dir_sift;
    }

    void setTraining_dir_shot(const std::string &training_dir_shot)
    {
        training_dir_shot_ = training_dir_shot;
    }

    void setModels_dir(const std::string &models_dir)
    {
        models_dir_ = models_dir;
    }

    void setSift_structure(const std::string &sift_structure)
    {
        sift_structure_ = sift_structure;
    }

    void setTraining_dir_ourcvfh(const std::string &training_dir_ourcvfh)
    {
        training_dir_ourcvfh_ = training_dir_ourcvfh;
    }

    cv::Ptr<SiftGPU> getSift() const
    {
        return sift_;
    }

    void set_sift(const cv::Ptr<SiftGPU> &value)
    {
        sift_ = value;
    }

    void getSavedHypotheses(std::map<std::string, v4r::rec_3d_framework::ObjectHypothesis<PointT> > & hypotheses) const
    {
        hypotheses = hypotheses_;
    }

    void getKeypointsMultipipe (boost::shared_ptr<pcl::PointCloud<PointT> > &pKeypointsMultipipe ) const
    {
        pKeypointsMultipipe = pKeypointsMultipipe_;
    }

    void getKeypointIndices(pcl::PointIndices &keypointIndices) const
    {
        keypointIndices.header = keypointIndices_.header;
        keypointIndices.indices = keypointIndices_.indices;
    }

    template <template<class > class Distance, typename FeatureT>
    void setISPK(typename pcl::PointCloud<FeatureT>::Ptr & signatures, PointInTPtr & p, pcl::PointIndices & keypoint_indices, size_t feature_type)
    {
        multi_recog_->setISPK<Distance, FeatureT>(signatures, p, keypoint_indices, feature_type);
    }

    void getModelsAndTransforms(std::vector<std::string> &models_verified, std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &transforms_verified) const
    {
        models_verified = model_ids_verified_;
        transforms_verified = transforms_verified_;
    }

    void getModelsAndTransforms(std::vector<ModelTPtr> &models_verified,
                                std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &transforms_verified) const
    {
        models_verified = models_verified_;
        transforms_verified = transforms_verified_;
    }

    void getAllHypotheses(std::vector<std::string> &models, std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &transforms) const
    {
        models = model_ids_;
        transforms = *transforms_;
    }

    void setModelsAndTransforms(const std::vector<std::string> &models, const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &transforms)
    {
        aligned_models_.resize(models.size());
        model_ids_.resize(models.size());
        *transforms_ = transforms;

        models_->clear();       // NOT IMPLEMENTED YET!!

        for(size_t i=0; i<models.size(); i++)
        {
            boost::filesystem::path modelpath(models[i]);
            model_ids_[i] =  modelpath.filename().string();
            PointInTPtr pModelPCl ( new pcl::PointCloud<pcl::PointXYZRGB> );
            PointInTPtr pModelPClTransformed ( new pcl::PointCloud<pcl::PointXYZRGB> );
            PointInTPtr pModelPCl2 ( new pcl::PointCloud<pcl::PointXYZRGB> );
            pcl::io::loadPCDFile ( models[i], * ( pModelPCl ) );

            pcl::transformPointCloud ( *pModelPCl, *pModelPClTransformed, transforms[i] );

            pcl::VoxelGrid<pcl::PointXYZRGB> sor;
            float leaf = 0.005f;
            sor.setLeafSize ( leaf, leaf, leaf );
            sor.setInputCloud ( pModelPClTransformed );
            sor.filter ( *pModelPCl2 );

            aligned_models_[i] = pModelPCl2;
            //models_ = models;
            //transforms_ = transforms;
        }
    }

    void setModelsAndTransforms(const std::vector<ModelTPtr> &models, const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &transforms)
    {
        aligned_models_.resize(models.size());
        aligned_normals_.resize(models.size());
        model_ids_.resize(models.size());
        *transforms_ = transforms;
        *models_ = models;
//        aligned_smooth_faces_.resize (models_->size ());

        for(size_t i=0; i<models.size(); i++)
        {
//            ModelTPtr m_with_faces;
//            model_only_source_->getModelById(models.at(i)->id_, m_with_faces);

            ConstPointInTPtr model_cloud = models.at(i)->getAssembled (hv_params_.resolution_);
            pcl::PointCloud<PointT>::Ptr model_aligned (new pcl::PointCloud<PointT>);
            pcl::transformPointCloud (*model_cloud, *model_aligned, transforms[i]);
            aligned_models_[i] = model_aligned;
            model_ids_[i] = models.at(i)->id_;

//            pcl::PointCloud<pcl::PointXYZL>::Ptr faces = models.at(i)->getAssembledSmoothFaces(hv_params_.resolution_);
//            pcl::PointCloud<pcl::PointXYZL>::Ptr faces_aligned(new pcl::PointCloud<pcl::PointXYZL>);
//            pcl::transformPointCloud (*faces, *faces_aligned, transforms[i]);
//            aligned_smooth_faces_ [i] = faces_aligned;

            pcl::PointCloud<pcl::Normal>::ConstPtr normal_cloud_const = models.at(i)->getNormalsAssembled (hv_params_.resolution_);
            pcl::PointCloud<pcl::Normal>::Ptr normal_cloud(new pcl::PointCloud<pcl::Normal>(*normal_cloud_const) );

            const Eigen::Matrix3f rot   = transforms_->at(i).block<3, 3> (0, 0);
//            const Eigen::Vector3f trans = transforms_->at(i).block<3, 1> (0, 3);
            for(size_t jj=0; jj < normal_cloud->points.size(); jj++)
            {
                const pcl::Normal norm_pt = normal_cloud->points[jj];
                normal_cloud->points[jj].getNormalVector3fMap() = rot * norm_pt.getNormalVector3fMap();
            }
            aligned_normals_[i] = normal_cloud;
        }
    }

    void setInputCloud(const pcl::PointCloud<PointT>::ConstPtr pInputCloud,
                       const pcl::PointCloud<pcl::Normal>::ConstPtr pSceneNormals)
    {
        pcl::copyPointCloud(*pInputCloud, *pInputCloud_);
        pcl::copyPointCloud(*pSceneNormals, *pSceneNormals_);
        model_ids_verified_.clear();
        transforms_verified_.clear();
        models_verified_.clear();
        aligned_models_.clear();
        model_ids_.clear();

        if(transforms_)
            transforms_->clear();

//        if(models_)
//            models_->clear();
    }

    void setInputCloud(const pcl::PointCloud<PointT>::ConstPtr pInputCloud)
    {
        v4r::common::computeNormals(pInputCloud, pSceneNormals_, sv_params_.normal_computation_method_);
        setInputCloud(pInputCloud, pSceneNormals_);
    }

    void poseRefinement()
    {
        multi_recog_->getPoseRefinement(models_, transforms_);
    }

    bool hypothesesVerification(std::vector<bool> &mask_hv);
//    bool hypothesesVerificationGpu(std::vector<bool> &mask_hv);

    void multiplaneSegmentation();

    void visualizeHypotheses();

    void constructHypotheses();

    void preFilterWithFSV(const pcl::PointCloud<PointT>::ConstPtr scene_cloud, std::vector<float> &fsv);

    void constructHypothesesFromFeatureMatches(std::map < std::string,v4r::rec_3d_framework::ObjectHypothesis<PointT> > hypothesesInput,
                                               pcl::PointCloud<PointT>::Ptr pKeypoints,
                                               pcl::PointCloud<pcl::Normal>::Ptr pKeypointNormals,
                                               std::vector<Hypothesis<PointT> > &hypothesesOutput,
                                               std::vector <pcl::Correspondences> &corresp_clusters);

    /**
     * @brief retraining the models - this function needs to be called as soon as object models are added, updated or deleted to the training/model directory
     * @param model_ids - name of object models
     * @return
     */
    bool retrain (const std::vector<std::string> &model_ids);

    void printParams() const;
};
}

#endif //SINGLEVIEW_OBJECT_RECOGNIZER_H
