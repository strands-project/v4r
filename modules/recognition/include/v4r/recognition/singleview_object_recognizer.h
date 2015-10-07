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
};

namespace v4r
{
class V4R_EXPORTS SingleViewRecognizer
{
protected:
    typedef pcl::PointXYZRGB PointT;
    typedef pcl::PointCloud<PointT> PointInT;
    typedef pcl::PointCloud<PointT>::Ptr PointInTPtr;
    typedef pcl::PointCloud<PointT>::ConstPtr ConstPointInTPtr;
    typedef v4r::Model<PointT> ModelT;
    typedef boost::shared_ptr<ModelT> ModelTPtr;
    typedef flann::L1<float> DistT;
    typedef pcl::Histogram<128> FeatureT;

    boost::shared_ptr<v4r::MultiRecognitionPipeline<PointT> > multi_recog_;

    std::map<std::string, v4r::ObjectHypothesis<PointT> > hypotheses_;
    boost::shared_ptr< pcl::PointCloud<PointT> > pKeypointsMultipipe_;
    pcl::PointIndices keypointIndices_;
    cv::Ptr<SiftGPU> sift_;
    pcl::PointCloud<PointT>::Ptr pInputCloud_;
    pcl::PointCloud<pcl::Normal>::Ptr pSceneNormals_;
    std::vector<ModelTPtr> models_, models_verified_;
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transforms_, transforms_verified_;
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr> aligned_models_;
    std::vector<pcl::PointCloud<pcl::Normal>::ConstPtr> aligned_normals_;
    std::vector<v4r::PlaneModel<PointT> > planes_found_;
    std::vector<pcl::PointCloud<PointT>::Ptr> verified_planes_;

    boost::shared_ptr < v4r::CorrespondenceGrouping<PointT, PointT> > cast_cg_alg_;

    boost::shared_ptr<pcl::visualization::PCLVisualizer> vis_;
    int vp1_, vp2_, vp3_;

public:
    struct hv_params{
            bool requires_normals_;
            bool initial_status_;
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

    float resolution_;

    std::string models_dir_;
    std::string training_dir_;
    std::string idx_flann_fn_sift_;
    std::string idx_flann_fn_shot_;

    SingleViewRecognizer ()
    {
        resolution_ = 0.005f;

        sv_params_.do_sift_ = true;
        sv_params_.do_shot_ = false;
        sv_params_.do_ourcvfh_ = false;
        sv_params_.add_planes_ = true;
        sv_params_.knn_shot_ = 1;
        sv_params_.knn_sift_ = 5;
        sv_params_.normal_computation_method_ = 2;

        sv_params_.icp_iterations_ = 0;
        sv_params_.icp_type_ = 1;
        sv_params_.chop_at_z_ = 2.0f;

        hv_params_.requires_normals_ = false;

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

        idx_flann_fn_sift_ = "sift_flann.idx";
        idx_flann_fn_shot_ = "shot_flann.idx";
    }

    bool recognize ();

    void initialize();

    void getVerifiedPlanes(std::vector<pcl::PointCloud<PointT>::Ptr> &planes) const
    {
        planes = verified_planes_;
    }

    void setModels_dir(const std::string &models_dir)
    {
        models_dir_ = models_dir;
    }

    void set_training_dir(const std::string &dir)
    {
        training_dir_ = dir;
    }

    cv::Ptr<SiftGPU> getSift() const
    {
        return sift_;
    }

    void set_sift(const cv::Ptr<SiftGPU> &value)
    {
        sift_ = value;
    }

    void getSavedHypotheses(std::map<std::string, v4r::ObjectHypothesis<PointT> > & hypotheses) const
    {
        hypotheses = hypotheses_;
    }

    void getKeypointsMultipipe (boost::shared_ptr<pcl::PointCloud<PointT> > &pKeypointsMultipipe ) const
    {
        pKeypointsMultipipe = pKeypointsMultipipe_;
    }

    void getKeypointIndices(pcl::PointIndices &keypointIndices) const
    {
        keypointIndices = keypointIndices_;
    }

    template <template<class > class Distance, typename FeatureT>
    void setFeatAndKeypoints(typename pcl::PointCloud<FeatureT>::Ptr & signatures, pcl::PointIndices & keypoint_indices, size_t feature_type)
    {
        multi_recog_->setFeatAndKeypoints<Distance, FeatureT>(signatures, keypoint_indices, feature_type);
    }

    void getModelsAndTransforms(std::vector<ModelTPtr> &models_verified,
                                std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &transforms_verified) const
    {
        models_verified = models_verified_;
        transforms_verified = transforms_verified_;
    }

    void setModelsAndTransforms(const std::vector<ModelTPtr> &models, const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &transforms)
    {
        aligned_models_.resize(models.size());
        aligned_normals_.resize(models.size());
        transforms_ = transforms;
        models_ = models;

        for(size_t i=0; i<models.size(); i++)
        {
            ConstPointInTPtr model_cloud = models.at(i)->getAssembled (resolution_);
            pcl::PointCloud<PointT>::Ptr model_aligned (new pcl::PointCloud<PointT>);
            pcl::transformPointCloud (*model_cloud, *model_aligned, transforms[i]);
            aligned_models_[i] = model_aligned;

            pcl::PointCloud<pcl::Normal>::ConstPtr normal_cloud_const = models.at(i)->getNormalsAssembled (resolution_);
            pcl::PointCloud<pcl::Normal>::Ptr normal_cloud(new pcl::PointCloud<pcl::Normal>(*normal_cloud_const) );

            const Eigen::Matrix3f rot = transforms_[i].block<3, 3> (0, 0);
//            const Eigen::Vector3f trans = transforms_->at(i).block<3, 1> (0, 3);
            for(size_t jj=0; jj < normal_cloud->points.size(); jj++)
            {
                const pcl::Normal norm_pt = normal_cloud->points[jj];
                normal_cloud->points[jj].getNormalVector3fMap() = rot * norm_pt.getNormalVector3fMap();
            }
            aligned_normals_[i] = normal_cloud;
        }
    }

    void setInputCloud(const pcl::PointCloud<PointT>::Ptr &cloud,
                       const pcl::PointCloud<pcl::Normal>::Ptr &normals)
    {
        pInputCloud_ = cloud;
        pSceneNormals_ = normals;
        transforms_.clear();
        transforms_verified_.clear();
        models_.clear();
        models_verified_.clear();
        aligned_models_.clear();
    }

    void setInputCloud(const pcl::PointCloud<PointT>::Ptr &cloud)
    {
        pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>());
        v4r::computeNormals<PointT>(cloud, normals, sv_params_.normal_computation_method_);
        setInputCloud(cloud, normals);
    }

    void poseRefinement()
    {
        multi_recog_->getPoseRefinement(models_, transforms_);
    }

    std::vector<bool> hypothesesVerification();
//    bool hypothesesVerificationGpu(std::vector<bool> &mask_hv);

    void multiplaneSegmentation();

    void constructHypotheses();

    void preFilterWithFSV(const pcl::PointCloud<PointT>::ConstPtr scene_cloud, std::vector<float> &fsv);

    void constructHypothesesFromFeatureMatches(std::map < std::string,v4r::ObjectHypothesis<PointT> > hypothesesInput,
                                               const pcl::PointCloud<PointT>::Ptr pKeypoints,
                                               const pcl::PointCloud<pcl::Normal>::Ptr pKeypointNormals,
                                               std::vector<Hypothesis<PointT> > &hypothesesOutput,
                                               std::vector <pcl::Correspondences> &corresp_clusters);

    /**
     * @brief retraining the models - this function needs to be called as soon as object models are added, updated or deleted to the training/model directory
     * @param model_ids - name of object models
     * @return
     */
    bool retrain (const std::vector<std::string> &model_ids = std::vector<std::string>());

    void printParams(std::ostream &ostr = std::cout) const;

    void visualize();
};
}

#endif //SINGLEVIEW_OBJECT_RECOGNIZER_H
