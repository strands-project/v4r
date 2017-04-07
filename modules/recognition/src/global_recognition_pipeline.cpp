#include <v4r/common/miscellaneous.h>
#include <v4r/features/types.h>
#include <v4r/segmentation/plane_utils.h>
#include <v4r/recognition/global_recognition_pipeline.h>

#include <pcl/common/time.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <glog/logging.h>

namespace v4r
{

template<typename PointT>
void
GlobalRecognitionPipeline<PointT>::initialize(const std::string &trained_dir, bool force_retrain)
{
    CHECK ( !global_recognizers_.empty() ) << "No local recognizers provided!";

    for(typename GlobalRecognizer<PointT>::Ptr &r : global_recognizers_)
    {
        r->setModelDatabase( m_db_ );
        r->setNormalEstimator( normal_estimator_ );
        r->setVisualizationParameter(vis_param_);
        r->initialize(trained_dir, force_retrain);
    }
}

template<typename PointT>
void
GlobalRecognitionPipeline<PointT>::recognize()
{
    CHECK(seg_);
    planes_.clear();
    clusters_.clear();

    obj_hypotheses_.clear();

    seg_->setInputCloud(scene_);
    seg_->setNormalsCloud(scene_normals_);
    seg_->segment();
    seg_->getSegmentIndices(clusters_);

    obj_hypotheses_.resize(clusters_.size()); // each cluster builds a hypothesis group

    if(visualize_clusters_)
    {
        obj_hypotheses_wo_elongation_check_.clear();
        obj_hypotheses_wo_elongation_check_.resize(clusters_.size() );
    }

    size_t kept=0;
    for(size_t i=0; i<clusters_.size(); i++)
    {
        ObjectHypothesesGroup<PointT> &ohg = obj_hypotheses_[kept];
        ohg.ohs_.clear();
        ohg.global_hypotheses_ = true;

        typename GlobalRecognizer<PointT>::Cluster::Ptr cluster (
                    new typename GlobalRecognizer<PointT>::Cluster (*scene_, clusters_[i] ) );

        cluster->setTablePlane( table_plane_ );

        ///TODO pragma omp parallel with lock
        for (size_t g_id=0; g_id<global_recognizers_.size(); g_id++)
        {
            typename GlobalRecognizer<PointT>::Ptr r = global_recognizers_[g_id];
            r->setInputCloud( scene_ );
            r->setSceneNormals( scene_normals_ );
            r->setCluster( cluster );
            r->recognize();
            std::vector<typename ObjectHypothesis<PointT>::Ptr > ohs = r->getFilteredHypotheses();
            ohg.ohs_.insert( ohg.ohs_.end(), ohs.begin(), ohs.end() );

            if(visualize_clusters_)
            {
                std::vector<typename ObjectHypothesis<PointT>::Ptr > ohs_unfiltered = r->getAllHypotheses();
                obj_hypotheses_wo_elongation_check_[i].ohs_.insert( obj_hypotheses_wo_elongation_check_[i].ohs_.end(), ohs_unfiltered.begin(), ohs_unfiltered.end() );
                obj_hypotheses_wo_elongation_check_[i].global_hypotheses_ = true;
            }
        }

        if(!ohg.ohs_.empty())
            kept++;
    }
    obj_hypotheses_.resize(kept);

    if (visualize_clusters_)
    {
        visualize();
        obj_hypotheses_wo_elongation_check_.clear();
    }
}


template<typename PointT>
void
GlobalRecognitionPipeline<PointT>::visualize()
{
    static pcl::visualization::PCLVisualizer::Ptr vis_;
    static int vp1_, vp2_, vp3_, vp4_, vp5_;

    if(!vis_)
    {
        vis_.reset ( new pcl::visualization::PCLVisualizer("Global recognition results") );
        vis_->createViewPort(0  , 0  , 0.33, 0.5, vp1_);
        vis_->createViewPort(0.33, 0  , 0.66  , 0.5, vp2_);
        vis_->createViewPort(0.66, 0  , 1  , 0.5, vp3_);
        vis_->createViewPort(0  , 0.5, 0.33, 1 , vp4_);
        vis_->createViewPort(0.33, 0.5, 0.66  ,  1, vp5_);
//        vis_->createViewPort(0.66, 0.5, 1  ,  1, vp6_);

        vis_->setBackgroundColor(vis_param_->bg_color_[0], vis_param_->bg_color_[1], vis_param_->bg_color_[2], vp1_);
        vis_->setBackgroundColor(vis_param_->bg_color_[0], vis_param_->bg_color_[1], vis_param_->bg_color_[2], vp2_);
        vis_->setBackgroundColor(vis_param_->bg_color_[0], vis_param_->bg_color_[1], vis_param_->bg_color_[2], vp3_);
        vis_->setBackgroundColor(vis_param_->bg_color_[0], vis_param_->bg_color_[1], vis_param_->bg_color_[2], vp4_);
        vis_->setBackgroundColor(vis_param_->bg_color_[0], vis_param_->bg_color_[1], vis_param_->bg_color_[2], vp5_);
//        vis_->setBackgroundColor(1,1,1,vp6_);
    }
    vis_->removeAllPointClouds();
    vis_->removeAllShapes();

#if PCL_VERSION >= 100800
    vis_->removeAllCoordinateSystems(vp3_);
    vis_->removeAllCoordinateSystems(vp4_);
        for(size_t co_id=0; co_id<coordinate_axis_ids_global_.size(); co_id++)
            vis_->removeCoordinateSystem( coordinate_axis_ids_global_[co_id] );
        coordinate_axis_ids_global_.clear();
#endif
    vis_->addPointCloud(scene_, "cloud", vp1_);

    if(!vis_param_->no_text_)
        vis_->addText("input cloud", 10, 10, vis_param_->fontsize_, vis_param_->text_color_[0], vis_param_->text_color_[1], vis_param_->text_color_[2], "input", vp1_);


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
    vis_->addPointCloud(colored_cloud,"segments", vp2_);
    if(!vis_param_->no_text_)
        vis_->addText("segments", 10, 10, vis_param_->fontsize_, vis_param_->text_color_[0], vis_param_->text_color_[1], vis_param_->text_color_[2], "segments", vp2_);

    // VISUALIZE PLANE
    if (table_plane_set_)
    {
        pcl::visualization::PointCloudColorHandlerCustom<PointT> gray (scene_, 128, 128, 128);
        vis_->addPointCloud(scene_, gray, "input_vp3", vp3_);
        vis_->setPointCloudRenderingProperties( pcl::visualization::PCL_VISUALIZER_OPACITY, 0.2, "input_vp3");
        typename pcl::PointCloud<pcl::PointXYZRGB>::Ptr plane_cloud (new pcl::PointCloud<pcl::PointXYZRGB>());
        const std::vector<int> plane_inliers = get_all_plane_inliers(*scene_, table_plane_, 0.015f);
        pcl::copyPointCloud(*scene_, plane_inliers, *plane_cloud);
        for(PointT &p : plane_cloud->points)
        {
            p.r = p.b = 0.f;
            p.g = 255.f;
        }

        vis_->addPointCloud(plane_cloud, "plane", vp3_);
        if(!vis_param_->no_text_)
            vis_->addText("table plane", 10, 10, vis_param_->fontsize_, vis_param_->text_color_[0], vis_param_->text_color_[1], vis_param_->text_color_[2], "table plane", vp3_);
    }

    for(size_t i=0; i < obj_hypotheses_wo_elongation_check_.size(); i++)
    {
        const ObjectHypothesesGroup<PointT> &ohs = obj_hypotheses_wo_elongation_check_[i];
        for(size_t k=0; k<ohs.ohs_.size(); k++)
        {
            bool found_model;
            typename Model<PointT>::ConstPtr m = m_db_->getModelById( ohs.ohs_[k]->class_id_, ohs.ohs_[k]->model_id_, found_model );
            const Eigen::Matrix4f &tf = ohs.ohs_[k]->transform_;
//            float conf = ohs.ohs_[k]->confidence_;

            const std::string model_id = m->id_.substr(0, m->id_.length() - 4);
            std::stringstream model_label;
            model_label << model_id << "_" << i << "_" << k << "_vp3";
            typename pcl::PointCloud<PointT>::Ptr model_aligned ( new pcl::PointCloud<PointT>() );
            typename pcl::PointCloud<PointT>::ConstPtr model_cloud = m->getAssembled( 7 );
            pcl::transformPointCloud( *model_cloud, *model_aligned, tf);
            vis_->addPointCloud(model_aligned, model_label.str(), vp5_);

#if PCL_VERSION >= 100800
            Eigen::Matrix3f rot_tmp  = tf.block<3,3>(0,0);
            Eigen::Vector3f trans_tmp = tf.block<3,1>(0,3);
            Eigen::Affine3f affine_trans;
            affine_trans.fromPositionOrientationScale(trans_tmp, rot_tmp, Eigen::Vector3f::Ones());
            std::stringstream co_id; co_id << i << "vp3";
            vis_->addCoordinateSystem(0.15f, affine_trans, co_id.str(), vp5_);
            coordinate_axis_ids_global_.push_back(co_id.str());
#endif
        }
    }
    if(!vis_param_->no_text_)
        vis_->addText("all hypotheses", 10, 10, vis_param_->fontsize_, vis_param_->text_color_[0], vis_param_->text_color_[1], vis_param_->text_color_[2], "all_hypotheses", vp5_);


    size_t disp_id=0;
    for(size_t i=0; i < obj_hypotheses_.size(); i++)
    {
        const ObjectHypothesesGroup<PointT> &ohs = obj_hypotheses_[i];
        for(size_t k=0; k<ohs.ohs_.size(); k++)
        {
            bool found_model;
            typename Model<PointT>::ConstPtr m = m_db_->getModelById( ohs.ohs_[k]->class_id_, ohs.ohs_[k]->model_id_, found_model );
            const Eigen::Matrix4f &tf = ohs.ohs_[k]->transform_;
            float conf = ohs.ohs_[k]->confidence_;

            std::stringstream model_id; model_id << m->id_ << ": " << conf;
            std::stringstream unique_id; unique_id << i << "_" << k;
//            vis_->addText(model_id.str(), 12, 12 + 12*disp_id, 10,
//                          rgb_cluster_colors(0, i)/255.f,
//                          rgb_cluster_colors(1, i)/255.f,
//                          rgb_cluster_colors(2, i)/255.f,
//                          unique_id.str(), vp2_);
            disp_id++;


            std::stringstream model_label, model_label_refined;
            model_label << model_id.str() << "_" << i << "_" << k << "_vp4";
            model_label_refined << model_id.str() << "_" << i << "_" << k << "_vp5";
            typename pcl::PointCloud<PointT>::Ptr model_aligned ( new pcl::PointCloud<PointT>() );
            typename pcl::PointCloud<PointT>::ConstPtr model_cloud = m->getAssembled( 7 );
            pcl::transformPointCloud( *model_cloud, *model_aligned, tf);
            vis_->addPointCloud(model_aligned, model_label.str(), vp4_);

#if PCL_VERSION >= 100800
            Eigen::Matrix3f rot_tmp  = tf.block<3,3>(0,0);
            Eigen::Vector3f trans_tmp = tf.block<3,1>(0,3);
            Eigen::Affine3f affine_trans;
            affine_trans.fromPositionOrientationScale(trans_tmp, rot_tmp, Eigen::Vector3f::Ones());
            std::stringstream co_id; co_id << i << "vp4";
            vis_->addCoordinateSystem(0.15f, affine_trans, co_id.str(), vp4_);
            coordinate_axis_ids_global_.push_back(co_id.str());
#endif
        }
    }

    if(!vis_param_->no_text_)
        vis_->addText("filtered hypotheses", 10, 10, vis_param_->fontsize_, vis_param_->text_color_[0], vis_param_->text_color_[1], vis_param_->text_color_[2], "filtered_hypotheses", vp4_);

    vis_->spin();
}

template class V4R_EXPORTS GlobalRecognitionPipeline<pcl::PointXYZRGB>;
}
