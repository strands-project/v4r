#include <v4r/common/miscellaneous.h>
#include <v4r/recognition/global_recognition_pipeline.h>
#include <v4r/segmentation/plane_utils.h>
#include <v4r/features/types.h>

#include <pcl/common/time.h>
#include <glog/logging.h>

namespace v4r
{

template<typename PointT>
void
GlobalRecognitionPipeline<PointT>::initialize(const std::string &trained_dir, bool force_retrain)
{
    CHECK ( !global_recognizers_.empty() ) << "No local recognizers provided!";

    for(auto &r : global_recognizers_)
    {
        r->setModelDatabase( m_db_ );
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

    {
        pcl::ScopeTime t("Plane extraction");
        plane_extractor_->setInputCloud(scene_);
        plane_extractor_->setNormalsCloud( scene_normals_ );
        plane_extractor_->compute();
        planes_ = plane_extractor_->getPlanes();
        (void)t;
    }


    Eigen::Vector4f table_plane = Eigen::Vector4f::Zero();

    if(planes_.empty())
    {
        std::cout << " Could not extract any plane with the chosen parameters. Segmenting the whole input cloud!" << std::endl;
        seg_->setInputCloud(scene_);
    }
    else
    {
        typename pcl::PointCloud<PointT>::Ptr above_plane_cloud;

        size_t selected_plane = 0;
        if (planes_.size() > 1 )
        {
            std::cout << "Extracted multiple planes from input cloud. Selecting the one with the most inliers with the chosen plane inlier threshold of " << param_.plane_inlier_threshold_ << std::endl;

            size_t max_plane_inliers = 0;
            for(size_t plane_id=0; plane_id<planes_.size(); plane_id++)
            {
                std::vector<int> plane_indices = v4r::get_all_plane_inliers( *scene_, planes_[ selected_plane], param_.plane_inlier_threshold_ );
                if( plane_indices.size() > max_plane_inliers )
                {
                    selected_plane = plane_id;
                    max_plane_inliers = plane_indices.size();
                }
            }
        }

        table_plane = planes_[selected_plane];
        std::vector<int> above_plane_indices = v4r::get_above_plane_inliers( *scene_, table_plane, param_.plane_inlier_threshold_ );
        boost::dynamic_bitset<> above_plane_mask = v4r::createMaskFromIndices( above_plane_indices, scene_->points.size() );

        above_plane_cloud.reset (new pcl::PointCloud<PointT> (*scene_));
        for(size_t i=0; i<above_plane_cloud->points.size(); i++) // keep organized
        {
            if( !above_plane_mask[i] )
            {
                PointT &p = above_plane_cloud->points[i];
                p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
            }
        }

        seg_->setInputCloud(above_plane_cloud);
    }

    seg_->setNormalsCloud(scene_normals_);
    seg_->segment();
    seg_->getSegmentIndices(clusters_);
//    bool plane_found = seg_->getDominantPlane(table_plane);

    obj_hypotheses_.resize(clusters_.size()); // each cluster builds a hypothesis group
    size_t kept=0;
    for(size_t i=0; i<clusters_.size(); i++)
    {
        ObjectHypothesesGroup<PointT> &ohg = obj_hypotheses_[kept];
        ohg.ohs_.clear();
        ohg.global_hypotheses_ = true;

        typename GlobalRecognizer<PointT>::Cluster::Ptr cluster (
                    new typename GlobalRecognizer<PointT>::Cluster (*scene_, clusters_[i] ) );

        cluster->setTablePlane( table_plane );

        ///TODO pragma omp parallel with lock
        for (size_t g_id=0; g_id<global_recognizers_.size(); g_id++)
        {
            typename GlobalRecognizer<PointT>::Ptr r = global_recognizers_[g_id];
            r->setInputCloud( scene_ );
            r->setCluster( cluster );
            r->recognize();
            std::vector<typename ObjectHypothesis<PointT>::Ptr > ohs = r->getHypotheses();
            ohg.ohs_.insert( ohg.ohs_.end(), ohs.begin(), ohs.end() );
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
    if(!vis_)
    {
        vis_.reset ( new pcl::visualization::PCLVisualizer("Global recognition results") );
        vis_->createViewPort(0  , 0  , 0.33, 0.5, vp1_);
        vis_->createViewPort(0.33, 0  , 0.66  , 0.5, vp2_);
        vis_->createViewPort(0.66, 0  , 1  , 0.5, vp3_);
        vis_->createViewPort(0  , 0.5, 0.33, 1 , vp4_);
        vis_->createViewPort(0.33, 0.5, 0.66  ,  1, vp5_);
//        vis_->createViewPort(0.66, 0.5, 1  ,  1, vp6_);

        vis_->setBackgroundColor(1,1,1,vp1_);
        vis_->setBackgroundColor(1,1,1,vp2_);
        vis_->setBackgroundColor(1,1,1,vp3_);
        vis_->setBackgroundColor(1,1,1,vp4_);
        vis_->setBackgroundColor(1,1,1,vp5_);
//        vis_->setBackgroundColor(1,1,1,vp6_);
    }
    vis_->removeAllPointClouds();
    vis_->removeAllShapes();
#if PCL_VERSION >= 100702
        for(size_t co_id=0; co_id<coordinate_axis_ids_global_.size(); co_id++)
            vis_->removeCoordinateSystem( coordinate_axis_ids_global_[co_id] );
        coordinate_axis_ids_global_.clear();
#endif
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
    vis_->addPointCloud(colored_cloud,"segments", vp2_);


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
            typename pcl::PointCloud<PointT>::ConstPtr model_cloud = m->getAssembled( 10 );
            pcl::transformPointCloud( *model_cloud, *model_aligned, tf);
            vis_->addPointCloud(model_aligned, model_label.str(), vp3_);

    #if PCL_VERSION >= 100702
            Eigen::Matrix3f rot_tmp  = tf.block<3,3>(0,0);
            Eigen::Vector3f trans_tmp = tf.block<3,1>(0,3);
            Eigen::Affine3f affine_trans;
            affine_trans.fromPositionOrientationScale(trans_tmp, rot_tmp, Eigen::Vector3f::Ones());
            std::stringstream co_id; co_id << i << "vp3";
            vis_->addCoordinateSystem(0.15f, affine_trans, co_id.str(), vp3_);
            coordinate_axis_ids_global_.push_back(co_id.str());
    #endif
        }
    }

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
            typename pcl::PointCloud<PointT>::ConstPtr model_cloud = m->getAssembled( 10 );
            pcl::transformPointCloud( *model_cloud, *model_aligned, tf);
            vis_->addPointCloud(model_aligned, model_label.str(), vp4_);

    #if PCL_VERSION >= 100702
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
    vis_->spin();
}

template class V4R_EXPORTS GlobalRecognitionPipeline<pcl::PointXYZRGB>;
}
