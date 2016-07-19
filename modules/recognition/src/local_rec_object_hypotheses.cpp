#include <v4r/recognition/local_rec_object_hypotheses.h>

namespace v4r
{
template <typename PointT>
void
LocalObjectHypothesis<PointT>::visualize(const pcl::PointCloud<pcl::PointXYZRGB> & scene,
                                    const pcl::PointCloud<pcl::PointXYZRGB> & scene_kp) const
{
    (void)scene;
    (void)scene_kp;
    std::cerr << "This function is not implemented for this point cloud type!" << std::endl;
}

template <>
void
LocalObjectHypothesis<pcl::PointXYZRGB>::visualize(const pcl::PointCloud<pcl::PointXYZRGB> & scene,
                                              const pcl::PointCloud<pcl::PointXYZRGB> & scene_kp) const
{
    if(!vis_)
        vis_.reset(new pcl::visualization::PCLVisualizer("correspondences for hypothesis"));

    vis_->removeAllPointClouds();
    vis_->removeAllShapes();
    vis_->addPointCloud(scene.makeShared(), "scene");
    vis_->setBackgroundColor(1,1,1);
    pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr model_cloud = model_->getAssembled( 3 );
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr model_aligned ( new pcl::PointCloud<pcl::PointXYZRGB>() );
    pcl::copyPointCloud(*model_cloud, *model_aligned);
    vis_->addPointCloud(model_aligned, "model_aligned");
//    vis_->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal> (model_->keypoints_, model_->kp_normals_, 10, 0.05, "normals_model");

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr kp_colored_scene ( new pcl::PointCloud<pcl::PointXYZRGB>() );
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr kp_colored_model ( new pcl::PointCloud<pcl::PointXYZRGB>() );
    kp_colored_scene->points.resize(model_scene_corresp_.size());
    kp_colored_model->points.resize(model_scene_corresp_.size());

//    pcl::Correspondences sorted_corrs = model_scene_corresp_;
//    std::sort (sorted_corrs.begin (), sorted_corrs.end (), this->gcGraphCorrespSorter);

    for(size_t j=0; j<5; j++)
    {
        for (size_t i=(size_t)((j/5.f)*model_scene_corresp_.size()); i<(size_t)(((j+1.f)/5.f)*model_scene_corresp_.size()); i++)
        {
            const pcl::Correspondence &c = model_scene_corresp_[i];
            pcl::PointXYZRGB kp_m = model_->keypoints_->points[c.index_query];
            pcl::PointXYZRGB kp_s = scene_kp[c.index_match];

            const float r = kp_m.r = kp_s.r = 100 + rand() % 155;
            const float g = kp_m.g = kp_s.g = 100 + rand() % 155;
            const float b = kp_m.b = kp_s.b = 100 + rand() % 155;
            kp_colored_scene->points[i] = kp_s;
            kp_colored_model->points[i] = kp_m;

            std::stringstream ss; ss << "correspondence " << i << j;
            vis_->addLine(kp_s, kp_m, r/255, g/255, b/255, ss.str());
            vis_->addSphere(kp_s, 0.005f, r/255, g/255, b/255, ss.str() + "kp_s");
            vis_->addSphere(kp_m, 0.005f, r/255, g/255, b/255, ss.str() + "kp_m");
        }

        vis_->addPointCloud(scene_kp.makeShared(), "kps_scene");
        vis_->addPointCloud(kp_colored_model, "kps_model");
        vis_->spin();
    }
}

template class V4R_EXPORTS LocalObjectHypothesis<pcl::PointXYZRGB>;
template class V4R_EXPORTS LocalObjectHypothesis<pcl::PointXYZ>;

}

