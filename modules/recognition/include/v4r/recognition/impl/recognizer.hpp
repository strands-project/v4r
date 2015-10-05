#include <v4r/recognition/recognizer.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

namespace v4r
{

template <typename PointT>
void
ObjectHypothesis<PointT>::visualize() const
{
    std::cerr << "This function is not implemented for this point cloud type!" << std::endl;
}

//template<typename PointT>
//ObjectHypothesis<PointT> &
//ObjectHypothesis<PointT>::operator+=(const ObjectHypothesis<PointT> &rhs)
//{
//    if (this->model_->id_.compare(rhs.model_->id_)!= 0)
//    {
//        std::cerr << "Models do not have same id. Cannot merge them!" << std::endl;
//        return *this;
//    }

//    size_t existing_corrs = this->model_scene_corresp_->size();
//    size_t new_corrs = rhs.model_scene_corresp_->size();

//    this->model_scene_corresp_->insert( this->model_scene_corresp_->  end(),
//                                         rhs.model_scene_corresp_->begin(),
//                                         rhs.model_scene_corresp_->  end() );


//    for (size_t c_id=0; c_id<new_corrs; c_id++)
//    {
//        const pcl::Correspondence &c_old =   rhs.model_scene_corresp_->at( c_id );
//        pcl::Correspondence &c_new = this->model_scene_corresp_->at( existing_corrs + c_id );

//        c_new.index_match = c_old.index_match + this->scene_keypoints->points.size();
//    }
//    *this->scene_keypoints += *rhs.scene_keypoints;
//    *this->scene_kp_normals_ += *rhs.scene_kp_normals_;

//    this->indices_to_flann_models_.insert(
//                this->indices_to_flann_models_.end(),
//                rhs.indices_to_flann_models_.begin(),
//                rhs.indices_to_flann_models_.end());

//    return *this;
//}

template <>
void
ObjectHypothesis<pcl::PointXYZRGB>::visualize() const
{
    if(!vis_)
    {
        vis_.reset(new pcl::visualization::PCLVisualizer("correspondences for hypothesis"));
//        vis_->createViewPort(0,0,0.5,1,vp1_);
//        vis_->createViewPort(0.5,0,1,1,vp2_);
    }

    pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr model_cloud = model_->getAssembled( 0.003f );
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr model_aligned ( new pcl::PointCloud<pcl::PointXYZRGB>() );
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_vis ( new pcl::PointCloud<pcl::PointXYZRGB>() );
    Eigen::Vector4f zero_origin; zero_origin[0] = zero_origin[1] = zero_origin[2] = zero_origin[3] = 0.f;
    pcl::copyPointCloud( *scene_, *scene_vis);
    scene_vis->sensor_origin_ = zero_origin;
    scene_vis->sensor_orientation_ = Eigen::Quaternionf::Identity();
    pcl::copyPointCloud( *model_cloud, *model_aligned);
    vis_->addPointCloud(scene_vis, "scene");
    vis_->addPointCloud(model_aligned, "model_aligned");
    vis_->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal> (model_->keypoints_, model_->kp_normals_, 10, 0.05, "normals_model");

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr kp_colored_scene ( new pcl::PointCloud<pcl::PointXYZRGB>() );
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr kp_colored_model ( new pcl::PointCloud<pcl::PointXYZRGB>() );
    kp_colored_scene->points.resize(model_scene_corresp_->size());
    kp_colored_model->points.resize(model_scene_corresp_->size());
    for (size_t i=0; i<model_scene_corresp_->size(); i++)
    {
        const pcl::Correspondence &c = model_scene_corresp_->at(i);
        pcl::PointXYZRGB kp_m = model_->keypoints_->points[c.index_query];
        pcl::PointXYZRGB kp_s = scene_->points[c.index_match];

        const float r = kp_m.r = kp_s.r = 100 + rand() % 155;
        const float g = kp_m.g = kp_s.g = 100 + rand() % 155;
        const float b = kp_m.b = kp_s.b = 100 + rand() % 155;
        kp_colored_scene->points[i] = kp_s;
        kp_colored_model->points[i] = kp_m;

        std::stringstream ss; ss << "correspondence " << i;
        vis_->addLine(kp_s, kp_m, r/255, g/255, b/255, ss.str());
        vis_->addSphere(kp_s, 2, r/255, g/255, b/255, ss.str() + "kp_s", vp1_);
        vis_->addSphere(kp_m, 2, r/255, g/255, b/255, ss.str() + "kp_m", vp1_);
    }

    vis_->addPointCloud(kp_colored_scene, "kps_s");
    vis_->addPointCloud(kp_colored_model, "kps_m");

    vis_->spin();
}

}
