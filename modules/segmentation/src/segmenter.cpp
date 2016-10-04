#include <v4r/segmentation/segmenter.h>

namespace v4r
{

template<typename PointT>
void
Segmenter<PointT>::visualize() const
{
    if(!vis_)
    {
        vis_.reset ( new pcl::visualization::PCLVisualizer("Segmentation Results") );
        vis_->createViewPort(0,0,0.33,1,vp1_);
        vis_->createViewPort(0.33,0,0.66,1,vp2_);
        vis_->createViewPort(0.66,0,1,1,vp3_);
    }
    vis_->removeAllPointClouds();
    vis_->removeAllShapes();
    vis_->addPointCloud(scene_, "cloud", vp1_);


    typename pcl::PointCloud<PointT>::Ptr colored_cloud (new pcl::PointCloud<PointT>());
    for(size_t i=0; i < clusters_.size(); i++)
    {
        pcl::PointCloud<PointT> cluster;
        pcl::copyPointCloud(*scene_, clusters_[i], cluster);

        const uint8_t r = rand()%255;
        const uint8_t g = rand()%255;
        const uint8_t b = rand()%255;
        for(size_t pt_id=0; pt_id<cluster.points.size(); pt_id++)
        {
            cluster.points[pt_id].r = r;
            cluster.points[pt_id].g = g;
            cluster.points[pt_id].b = b;
        }
        *colored_cloud += cluster;
    }
    vis_->addPointCloud(colored_cloud,"segments", vp2_);

    typename pcl::PointCloud<PointT>::Ptr planes (new pcl::PointCloud<PointT>(*scene_));

    Eigen::Matrix3Xf plane_colors(3, all_planes_.size());
    for(size_t i=0; i<all_planes_.size(); i++)
    {
        plane_colors(0, i) = rand()%255;
        plane_colors(1, i) = rand()%255;
        plane_colors(2, i) = rand()%255;
    }

    for(PointT &pt :planes->points)
    {
        if ( !pcl::isFinite( pt ) )
            continue;

        const Eigen::Vector4f xyz_p = pt.getVector4fMap ();
        pt.g = pt.b = pt.r = 0;


        for(size_t i=0; i<all_planes_.size(); i++)
        {
            float val = xyz_p.dot(all_planes_[i]->coefficients_);

            if ( std::abs(val) < 0.02f)
            {
                pt.r = plane_colors(0,i);
                pt.g = plane_colors(1,i);
                pt.b = plane_colors(2,i);
            }
        }

        float val = xyz_p.dot(dominant_plane_);

        if ( std::abs(val) < 0.02f)
            pt.r = 255;
    }
    vis_->addPointCloud(planes,"table plane", vp3_);
    vis_->addText("input", 10, 10, 15, 1, 1, 1, "input", vp1_);
    vis_->addText("segments", 10, 10, 15, 1, 1, 1, "segments", vp2_);
    vis_->addText("dominant plane", 10, 10, 15, 1, 1, 1, "dominant_plane", vp3_);
    vis_->addText("all other planes", 10, 25, 15, 1, 1, 1, "other_planes", vp3_);
    vis_->resetCamera();
    vis_->spin();
}

template class V4R_EXPORTS Segmenter<pcl::PointXYZRGB>;
}
