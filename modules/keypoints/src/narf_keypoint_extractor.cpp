#include <v4r/keypoints/narf_keypoint_extractor.h>
#include <pcl/visualization/range_image_visualizer.h>
#include <pcl/range_image/range_image_planar.h>
#include <pcl/features/range_image_border_extractor.h>
#include <pcl/keypoints/narf_keypoint.h>

namespace v4r
{

template<typename PointT>
void
NarfKeypointExtractor<PointT>::compute ()
{
    Eigen::Affine3f sensorPose =
            Eigen::Affine3f(Eigen::Translation3f(input_->sensor_origin_[0], input_->sensor_origin_[1], input_->sensor_origin_[2]))
            * Eigen::Affine3f(input_->sensor_orientation_);
    pcl::RangeImagePlanar rangeImage;
    rangeImage.createFromPointCloudWithFixedSize(*input_, param_.cam_->getWidth(), param_.cam_->getHeight(),
            param_.cam_->getCx(), param_.cam_->getCy(), param_.cam_->getFocalLength(), param_.cam_->getFocalLength(),
            sensorPose, pcl::RangeImage::CAMERA_FRAME, param_.noise_level_, param_.minimum_range_);

    pcl::RangeImageBorderExtractor borderExtractor;
    // Keypoint detection object.
    pcl::NarfKeypoint detector(&borderExtractor);
    detector.setRangeImage(&rangeImage);
    // The support size influences how big the surface of interest will be,
    // when finding keypoints from the border information.
    detector.getParameters().support_size = param_.support_size_;
    detector.getParameters().min_distance_between_interest_points = param_.min_distance_between_interest_points_;
    detector.getParameters().optimal_distance_to_high_surface_change = param_.optimal_distance_to_high_surface_change_;
    detector.getParameters().min_interest_value = param_.min_interest_value_;
    detector.getParameters().min_surface_change_score = param_.min_surface_change_score_;
    detector.getParameters().optimal_range_image_patch_size = param_.optimal_range_image_patch_size_;
    pcl::PointCloud<int> sampled_indices;
    detector.compute(sampled_indices);

    keypoint_indices_.resize(sampled_indices.points.size());
    for(size_t i=0; i < sampled_indices.points.size(); i++)
        keypoint_indices_[i] = sampled_indices.points[i];
}


template class V4R_EXPORTS NarfKeypointExtractor<pcl::PointXYZ>;
template class V4R_EXPORTS NarfKeypointExtractor<pcl::PointXYZRGB>;
}
