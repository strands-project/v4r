#include <v4r/apps/CloudSegmenter.h>
#include <v4r/common/miscellaneous.h>
#include <v4r/segmentation/plane_utils.h>
#include <v4r/segmentation/segmentation_utils.h>
#include <v4r/segmentation/types.h>

#include <pcl/common/time.h>
#include <pcl/impl/instantiate.hpp>

#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <glog/logging.h>

namespace po = boost::program_options;

namespace v4r
{
namespace apps
{

template<typename PointT>
void
CloudSegmenter<PointT>::initialize(std::vector<std::string> &command_line_arguments)
{
    int segmentation_method = v4r::SegmentationType::OrganizedConnectedComponents;
    int plane_extraction_method = v4r::PlaneExtractionType::Tile;
    int normal_computation_method = v4r::NormalEstimatorType::PCL_INTEGRAL_NORMAL;

    po::options_description desc("Point Cloud Segmentation\n======================================\n**Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("segmentation_method", po::value<int>(&segmentation_method)->default_value(segmentation_method), "segmentation method")
        ("plane_extraction_method", po::value<int>(&plane_extraction_method)->default_value(plane_extraction_method), "plane extraction method")
        ("normal_computation_method,n", po::value<int>(&normal_computation_method)->default_value(normal_computation_method), "normal computation method (if needed by segmentation approach)")
    ;
    po::variables_map vm;
    po::parsed_options parsed = po::command_line_parser(command_line_arguments).options(desc).allow_unregistered().run();
    command_line_arguments = po::collect_unrecognized(parsed.options, po::include_positional);
    po::store(parsed, vm);
    if (vm.count("help")) { std::cout << desc << std::endl; }
    try { po::notify(vm); }
    catch(std::exception& e) { std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl;  }

    plane_extractor_ = v4r::initPlaneExtractor<PointT> ( plane_extraction_method, command_line_arguments );

    if(!param_.skip_segmentation_)
        segmenter_ = v4r::initSegmenter<PointT>( segmentation_method, command_line_arguments);

    if( ( (segmenter_ && segmenter_->getRequiresNormals()) ||
          (plane_extractor_ && plane_extractor_->getRequiresNormals()) ) )
        normal_estimator_ = v4r::initNormalEstimator<PointT> ( normal_computation_method, command_line_arguments );

    if( !command_line_arguments.empty() )
    {
        std::cerr << "Unused command line arguments: ";
        for(size_t c=0; c<command_line_arguments.size(); c++)
            std::cerr << command_line_arguments[c] << " ";

        std::cerr << "!" << std::endl;
    }
}

template<typename PointT>
void
CloudSegmenter<PointT>::segment(const typename pcl::PointCloud<PointT>::ConstPtr &cloud)
{
    processed_cloud_.reset (new pcl::PointCloud<PointT>(*cloud));

    if( !normals_ &&
            ( (segmenter_ && segmenter_->getRequiresNormals()) ||
              (plane_extractor_ && plane_extractor_->getRequiresNormals()) ) )
    {
        pcl::ScopeTime t("Normal computation");
        normal_estimator_->setInputCloud( cloud );
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        normals = normal_estimator_->compute();
        normals_ = normals;
        (void)t;
    }

    for (PointT &p : processed_cloud_->points )
    {
        if( pcl::isFinite(p) && p.z > param_.chop_z_ )
            p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
    }

    if( !param_.skip_plane_extraction_ )
    {
        pcl::ScopeTime t("Plane extraction");
        plane_extractor_->setInputCloud(processed_cloud_);
        plane_extractor_->setNormalsCloud( normals_ );
        plane_extractor_->compute();
        planes_ = plane_extractor_->getPlanes();
        plane_inliers_ = plane_extractor_->getPlaneInliers();

        if(planes_.empty())
        {
            LOG(WARNING) << " Could not extract any plane with the chosen parameters. Segmenting the whole input cloud!";
        }
        else // get plane inliers
        {
            if( plane_inliers_.size() != planes_.size()) // the plane inliers are not already extracted by the algorithm - do it explicity
            {
                plane_inliers_.resize(planes_.size());
                for(size_t plane_id=0; plane_id<planes_.size(); plane_id++)
                    plane_inliers_[plane_id] = v4r::get_all_plane_inliers( *processed_cloud_, planes_[ plane_id], param_.plane_inlier_threshold_ );
            }

            // remove planes without sufficient plane inliers
            size_t kept=0;
            for(size_t plane_id=0; plane_id<planes_.size(); plane_id++)
            {
                if(plane_inliers_[plane_id].size() > param_.min_plane_inliers_)
                {
                    plane_inliers_[kept] = plane_inliers_[plane_id];
                    planes_[kept] = planes_[plane_id];
                    kept++;
                }
            }
            plane_inliers_.resize(kept);
            planes_.resize(kept);

            // get dominant plane id
            size_t dominant_plane_id = 0;
            size_t max_inliers = 0;
            for(size_t plane_id=0; plane_id<plane_inliers_.size(); plane_id++)
            {
                if( plane_inliers_[plane_id].size() > max_inliers )
                {
                    max_inliers = plane_inliers_[plane_id].size();
                    dominant_plane_id = plane_id;
                }
            }
            size_t selected_plane_id = dominant_plane_id;
            // check if there is a plane higher than the dominant plane which is parallel to it
            if(param_.use_highest_plane_)
            {
                for(size_t plane_id=0; plane_id<plane_inliers_.size(); plane_id++)
                {
                    if (selected_plane_id == plane_id)
                        continue;

                    const Eigen::Vector4f &current_highest_plane = planes_[selected_plane_id];
                    const Eigen::Vector4f &plane_tmp = planes_[plane_id];

                    if( CosAngleBetweenPlanes(plane_tmp, current_highest_plane) > param_.cosinus_angle_for_planes_to_be_parallel_ ) // check if parallel
                    {
                        //check if higher
                        if ( plane_tmp(3) / plane_tmp.head(3).norm() < current_highest_plane(3) / current_highest_plane.head(3).norm())
                        {
                            selected_plane_id = plane_id;
                        }
                    }
                }
            }


            selected_plane_ = planes_[selected_plane_id];

            // now filter
            {
                if( param_.remove_planes_ || param_.remove_selected_plane_)
                {
                    for(size_t plane_id=0; plane_id<plane_inliers_.size(); plane_id++)
                    {
                        if( plane_id != selected_plane_id && !param_.remove_planes_ )
                            continue;

                        for( int idx : plane_inliers_[plane_id] )
                        {
                            PointT &p = processed_cloud_->points[idx];
                            p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
                        }
                    }
                }


                if( param_.remove_points_below_selected_plane_)
                {
                    std::vector<int> above_plane_indices = v4r::get_above_plane_inliers( *processed_cloud_, planes_[ selected_plane_id], param_.min_distance_to_plane_ );
                    boost::dynamic_bitset<> above_plane_mask = v4r::createMaskFromIndices( above_plane_indices, processed_cloud_->points.size() );

                    for(size_t i=0; i<processed_cloud_->points.size(); i++) // keep organized
                    {
                        if( !above_plane_mask[i] )
                        {
                            PointT &p = processed_cloud_->points[i];
                            p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
                        }
                    }
                }
            }
        }
        (void)t;
    }


    if(!param_.skip_segmentation_)
    {
        pcl::ScopeTime t("Segmentation");
        segmenter_->setInputCloud( processed_cloud_ );
        segmenter_->setNormalsCloud( normals_ ); // since the cloud was kept organized, we can use the original normal cloud
        segmenter_->segment();
        segmenter_->getSegmentIndices(found_clusters_);
        (void)t;
    }
}

#define PCL_INSTANTIATE_CloudSegmenter(T) template class V4R_EXPORTS CloudSegmenter<T>;
PCL_INSTANTIATE(CloudSegmenter, PCL_XYZ_POINT_TYPES )

}
}
