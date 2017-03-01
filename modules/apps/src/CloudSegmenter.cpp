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
    int plane_extraction_method = v4r::PlaneExtractionType::OrganizedMultiplane;
    int normal_computation_method = v4r::NormalEstimatorType::PCL_INTEGRAL_NORMAL;

    po::options_description desc("Point Cloud Segmentation\n======================================\n**Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("segmentation_method", po::value<int>(&segmentation_method)->default_value(segmentation_method), "segmentation method")
        ("plane_extraction_method", po::value<int>(&plane_extraction_method)->default_value(plane_extraction_method), "plane extraction method")
        ("normal_computation_method,n", po::value<int>(&normal_computation_method)->default_value(normal_computation_method), "normal computation method (if needed by segmentation approach)")
        ("plane_inlier_threshold", po::value<float>(&plane_inlier_threshold_)->default_value(plane_inlier_threshold_), "inlier threshold for plane")
        ("chop_z,z", po::value<float>(&chop_z_)->default_value(chop_z_), "cut-off threshold in meter")
;
    po::variables_map vm;
    po::parsed_options parsed = po::command_line_parser(command_line_arguments).options(desc).allow_unregistered().run();
    command_line_arguments = po::collect_unrecognized(parsed.options, po::include_positional);
    po::store(parsed, vm);
    if (vm.count("help")) { std::cout << desc << std::endl; }
    try { po::notify(vm); }
    catch(std::exception& e) { std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl;  }

    plane_extractor_ = v4r::initPlaneExtractor<PointT> ( plane_extraction_method, command_line_arguments );
    segmenter_ = v4r::initSegmenter<PointT>( segmentation_method, command_line_arguments);

    if( segmenter_->getRequiresNormals() || plane_extractor_->getRequiresNormals() )
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
CloudSegmenter<PointT>::segment(typename pcl::PointCloud<PointT>::Ptr &cloud)
{
    pcl::PointCloud<pcl::Normal>::Ptr normals;

    if( segmenter_->getRequiresNormals() || plane_extractor_->getRequiresNormals() )
    {
        pcl::ScopeTime t("Normal computation");
        normal_estimator_->setInputCloud( cloud );
        normals.reset(new pcl::PointCloud<pcl::Normal>);
        normals = normal_estimator_->compute();
        (void)t;
    }

    for (PointT &p : cloud->points )
    {
        if( pcl::isFinite(p) && p.z > chop_z_ )
            p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
    }

    {
        pcl::ScopeTime t("Plane extraction");
        plane_extractor_->setInputCloud(cloud);
        plane_extractor_->setNormalsCloud( normals );
        plane_extractor_->compute();
        planes_ = plane_extractor_->getPlanes();
        (void)t;
    }

    typename pcl::PointCloud<PointT>::Ptr above_plane_cloud;

    if(planes_.empty())
    {
        std::cout << " Could not extract any plane with the chosen parameters. Segmenting the whole input cloud!" << std::endl;
        above_plane_cloud = cloud;
    }
    else
    {
        size_t selected_plane = 0;
        if (planes_.size() > 1 )
        {
            std::cout << "Extracted multiple planes from input cloud. Selecting the one with the most inliers with the chosen plane inlier threshold of " << plane_inlier_threshold_ << std::endl;

            size_t max_plane_inliers = 0;
            for(size_t plane_id=0; plane_id<planes_.size(); plane_id++)
            {
                std::vector<int> plane_indices = v4r::get_all_plane_inliers( *cloud, planes_[ selected_plane], plane_inlier_threshold_ );
                if( plane_indices.size() > max_plane_inliers )
                {
                    selected_plane = plane_id;
                    max_plane_inliers = plane_indices.size();
                }
            }
        }
        std::vector<int> above_plane_indices = v4r::get_above_plane_inliers( *cloud, planes_[ selected_plane], plane_inlier_threshold_ );
        boost::dynamic_bitset<> above_plane_mask = v4r::createMaskFromIndices( above_plane_indices, cloud->points.size() );

        above_plane_cloud.reset (new pcl::PointCloud<PointT> (*cloud));
        for(size_t i=0; i<above_plane_cloud->points.size(); i++) // keep organized
        {
            if( !above_plane_mask[i] )
            {
                PointT &p = above_plane_cloud->points[i];
                p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }

    {
        pcl::ScopeTime t("Segmentation");
        segmenter_->setInputCloud(above_plane_cloud);
        segmenter_->setNormalsCloud( normals ); // since the cloud was kept organized, we can use the original normal cloud
        segmenter_->segment();
        (void)t;
    }

    segmenter_->getSegmentIndices(found_clusters_);
}

#define PCL_INSTANTIATE_CloudSegmenter(T) template class V4R_EXPORTS CloudSegmenter<T>;
PCL_INSTANTIATE(CloudSegmenter, PCL_XYZ_POINT_TYPES )

}
}
