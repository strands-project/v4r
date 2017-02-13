#include <iostream>

#include <opencv2/opencv.hpp>

#include <pcl/common/time.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>

#include <v4r/common/normals.h>
#include <v4r/io/filesystem.h>
#include <v4r/common/miscellaneous.h>
#include <v4r/segmentation/all_headers.h>
#include <v4r/segmentation/types.h>
#include <v4r/segmentation/plane_utils.h>
#include <v4r/segmentation/segmentation_utils.h>

#include <boost/any.hpp>
#include <boost/program_options.hpp>
#include <glog/logging.h>

namespace po = boost::program_options;

template <typename PointT>
void
save_bb_image(const pcl::PointCloud<PointT> &cloud, const std::vector<std::vector<int> > &segments, const std::string &filename_prefix, int margin = 0)
{
    for(size_t i=0; i < segments.size(); i++)
    {
        std::stringstream filename; filename << filename_prefix << "_" << std::setfill('0') << std::setw(5) << i << ".jpg";
        int min_u, min_v, max_u, max_v;
        max_u = max_v = 0;
        min_u = cloud.width;
        min_v = cloud.height;

        for( int idx : segments[i] )
        {
            int u = idx % cloud.width;
            int v = idx / cloud.width;

            if (u>max_u)
                max_u = u;

            if (v>max_v)
                max_v = v;

            if (u<min_u)
                min_u = u;

            if (v<min_v)
                min_v = v;
        }

        min_u = std::max (0, min_u - margin);
        min_v = std::max (0, min_v - margin);
        max_u = std::min ((int)cloud.width, max_u + margin);
        max_v = std::min ((int)cloud.height, max_v + margin);

        int img_width  = max_u - min_u;
        int img_height = max_v - min_v;
        cv::Mat_<cv::Vec3b> image(img_height, img_width);

        for (int row = 0; row < img_height; row++)
        {
          for (int col = 0; col < img_width; col++)
          {
            cv::Vec3b & cvp = image.at<cv::Vec3b> (row, col);
            int position = (row + min_v) * cloud.width + (col + min_u);
            const PointT &pt = cloud.points[position];

            cvp[0] = pt.b;
            cvp[1] = pt.g;
            cvp[2] = pt.r;
          }
        }
        cv::imwrite(filename.str(), image);
    }
}


int
main (int argc, char ** argv)
{
    typedef pcl::PointXYZRGB PointT;
    std::string test_dir, out_dir = "/tmp/segmentation/";
    int segmentation_method = v4r::SegmentationType::EuclideanSegmentation;
    int plane_extraction_method = v4r::PlaneExtractionType::OrganizedMultiplane;
    int normal_computation_method = 2;
    int margin = 0;
    bool save_bounding_boxes = true;
    bool visualize = false;

    google::InitGoogleLogging(argv[0]);

    po::options_description desc("Point Cloud Segmentation Example\n======================================\n**Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("test_dir,t", po::value<std::string>(&test_dir)->required(), "Directory with test scenes stored as point clouds (.pcd).")
        ("out_dir,o", po::value<std::string>(&out_dir)->default_value(out_dir), "Output directory.")
        ("segmentation_method", po::value<int>(&segmentation_method)->default_value(segmentation_method), "segmentation method")
        ("plane_extraction_method", po::value<int>(&plane_extraction_method)->default_value(plane_extraction_method), "plane extraction method")
        ("normal_computation_method,n", po::value<int>(&normal_computation_method)->default_value(normal_computation_method), "normal computation method (if needed by segmentation approach)")
        ("margin", po::value<int>(&margin)->default_value(margin), "margin when computing bounding box")
        ("save_bb", po::value<bool>(&save_bounding_boxes)->default_value(save_bounding_boxes), "if true, saves bounding boxes")
        ("visualize,v", po::bool_switch(&visualize), "If set, visualizes segmented clusters.")
    ;
    po::variables_map vm;
    po::parsed_options parsed = po::command_line_parser(argc, argv).options(desc).allow_unregistered().run();
    std::vector<std::string> to_pass_further = po::collect_unrecognized(parsed.options, po::include_positional);
    po::store(parsed, vm);
    if (vm.count("help")) { std::cout << desc << std::endl; }
    try { po::notify(vm); }
    catch(std::exception& e) { std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl;  }


    typename v4r::PlaneExtractor<PointT>::Ptr plane_extractor = v4r::initPlaneExtractor<PointT> ( plane_extraction_method, to_pass_further );
    typename v4r::Segmenter<PointT>::Ptr segmenter = v4r::initSegmenter<PointT>( segmentation_method, to_pass_further);

    if( !to_pass_further.empty() )
    {
        std::cerr << "Unused command line arguments: ";
        for(size_t c=0; c<to_pass_further.size(); c++)
            std::cerr << to_pass_further[c] << " ";

        std::cerr << "!" << std::endl;
    }


    std::vector< std::string> sub_folder_names = v4r::io::getFoldersInDirectory( test_dir );
    if(sub_folder_names.empty())
        sub_folder_names.push_back("");

    std::sort(sub_folder_names.begin(), sub_folder_names.end());
    for (size_t sub_folder_id=0; sub_folder_id < sub_folder_names.size(); sub_folder_id++)
    {
        const std::string sequence_path = test_dir + "/" + sub_folder_names[ sub_folder_id ];
        const std::string out_path = out_dir + "/" + sub_folder_names[ sub_folder_id ];
        v4r::io::createDirIfNotExist(out_path);

        std::vector< std::string > views = v4r::io::getFilesInDirectory(sequence_path, ".*.pcd", false);

        for (size_t v_id=0; v_id<views.size(); v_id++)
        {
            const std::string fn = test_dir + "/" + sub_folder_names[sub_folder_id] + "/" + views[ v_id ];
            std::string out_fn_prefix = out_path + "/" + views[ v_id ];
            boost::replace_last(out_fn_prefix, ".pcd", "");

            std::cout << "Segmenting file " << fn << std::endl;

            typename pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
            pcl::PointCloud<pcl::Normal>::Ptr normals;

            pcl::io::loadPCDFile(fn, *cloud);

            if( segmenter->getRequiresNormals() || plane_extractor->getRequiresNormals() )
            {
                pcl::ScopeTime t("Normal computation");
                normals.reset(new pcl::PointCloud<pcl::Normal>);
                v4r::computeNormals<PointT>(cloud, normals, normal_computation_method);
                (void)t;
            }

            std::vector< Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> > planes;
            {
                pcl::ScopeTime t("Plane extraction");
                plane_extractor->setInputCloud(cloud);
                plane_extractor->setNormalsCloud( normals );
                plane_extractor->compute();
                planes = plane_extractor->getPlanes();
                (void)t;
            }

            float plane_inlier_threshold = 0.02f;
            size_t selected_plane = 0;

            if(visualize)
            {
                for(const Eigen::Vector4f &plane : planes )
                    v4r::visualizePlane<PointT> ( cloud, plane, 0.02f, "plane" );
            }

            std::vector<int> above_plane_indices = v4r::get_above_plane_inliers( *cloud, planes[ selected_plane], plane_inlier_threshold );
            boost::dynamic_bitset<> above_plane_mask = v4r::createMaskFromIndices( above_plane_indices, cloud->points.size() );

            pcl::PointCloud<PointT>::Ptr above_plane_cloud (new pcl::PointCloud<PointT> (*cloud));
            for(size_t i=0; i<above_plane_cloud->points.size(); i++) // keep organized
            {
                if( !above_plane_mask[i] )
                {
                    PointT &p = above_plane_cloud->points[i];
                    p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
                }
            }

            {
                pcl::ScopeTime t("Segmentation");
                segmenter->setInputCloud(above_plane_cloud);
                segmenter->setNormalsCloud( normals );
                segmenter->segment();
                (void)t;
            }

            std::vector<std::vector<int> > indices;
            segmenter->getSegmentIndices(indices);

            if(visualize)
            {
                //reset view point - otherwise this messes up PCL's visualization (this does not affect recognition results)
                cloud->sensor_orientation_ = Eigen::Quaternionf::Identity();
                cloud->sensor_origin_ = Eigen::Vector4f::Zero(4);
                v4r::visualizeClusters<PointT>( cloud, indices, "segmented clusters" );
            }

            if( save_bounding_boxes )
                save_bb_image( *cloud, indices, out_fn_prefix, margin );
        }
    }

    return 0;
}
