#include <iostream>

#include <opencv2/opencv.hpp>

#include <pcl/common/time.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>

#include <v4r/common/normals.h>
#include <v4r/io/filesystem.h>
#include <v4r/apps/CloudSegmenter.h>
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

    int margin = 0;
    bool save_bounding_boxes = true;
    bool visualize = false;

    google::InitGoogleLogging(argv[0]);

    po::options_description desc("Point Cloud Segmentation Example\n======================================\n**Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("test_dir,t", po::value<std::string>(&test_dir)->required(), "Directory with test scenes stored as point clouds (.pcd).")
        ("out_dir,o", po::value<std::string>(&out_dir)->default_value(out_dir), "Output directory.")
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

    v4r::apps::CloudSegmenterParameter param;
    to_pass_further = param.init(to_pass_further);
    v4r::apps::CloudSegmenter<PointT> cs(param);
    cs.initialize(to_pass_further);

    std::vector< std::string> sub_folder_names = v4r::io::getFoldersInDirectory( test_dir );
    if(sub_folder_names.empty())
        sub_folder_names.push_back("");

    std::sort(sub_folder_names.begin(), sub_folder_names.end());
    for (size_t sub_folder_id=0; sub_folder_id < sub_folder_names.size(); sub_folder_id++)
    {
        bf::path sequence_path = test_dir;
        sequence_path /= sub_folder_names[ sub_folder_id ];

        bf::path out_path = out_dir;
        out_path /= sub_folder_names[ sub_folder_id ];

        v4r::io::createDirIfNotExist(out_path);

        std::vector< std::string > views = v4r::io::getFilesInDirectory(sequence_path, ".*.pcd", false);

        for (size_t v_id=0; v_id<views.size(); v_id++)
        {
            bf::path fn = sequence_path;
            fn /= views[v_id];

            std::string out_basename = views[ v_id ];
            boost::replace_last(out_basename, ".pcd", "");

            bf::path out_fn = out_dir;
            out_fn /= out_basename;

            std::cout << "Segmenting file " << fn.string() << std::endl;

            typename pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);

            pcl::io::loadPCDFile(fn.string(), *cloud);
            //reset view point - otherwise this messes up PCL's visualization (this does not affect recognition results)
            cloud->sensor_orientation_ = Eigen::Quaternionf::Identity();
            cloud->sensor_origin_ = Eigen::Vector4f::Zero(4);
            cs.segment(cloud);
            std::vector<std::vector<int> > indices = cs.getClusters();

            if(visualize)
            {
                //reset view point - otherwise this messes up PCL's visualization (this does not affect recognition results)
                cloud->sensor_orientation_ = Eigen::Quaternionf::Identity();
                cloud->sensor_origin_ = Eigen::Vector4f::Zero(4);
                v4r::visualizeClusters<PointT> ( cloud, cs.getPlaneInliers(), "planes" );
                v4r::visualizeClusters<PointT>( cloud, indices, "segmented clusters" );
            }

            if( save_bounding_boxes )
                save_bb_image( *cloud, indices, out_fn.string(), margin );
        }
    }

    return 0;
}
