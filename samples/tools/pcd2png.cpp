#include <iostream>
#include <sstream>
#include <fstream>
#include <v4r/io/filesystem.h>
#include <v4r/common/pcl_opencv.h>
#include <v4r/common/faat_3d_rec_framework_defines.h>
#include <pcl/io/pcd_io.h>
#include <opencv2/opencv.hpp>

#include <boost/program_options.hpp>
#include <glog/logging.h>

namespace po = boost::program_options;

int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);
    std::string input_dir, output_dir = "/tmp/output_images/";
    const std::string view_prefix = "cloud";
    const std::string pose_prefix = "pose";
    const std::string indices_prefix = "object_indices";
    size_t img_height = 256, img_width = 256;

    po::options_description desc("Converter from RGB-D pointclouds (XYZRGB) into rgb image and depth map image\n======================================\n**Allowed options");
    desc.add_options()
            ("help,h", "produce help message")
            ("input_dir,i", po::value<std::string>(&input_dir)->required(), "input directory containing the .pcd files to be converted")
            ("output_dir,o", po::value<std::string>(&output_dir)->default_value(output_dir), "output directory")
            ("image_height,h", po::value<size_t>(&img_height)->default_value(img_height), "height of the output image")
            ("image_width,w", po::value<size_t>(&img_width)->default_value(img_width), "width of the output image")
    ;

   po::variables_map vm;
   po::store(po::parse_command_line(argc, argv, desc), vm);
   if (vm.count("help"))
   {
       std::cout << desc << std::endl;
       return false;
   }

   try  { po::notify(vm); }
   catch(std::exception& e)
   {
       std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl;
       return false;
   }

    std::vector< std::string> sub_folder_names = v4r::io::getFoldersInDirectory( input_dir );

    if( sub_folder_names.empty() )
        sub_folder_names.push_back("");

    std::vector<std::pair<std::string, size_t> > file2label;

    size_t id = 0;
    v4r::io::createDirIfNotExist(output_dir);

    for (size_t sub_folder_id=0; sub_folder_id < sub_folder_names.size(); sub_folder_id++)
    {
        const std::string sequence_path = input_dir + "/" + sub_folder_names[ sub_folder_id ];

        std::vector< std::string > views = v4r::io::getFilesInDirectory(sequence_path, ".*.pcd", false);

        for (size_t v_id=0; v_id<views.size(); v_id++)
        {
            const std::string input_fn = sequence_path + "/" + views[v_id];

            std::cout << "Converting image " << input_fn << std::endl;
            pcl::PointCloud<pcl::PointXYZRGB> cloud;
            pcl::io::loadPCDFile(input_fn, cloud);

            // Read object indices
            std::string indices_file (views[v_id]);
            boost::replace_all (indices_file, view_prefix, indices_prefix);
            pcl::PointCloud<IndexPoint> obj_indices_cloud;
            const std::string oi_file = sequence_path + "/" + + "/" + indices_file;
            pcl::io::loadPCDFile (oi_file, obj_indices_cloud);
            pcl::PointIndices indices;
            indices.indices.resize(obj_indices_cloud.points.size());
            for(size_t kk=0; kk < obj_indices_cloud.points.size(); kk++)
                indices.indices[kk] = obj_indices_cloud.points[kk].idx;


            // Read camera pose
            //            std::string pose_file (views[v_id]);
            //            boost::replace_all (pose_file, view_prefix, pose_prefix);


            // convert point cloud
            cv::Mat rgb =  v4r::ConvertPCLCloud2Image(cloud);
            cv::Mat rgb_cropped = v4r::ConvertPCLCloud2Image(cloud, indices.indices, img_height, img_width);
            cv::Mat depth = v4r::ConvertPCLCloud2DepthImage(cloud);
            cv::Mat depth_cropped = v4r::ConvertPCLCloud2DepthImageFixedSize(cloud, indices.indices, img_height, img_width);

            cv::Mat depth_mm = 1000./depth_cropped;

            cv::Mat depth_mm_ushort;
            depth_mm.convertTo(depth_mm_ushort, CV_16U);


            // save to disk
            std::stringstream fn; fn << output_dir << "/" << id << "_rgb.png";
            cv::imwrite(fn.str(), rgb);
            fn.str(""); fn << output_dir << "/" << id << "_rgb_cropped.png";
            cv::imwrite(fn.str(), rgb_cropped);
            fn.str(""); fn << output_dir << "/" << id << "_depth.png";
            cv::imwrite(fn.str(), depth);
            fn.str(""); fn << output_dir << "/" << id << "_depth_cropped.png";
            cv::imwrite(fn.str(), depth_cropped);
            fn.str(""); fn << output_dir << "/" << id << "_depth_cropped_unsigned.png";
            cv::imwrite(fn.str(), depth_mm_ushort);

            // add to file list
            file2label.push_back( std::pair<std::string, size_t> ( fn.str(), sub_folder_id ));
            id++;
        }
    }

    const std::string fn = output_dir + "/fileAndLabelList.txt";
    std::ofstream f( fn.c_str() );
    for(size_t i=0; i < file2label.size(); i++)
        f << file2label[i].first << " " << file2label[i].second << std::endl;

   f.close();

}
