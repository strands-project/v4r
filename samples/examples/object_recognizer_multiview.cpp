#include <v4r/common/miscellaneous.h>   // to extract Pose intrinsically stored in pcd file
#include <v4r/io/filesystem.h>
#include <v4r/recognition/multiview_object_recognizer.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/passthrough.h>
#include <iostream>
#include <boost/program_options.hpp>
#include <boost/format.hpp>
#include <glog/logging.h>

namespace po = boost::program_options;


int
main (int argc, char ** argv)
{
    typedef pcl::PointXYZRGB PointT;
    typedef v4r::Model<PointT> ModelT;
    typedef boost::shared_ptr<ModelT> ModelTPtr;

    std::string test_dir;
    bool visualize = false;

    google::InitGoogleLogging(argv[0]);

    po::options_description desc("Multiview Object Instance Recognizer\n======================================**Reference(s): Faeulhammer et al, ICRA / MVA 2015\n **Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("test_dir,t", po::value<std::string>(&test_dir)->required(), "Directory with test scenes stored as point clouds (.pcd). The camera pose is taken directly from the pcd header fields \"sensor_orientation_\" and \"sensor_origin_\" (if the test directory contains subdirectories, each subdirectory is considered as seperate sequence for multiview recognition)")
        ("visualize,v", po::bool_switch(&visualize), "visualize recognition results")
   ;
    po::variables_map vm;
    po::parsed_options parsed = po::command_line_parser(argc, argv).options(desc).allow_unregistered().run();
    po::store(parsed, vm);
    if (vm.count("help")) { std::cout << desc << std::endl; }
    try { po::notify(vm); }
    catch(std::exception& e) { std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl;  }

    v4r::MultiviewRecognizer<PointT> mv_r(argc, argv);


    // -----------TEST--------------
    std::vector< std::string> sub_folder_names = v4r::io::getFoldersInDirectory( test_dir);
    if( sub_folder_names.empty() )
        sub_folder_names.push_back("");

    for (size_t sub_folder_id=0; sub_folder_id < sub_folder_names.size(); sub_folder_id++)
    {
        const std::string sequence_path = test_dir + "/" + sub_folder_names[ sub_folder_id ];
        mv_r.set_scene_name(sequence_path);

        std::vector< std::string > views = v4r::io::getFilesInDirectory(sequence_path, ".*.pcd", false);
        for (size_t v_id=0; v_id<views.size(); v_id++)
        {
            const std::string fn = test_dir + "/" + sub_folder_names[sub_folder_id] + "/" + views[ v_id ];

            LOG(INFO) << "Recognizing file " << fn;
            typename pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
            pcl::io::loadPCDFile(fn, *cloud);

            Eigen::Matrix4f tf = v4r::RotTrans2Mat4f(cloud->sensor_orientation_, cloud->sensor_origin_);

            // reset view point otherwise pcl visualization is potentially messed up
            cloud->sensor_orientation_ = Eigen::Quaternionf::Identity();
            cloud->sensor_origin_ = Eigen::Vector4f::Zero();

            mv_r.setInputCloud (cloud);
            mv_r.setCameraPose(tf);
            mv_r.recognize();

            std::vector<ModelTPtr> verified_models = mv_r.getVerifiedModels();
            std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transforms_verified = mv_r.getVerifiedTransforms();

            if (visualize)
                mv_r.visualize();

            for(size_t m_id=0; m_id<verified_models.size(); m_id++)
                std::cout << "******" << verified_models[m_id]->id_ << std::endl <<  transforms_verified[m_id] << std::endl;
        }
        mv_r.cleanUp(); // delete all stored information from last sequences
    }
    return 0;
}
