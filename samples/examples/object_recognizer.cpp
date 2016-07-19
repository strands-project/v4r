#include <v4r/common/miscellaneous.h>  // to extract Pose intrinsically stored in pcd file

#include <v4r/io/filesystem.h>
#include <v4r/recognition/multi_pipeline_recognizer.h>
#include <v4r/recognition/object_hypothesis.h>

#include <pcl/common/time.h>
#include <pcl/filters/passthrough.h>

#include <iostream>
#include <sstream>

#include <boost/any.hpp>
#include <boost/program_options.hpp>
#include <boost/format.hpp>
#include <glog/logging.h>

namespace po = boost::program_options;

int
main (int argc, char ** argv)
{
    typedef pcl::PointXYZRGB PointT;

    std::string test_dir;
    bool visualize = false;
    double chop_z = std::numeric_limits<double>::max();

    google::InitGoogleLogging(argv[0]);

    po::options_description desc("Single-View Object Instance Recognizer\n======================================\n**Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("test_dir,t", po::value<std::string>(&test_dir)->required(), "Directory with test scenes stored as point clouds (.pcd). The camera pose is taken directly from the pcd header fields \"sensor_orientation_\" and \"sensor_origin_\" (if the test directory contains subdirectories, each subdirectory is considered as seperate sequence for multiview recognition)")
        ("visualize,v", po::bool_switch(&visualize), "visualize recognition results")
        ("chop_z,z", po::value<double>(&chop_z)->default_value(chop_z, boost::str(boost::format("%.2e") % chop_z) ), "points with z-component higher than chop_z_ will be ignored (low chop_z reduces computation time and false positives (noise increase with z)")
   ;
    po::variables_map vm;
    po::parsed_options parsed = po::command_line_parser(argc, argv).options(desc).allow_unregistered().run();
    std::vector<std::string> to_pass_further = po::collect_unrecognized(parsed.options, po::include_positional);
    po::store(parsed, vm);
    if (vm.count("help")) { std::cout << desc << std::endl; to_pass_further.push_back("-h"); }
    try { po::notify(vm); }
    catch(std::exception& e) { std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl;  }

    v4r::MultiRecognitionPipeline<PointT> r(to_pass_further);

    // ----------- TEST ----------
    std::vector< std::string> sub_folder_names = v4r::io::getFoldersInDirectory( test_dir );
    if(sub_folder_names.empty())
        sub_folder_names.push_back("");

    for (const std::string &sub_folder_name : sub_folder_names)
    {
        const std::string sequence_path = test_dir + "/" + sub_folder_name;

        std::vector< std::string > views = v4r::io::getFilesInDirectory(sequence_path, ".*.pcd", false);
        for (size_t v_id=0; v_id<views.size(); v_id++)
        {
            const std::string fn = sequence_path + "/" + views[ v_id ];

            LOG(INFO) << "Recognizing file " << fn;
            typename pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
            pcl::io::loadPCDFile(fn, *cloud);

            //reset view point - otherwise this messes up PCL's visualization (this does not affect recognition results)
            cloud->sensor_orientation_ = Eigen::Quaternionf::Identity();
            cloud->sensor_origin_ = Eigen::Vector4f::Zero(4);

//            if( chop_z > 0)
//            {
//                pcl::PassThrough<PointT> pass;
//                pass.setFilterLimits ( 0.f, chop_z );
//                pass.setFilterFieldName ("z");
//                pass.setInputCloud (cloud);
//                pass.setKeepOrganized (true);
//                pass.filter (*cloud);
//            }

            r.setInputCloud (cloud);
            r.recognize();

            std::vector<typename v4r::ObjectHypothesis<PointT>::Ptr > ohs = r.getVerifiedHypotheses();


//            std::vector<ModelTPtr> verified_models = r.getVerifiedModels();
//            std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transforms_verified;
//            transforms_verified = r.getVerifiedTransforms();

            if (visualize)
                r.visualize();

            for(size_t m_id=0; m_id<ohs.size(); m_id++)
            {
                std::cout << "********************" << ohs[m_id]->model_->id_ << std::endl
                          << ohs[m_id]->transform_ << std::endl << std::endl;
            }
        }
    }
}
