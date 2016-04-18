#include <v4r/common/miscellaneous.h>  // to extract Pose intrinsically stored in pcd file

#include <v4r/io/eigen.h>
#include <v4r/io/filesystem.h>
#include <v4r/recognition/multi_pipeline_recognizer.h>

#include <pcl/common/time.h>
#include <pcl/filters/passthrough.h>

#include <iostream>
#include <sstream>
#include <time.h>

#include <boost/program_options.hpp>
#include <glog/logging.h>

namespace po = boost::program_options;

int
main (int argc, char ** argv)
{
    typedef pcl::PointXYZRGB PointT;
    typedef v4r::Model<PointT> ModelT;
    typedef boost::shared_ptr<ModelT> ModelTPtr;

    std::map<std::string, size_t> rec_models_per_id_;
    typedef pcl::PointXYZRGB PointT;
    std::string test_dir, out_dir = "/tmp/sv_object_recognizer_results/";
    bool visualize = false;
    double chop_z = std::numeric_limits<double>::max();

    google::InitGoogleLogging(argv[0]);

    po::options_description desc("Single-View Object Instance Recognizer\n======================================\n**Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("test_dir,t", po::value<std::string>(&test_dir)->required(), "Directory with test scenes stored as point clouds (.pcd). The camera pose is taken directly from the pcd header fields \"sensor_orientation_\" and \"sensor_origin_\" (if the test directory contains subdirectories, each subdirectory is considered as seperate sequence for multiview recognition)")
        ("visualize,v", po::bool_switch(&visualize), "visualize recognition results")
        ("chop_z,z", po::value<double>(&chop_z)->default_value(chop_z, boost::str(boost::format("%.2e") % chop_z) ), "points with z-component higher than chop_z_ will be ignored (low chop_z reduces computation time and false positives (noise increase with z)")
        ("out_dir,o", po::value<std::string>(&out_dir)->default_value(out_dir), "Output directory where recognition results will be stored.")
   ;
    po::variables_map vm;
    po::parsed_options parsed = po::command_line_parser(argc, argv).options(desc).allow_unregistered().run();
    po::store(parsed, vm);
    if (vm.count("help")) { std::cout << desc << std::endl; }
    try { po::notify(vm); }
    catch(std::exception& e) { std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl;  }

    v4r::io::createDirIfNotExist(out_dir);

    v4r::MultiRecognitionPipeline<PointT> r(argc, argv);

    // ----------- TEST ----------
    std::vector< std::string> sub_folder_names = v4r::io::getFoldersInDirectory( test_dir );
    if(sub_folder_names.empty())
        sub_folder_names.push_back("");

    for (const std::string &sub_folder_name : sub_folder_names)
    {
        const std::string sequence_path = test_dir + "/" + sub_folder_name;
        const std::string out_path = out_dir + "/" + sub_folder_name;
        v4r::io::createDirIfNotExist(out_path);

        rec_models_per_id_.clear();     // shouldn't this go inside next for?

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

            if( chop_z > 0)
            {
                pcl::PassThrough<PointT> pass;
                pass.setFilterLimits ( 0.f, chop_z );
                pass.setFilterFieldName ("z");
                pass.setInputCloud (cloud);
                pass.setKeepOrganized (true);
                pass.filter (*cloud);
            }

            r.setInputCloud (cloud);
            pcl::StopWatch watch;
            r.recognize();
            v4r::io::writeFloatToFile( out_path + "/" + views[v_id].substr(0, views[v_id].length()-4) + "_time.nfo", watch.getTimeSeconds());

            std::vector<ModelTPtr> verified_models = r.getVerifiedModels();
            std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transforms_verified;
            transforms_verified = r.getVerifiedTransforms();

            if (visualize)
                r.visualize();

            for(size_t m_id=0; m_id<verified_models.size(); m_id++)
            {
                LOG(INFO) << "********************" << verified_models[m_id]->id_ << std::endl;

                const std::string model_id = verified_models[m_id]->id_;
                const Eigen::Matrix4f tf = transforms_verified[m_id];

                size_t num_models_per_model_id;

                std::map<std::string, size_t>::iterator it_rec_mod;
                it_rec_mod = rec_models_per_id_.find(model_id);
                if(it_rec_mod == rec_models_per_id_.end())
                {
                    rec_models_per_id_.insert(std::pair<std::string, size_t>(model_id, 1));
                    num_models_per_model_id = 0;
                }
                else
                {
                    num_models_per_model_id = it_rec_mod->second;
                    it_rec_mod->second++;
                }

                std::stringstream out_fn;
                out_fn << out_path << "/" << views[v_id].substr(0, views[v_id].length()-4) << "_"
                       << model_id << "_" << num_models_per_model_id << ".txt";

                ofstream or_file (out_fn.str().c_str());
                for (size_t row=0; row <4; row++)
                    for(size_t col=0; col<4; col++)
                        or_file << tf(row, col) << " ";
                or_file.close();
            }
        }
    }
}
