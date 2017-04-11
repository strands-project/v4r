
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <glog/logging.h>

#include <v4r/apps/ObjectRecognizer.h>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <v4r/io/filesystem.h>

namespace po = boost::program_options;

int
main (int argc, char ** argv)
{
    typedef pcl::PointXYZRGB PT;

    std::string test_dir;
    std::string out_dir = "/tmp/object_recognition_results/";
    std::string debug_dir = "";
    std::string recognizer_config = "cfg/multipipeline_config.xml";
    int verbosity = -1;

    po::options_description desc("Single-View Object Instance Recognizer\n======================================\n**Allowed options");
    desc.add_options()
            ("help,h", "produce help message")
            ("test_dir,t", po::value<std::string>(&test_dir)->required(), "Directory with test scenes stored as point clouds (.pcd). The camera pose is taken directly from the pcd header fields \"sensor_orientation_\" and \"sensor_origin_\" (if the test directory contains subdirectories, each subdirectory is considered as seperate sequence for multiview recognition)")
            ("out_dir,o", po::value<std::string>(&out_dir)->default_value(out_dir), "Output directory where recognition results will be stored.")
            ("dbg_dir", po::value<std::string>(&debug_dir)->default_value(debug_dir), "Output directory where debug information (generated object hypotheses) will be stored (skipped if empty)")
            ("recognizer_config", po::value<std::string>(&recognizer_config)->default_value(recognizer_config), "Config XML of the multi-pipeline recognizer")
            ("verbosity", po::value<int>(&verbosity)->default_value(verbosity), "set verbosity level for output (<0 minimal output)")
            ;
    po::variables_map vm;
    po::parsed_options parsed = po::command_line_parser(argc, argv).options(desc).allow_unregistered().run();
    std::vector<std::string> to_pass_further = po::collect_unrecognized(parsed.options, po::include_positional);
    po::store(parsed, vm);
    if (vm.count("help")) { std::cout << desc << std::endl; to_pass_further.push_back("-h"); }
    try { po::notify(vm); }
    catch(std::exception& e) { std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl;  }

    if(verbosity>=0)
    {
        FLAGS_logtostderr = 1;
        FLAGS_v = verbosity;
        std::cout << "Enabling verbose logging." << std::endl;
    }
    google::InitGoogleLogging(argv[0]);

    v4r::apps::ObjectRecognizerParameter param(recognizer_config);
    v4r::apps::ObjectRecognizer<PT> recognizer (param);
    recognizer.initialize(to_pass_further);

    std::vector< std::string> sub_folder_names = v4r::io::getFoldersInDirectory( test_dir );
    if(sub_folder_names.empty()) sub_folder_names.push_back("");

    for (const std::string &sub_folder_name : sub_folder_names)
    {
        std::vector< std::string > views = v4r::io::getFilesInDirectory( test_dir+"/"+sub_folder_name, ".*.pcd", false );
        for (size_t v_id=0; v_id<views.size(); v_id++)
        {
            bf::path test_path = test_dir;
            test_path /= sub_folder_name;
            test_path /= views[v_id];

            std::vector<double> elapsed_time;

            LOG(INFO) << "Recognizing file " << test_path.string();
            pcl::PointCloud<PT>::Ptr cloud(new pcl::PointCloud<PT>());
            pcl::io::loadPCDFile( test_path.string(), *cloud);

            //reset view point - otherwise this messes up PCL's visualization (this does not affect recognition results)
            cloud->sensor_orientation_ = Eigen::Quaternionf::Identity();
            cloud->sensor_origin_ = Eigen::Vector4f::Zero(4);

            std::vector<typename v4r::ObjectHypothesis<PT>::Ptr > verified_hypotheses = recognizer.recognize(cloud);
            std::vector<v4r::ObjectHypothesesGroup<PT> > generated_object_hypotheses = recognizer.getGeneratedObjectHypothesis();

            if ( !out_dir.empty() )  // write results to disk (for each verified hypothesis add a row in the text file with object name, dummy confidence value and object pose in row-major order)
            {
                std::string out_basename = views[v_id];
                boost::replace_last(out_basename, ".pcd", ".anno");
                bf::path out_path = out_dir;
                out_path /= sub_folder_name;
                out_path /= out_basename;

                v4r::io::createDirForFileIfNotExist(out_path.string());

                // save verified hypotheses
                std::ofstream f ( out_path.string().c_str() );
                for ( const v4r::ObjectHypothesis<PT>::Ptr &voh : verified_hypotheses )
                {
                    f << voh->model_id_ << " (" << voh->confidence_ << "): ";
                    for (size_t row=0; row <4; row++)
                        for(size_t col=0; col<4; col++)
                            f << voh->transform_(row, col) << " ";
                    f << std::endl;
                }
                f.close();

                // save generated hypotheses
                std::string out_path_generated_hypotheses = out_path.string();
                boost::replace_last(out_path_generated_hypotheses, ".anno", ".generated_hyps");
                f.open ( out_path_generated_hypotheses.c_str() );
                for ( const v4r::ObjectHypothesesGroup<PT> &gohg : generated_object_hypotheses )
                {
                    for ( const v4r::ObjectHypothesis<PT>::Ptr &goh : gohg.ohs_ )
                    {
                        f << goh->model_id_ << " (-1.): ";
                        for (size_t row=0; row <4; row++)
                            for(size_t col=0; col<4; col++)
                                f << goh->transform_(row, col) << " ";
                        f << std::endl;

                    }
                }
                f.close();

                // save elapsed time(s)
                std::string out_path_times = out_path.string();
                boost::replace_last(out_path_times, ".anno", ".times");
                f.open( out_path_times.c_str() );
                for( const auto &t : elapsed_time)
                    f << t << " ";
                f.close();
            }
        }
    }
}

