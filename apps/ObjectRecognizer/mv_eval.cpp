
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <glog/logging.h>

#include <v4r/apps/ObjectRecognizer.h>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <v4r/io/filesystem.h>

#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

namespace po = boost::program_options;

int
main (int argc, char ** argv)
{
    typedef pcl::PointXYZRGB PT;

    bf::path test_dir;
    bf::path out_dir = "/tmp/object_recognition_results/";
    bf::path recognizer_config_dir = "cfg";
    int verbosity = -1;
    bool shuffle_views = true;
    size_t view_sample_size = 1;

    /* initialize random seed: */
    srand (time(NULL));

    po::options_description desc("Single-View Object Instance Recognizer\n======================================\n**Allowed options");
    desc.add_options()
            ("help,h", "produce help message")
            ("test_dir,t", po::value<bf::path>(&test_dir)->required(), "Directory with test scenes stored as point clouds (.pcd). The camera pose is taken directly from the pcd header fields \"sensor_orientation_\" and \"sensor_origin_\" (if the test directory contains subdirectories, each subdirectory is considered as seperate sequence for multiview recognition)")
            ("out_dir,o", po::value<bf::path>(&out_dir)->default_value(out_dir), "Output directory where recognition results will be stored.")
            ("cfg", po::value<bf::path>(&recognizer_config_dir)->default_value(recognizer_config_dir), "Path to config directory containing the xml config files for the various recognition pipelines and parameters.")
            ("verbosity", po::value<int>(&verbosity)->default_value(verbosity), "set verbosity level for output (<0 minimal output)")
            ("shuffle_views", po::value<bool>(&shuffle_views)->default_value(shuffle_views), "if true, randomly selects viewpoints. Otherwise in the sequence given by the filenames.")
            ("view_sample_size", po::value<size_t>(&view_sample_size)->default_value(view_sample_size), "view sample size. Only every n-th view will be recognized to speed up evaluation.")
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

    v4r::apps::ObjectRecognizer<PT> recognizer;
    recognizer.initialize(to_pass_further, recognizer_config_dir);

    std::vector< std::string> sub_folder_names = v4r::io::getFoldersInDirectory( test_dir );
    if(sub_folder_names.empty()) sub_folder_names.push_back("");

    for (const std::string &sub_folder_name : sub_folder_names)
    {
        recognizer.resetMultiView();
        std::vector< std::string > views = v4r::io::getFilesInDirectory( test_dir / sub_folder_name, ".*.pcd", false );
        size_t kept=0;
        for(size_t i=0; i<views.size(); i = i+view_sample_size)
            views[kept++] = views[i];

        views.resize(kept);


        const v4r::apps::ObjectRecognizerParameter &param = recognizer.getParam();
        if( views.size() < param.max_views_)
        {
            LOG(WARNING) << "There are not enough views (" << views.size() << ") within this sequence to evaluate on " << param.max_views_ << " views! Skipping sequence.";
            continue;
        }

        // randomly walk through views
        std::vector<int> ivec (views.size() );
        std::iota(ivec.begin(), ivec.end(), 0);

        if(shuffle_views)
            std::random_shuffle(ivec.begin(), ivec.end());

        ivec.insert(ivec.end(), ivec.begin(), ivec.end());

        std::cout << "Evaluation order for " << sub_folder_name << ":" << std::endl;
        std::for_each(ivec.begin(), ivec.end(), [](int elem){std::cout << elem << " ";});
        std::cout << std::endl;

        boost::dynamic_bitset<> view_is_evaluated ( views.size(), 0 );

        size_t counter=0;
        while(1)
        {
            int v_id = ivec[counter++];

            if( view_is_evaluated[v_id] ) //everything evaluated
                break;

            bf::path test_path = test_dir / sub_folder_name / views[v_id];

            LOG(INFO) << "Recognizing file " << test_path.string();
            pcl::PointCloud<PT>::Ptr cloud(new pcl::PointCloud<PT>());
            pcl::io::loadPCDFile( test_path.string(), *cloud);

            //reset view point - otherwise this messes up PCL's visualization (this does not affect recognition results)
//            cloud->sensor_orientation_ = Eigen::Quaternionf::Identity();
//            cloud->sensor_origin_ = Eigen::Vector4f::Zero(4);

            std::vector<v4r::ObjectHypothesesGroup > generated_object_hypotheses = recognizer.recognize(cloud);
            std::vector<std::pair<std::string, float> > elapsed_time = recognizer.getElapsedTimes();

            if ( counter >= param.max_views_  && !out_dir.empty() )  // write results to disk (for each verified hypothesis add a row in the text file with object name, dummy confidence value and object pose in row-major order)
            {
                view_is_evaluated.set(v_id);
                std::string out_basename = views[v_id];
                boost::replace_last(out_basename, ".pcd", ".anno");
                bf::path out_path = out_dir / sub_folder_name / out_basename;

                std::string out_path_generated_hypotheses = out_path.string();
                boost::replace_last(out_path_generated_hypotheses, ".anno", ".generated_hyps");

                v4r::io::createDirForFileIfNotExist(out_path.string());

                // save hypotheses
                std::ofstream f_generated ( out_path_generated_hypotheses.c_str() );
                std::ofstream f_verified ( out_path.string().c_str() );
                for(size_t ohg_id=0; ohg_id<generated_object_hypotheses.size(); ohg_id++)
                {
                    for(const v4r::ObjectHypothesis::Ptr &oh : generated_object_hypotheses[ohg_id].ohs_)
                    {
                        f_generated << oh->model_id_ << " (" << oh->confidence_ << "): ";
                        for (size_t row=0; row <4; row++)
                            for(size_t col=0; col<4; col++)
                                f_generated << oh->transform_(row, col) << " ";
                        f_generated << std::endl;

                        if( oh->is_verified_ )
                        {
                            f_verified << oh->model_id_ << " (" << oh->confidence_ << "): ";
                            for (size_t row=0; row <4; row++)
                                for(size_t col=0; col<4; col++)
                                    f_verified << oh->transform_(row, col) << " ";
                            f_verified << std::endl;
                        }
                    }
                }
                f_generated.close();
                f_verified.close();

                // save elapsed time(s)
                std::string out_path_times = out_path.string();
                boost::replace_last(out_path_times, ".anno", ".times");
                f_verified.open( out_path_times.c_str() );
                for( const std::pair<std::string,float> &t : elapsed_time)
                    f_verified << t.second << " " << t.first << std::endl;
                f_verified.close();
            }
        }
    }
}

