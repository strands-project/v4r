/******************************************************************************
 * Copyright (c) 2017 Thomas Faeulhammer
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 ******************************************************************************/

#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <boost/serialization/vector.hpp>
#include <glog/logging.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#include <v4r/apps/ObjectRecognizer.h>
#include <v4r/io/filesystem.h>

namespace po = boost::program_options;

int
main (int argc, char ** argv)
{
    typedef pcl::PointXYZRGB PT;

    bf::path test_dir;
    bf::path out_dir = "/tmp/object_recognition_results/";
    bf::path recognizer_config_dir = "cfg";
    int verbosity = -1;

    po::options_description desc("Object Instance Recognizer\n======================================\n**Allowed options");
    desc.add_options()
            ("help,h", "produce help message")
            ("test_dir,t", po::value<bf::path>(&test_dir)->required(), "Directory with test scenes stored as point clouds (.pcd). The camera pose is taken directly from the pcd header fields \"sensor_orientation_\" and \"sensor_origin_\" (if the test directory contains subdirectories, each subdirectory is considered as seperate sequence for multiview recognition)")
            ("out_dir,o", po::value<bf::path>(&out_dir)->default_value(out_dir), "Output directory where recognition results will be stored.")
            ("cfg", po::value<bf::path>(&recognizer_config_dir)->default_value(recognizer_config_dir), "Path to config directory containing the xml config files for the various recognition pipelines and parameters.")
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

    v4r::apps::ObjectRecognizer<PT> recognizer;
    recognizer.initialize(to_pass_further, recognizer_config_dir);

    std::vector< std::string> sub_folder_names = v4r::io::getFoldersInDirectory( test_dir );
    if(sub_folder_names.empty()) sub_folder_names.push_back("");

    for (const std::string &sub_folder_name : sub_folder_names)
    {
        recognizer.resetMultiView();
        std::vector< std::string > views = v4r::io::getFilesInDirectory( test_dir / sub_folder_name, ".*.pcd", false );
        for (size_t v_id=0; v_id<views.size(); v_id++)
        {
            bf::path test_path = test_dir / sub_folder_name / views[v_id];

            LOG(INFO) << "Recognizing file " << test_path.string();
            pcl::PointCloud<PT>::Ptr cloud(new pcl::PointCloud<PT>());
            pcl::io::loadPCDFile( test_path.string(), *cloud);

            //reset view point - otherwise this messes up PCL's visualization (this does not affect recognition results)
//            cloud->sensor_orientation_ = Eigen::Quaternionf::Identity();
//            cloud->sensor_origin_ = Eigen::Vector4f::Zero(4);

            std::vector<v4r::ObjectHypothesesGroup > generated_object_hypotheses = recognizer.recognize(cloud);
            std::vector<std::pair<std::string, float> > elapsed_time = recognizer.getElapsedTimes();

            if ( !out_dir.empty() )  // write results to disk (for each verified hypothesis add a row in the text file with object name, dummy confidence value and object pose in row-major order)
            {
                std::string out_basename = views[v_id];
                boost::replace_last(out_basename, ".pcd", ".anno");
                bf::path out_path = out_dir / sub_folder_name / out_basename;

                std::string out_path_generated_hypotheses = out_path.string();
                boost::replace_last(out_path_generated_hypotheses, ".anno", ".generated_hyps");

                std::string out_path_generated_hypotheses_serialized = out_path.string();
                boost::replace_last(out_path_generated_hypotheses_serialized, ".anno", ".generated_hyps_serialized");

                v4r::io::createDirForFileIfNotExist( out_path );

                // save hypotheses
                std::ofstream f_generated ( out_path_generated_hypotheses.c_str() );
                std::ofstream f_verified ( out_path.string().c_str() );
                std::ofstream f_generated_serialized ( out_path_generated_hypotheses_serialized.c_str() );
                boost::archive::text_oarchive oa(f_generated_serialized);
                oa << generated_object_hypotheses;
                f_generated_serialized.close();
                for(size_t ohg_id=0; ohg_id<generated_object_hypotheses.size(); ohg_id++)
                {
                    for(const v4r::ObjectHypothesis::Ptr &oh : generated_object_hypotheses[ohg_id].ohs_)
                    {
                        f_generated << oh->model_id_ << " (" << oh->confidence_ << "): ";
                        const Eigen::Matrix4f tf = oh->pose_refinement_ * oh->transform_;

                        for (size_t row=0; row <4; row++)
                            for(size_t col=0; col<4; col++)
                                f_generated << tf(row, col) << " ";
                        f_generated << std::endl;

                        if( oh->is_verified_ )
                        {
                            f_verified << oh->model_id_ << " (" << oh->confidence_ << "): ";
                            for (size_t row=0; row <4; row++)
                                for(size_t col=0; col<4; col++)
                                    f_verified << tf(row, col) << " ";
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

