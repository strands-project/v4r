#define BOOST_NO_SCOPED_ENUMS
#define BOOST_NO_CXX11_SCOPED_ENUMS

#include <v4r_config.h>
#include <v4r/io/filesystem.h>

#include <boost/any.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdlib.h>


namespace po = boost::program_options;
namespace bf = boost::filesystem;


int
main (int argc, char ** argv)
{
    const std::string info_file = "/home/thomas/Documents/icra16_keyframes/controlled_run.nfo";
    const std::string input_dir = "/media/Data/datasets/icra16/learnt_models";
    const std::string out_dir = "/home/thomas/learnt_models/";

    v4r::io::createDirIfNotExist(out_dir + "/models");
    v4r::io::createDirIfNotExist(out_dir + "/training_dir");

    std::ifstream info(info_file.c_str());
    std::string test_id, patrol_run_id, object_id;
    while (info >> test_id >> patrol_run_id >> object_id) {
        const std::string src_training_dir = input_dir + "/recognition_structure/" + patrol_run_id + "_object.pcd";
        if (v4r::io::existsFolder(src_training_dir)) {
            v4r::io::copyDir( src_training_dir, out_dir + "/training_dir/" + test_id + ".pcd");
            bf::copy_file( input_dir + "/models/"  + patrol_run_id + "_object.pcd",
                           out_dir + "/models/" + test_id + ".pcd");
        }
    }

    return 0;
}
