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
    const std::string info_file = "/home/thomas/Documents/icra16/keyframes/uncontrolled_run.nfo";
    const std::string runs = "/media/Data/datasets/icra16/object_masks/uncontrolled";
    const std::string runs_old = "/media/Data/datasets/icra16/keyframes/uncontrolled";
    const std::string bag_files = "/media/Data/datasets/icra16/bag_files";

    std::ifstream info(info_file.c_str());
    std::string test_id, patrol_run_id, object_id;
    while (info >> test_id >> patrol_run_id >> object_id) {
            const std::string mask_file = runs_old + "/" + test_id + "/mask.txt";
            const std::string dst_file = runs + "/" + test_id + "/mask.txt";
            v4r::io::createDirForFileIfNotExist(dst_file);
            if(v4r::io::existsFile(mask_file) && !v4r::io::existsFile(dst_file))
                bf::copy_file(mask_file, dst_file);
    }

    return 0;
}
