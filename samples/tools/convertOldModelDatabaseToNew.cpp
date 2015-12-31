#include <iostream>
#include <fstream>

#define BOOST_NO_CXX11_SCOPED_ENUMS
#include <boost/filesystem.hpp>
#undef BOOST_NO_CXX11_SCOPED_ENUMS
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <pcl/io/pcd_io.h>

#include <v4r/common/faat_3d_rec_framework_defines.h>
#include <v4r/common/miscellaneous.h>
#include <v4r/io/eigen.h>
#include <v4r/io/filesystem.h>

namespace bf = boost::filesystem ;
namespace po = boost::program_options;

int
main (int argc, char ** argv)
{
    std::string model_dir, training_dir, output_dir = "/tmp/new_dataset/";
    std::string view_prefix = "cloud_";
    std::string pose_prefix = "pose_";
    std::string indices_prefix = "object_indices_";

    po::options_description desc("\nConverter of old to new object model dataset structure\n **Allowed options");
    desc.add_options()
            ("help,h", "produce help message")
            ("input_training_dir,t", po::value<std::string>(&training_dir)->required(), "root directory containing directories with training views, pose and object mask of each object model (recognition structure)")
            ("input_models_dir,m", po::value<std::string>(&model_dir)->required(), "directory containing the model .pcd files")
            ("output_dir,o", po::value<std::string>(&output_dir)->default_value(output_dir), "output directory")
            ("view_prefix", po::value<std::string>(&view_prefix)->default_value(view_prefix), "training view basename prefix")
            ("pose_prefix", po::value<std::string>(&pose_prefix)->default_value(pose_prefix), "camera pose basename prefix")
            ("indices_prefix", po::value<std::string>(&indices_prefix)->default_value(indices_prefix), "object indices basename prefix")
            ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        return false;
    }

    try
    {
        po::notify(vm);
    }
    catch(std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl;
        return false;
    }

    std::vector< std::string> object_models = v4r::io::getFilesInDirectory(model_dir, ".*.pcd", false);
    if( object_models.empty() )
    {
        std::cerr << "No .pcd files found in given models directory " << model_dir << "! " << std::endl;
        return -1;
    }

    for (const std::string &model_path : object_models)
    {
        const std::string model = bf::basename(model_path);
        std::cout << "converting model " << model << "..." << std::endl;

        std::string out_model_dir = output_dir + "/" + model;
        v4r::io::createDirIfNotExist(out_model_dir);
        v4r::io::createDirIfNotExist(out_model_dir + "/views");

        bf::copy_file(model_path, out_model_dir + "/3D_model.pcd");

        const std::string training_view_regex_pattern = ".*" + view_prefix + ".*.pcd";
        std::vector<std::string> views = v4r::io::getFilesInDirectory( training_dir + "/" + model + ".pcd", training_view_regex_pattern, false);

        if(views.empty())
                std::cerr << "Cannot find any training views in " << training_dir << "/" << model << ".pcd! Is the training dir set correctly?" << std::endl;
        for(const std::string &view : views)
        {
            bf::copy_file(view, out_model_dir + "/views/" + bf::basename(view) + ".pcd");
            std::string pose_fn (view);
            boost::replace_last(pose_fn, view_prefix, pose_prefix);
            boost::replace_last(pose_fn, ".pcd", ".txt");

            if(v4r::io::existsFile(pose_fn))
                bf::copy_file( pose_fn, out_model_dir + "/views/" + bf::basename(pose_fn) + ".txt");
            else
                std::cerr << "Pose file " << pose_fn << " does not exist! ";

            std::string indices_fn (view);
            boost::replace_last(indices_fn, view_prefix, indices_prefix);


            if(v4r::io::existsFile(indices_fn))
            {
                pcl::PointCloud<IndexPoint> obj_indices_cloud;
                pcl::io::loadPCDFile (indices_fn, obj_indices_cloud);

                const std::string object_mask_fn = out_model_dir + "/views/" + bf::basename(indices_fn) + ".txt";
                std::ofstream f ( object_mask_fn.c_str() );
                for(size_t kk=0; kk < obj_indices_cloud.points.size(); kk++)
                    f << obj_indices_cloud.points[kk].idx << std::endl;
                f.close();
            }
            else
                std::cerr << "Object indices file " << indices_fn << " does not exist! ";

        }
    }
}
