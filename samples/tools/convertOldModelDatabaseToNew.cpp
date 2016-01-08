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
    std::string entropy_prefix = "entropy_";

    po::options_description desc("\nConverter of old to new object model dataset structure\n **Allowed options");
    desc.add_options()
            ("help,h", "produce help message")
            ("input_training_dir,t", po::value<std::string>(&training_dir)->required(), "root directory containing directories with training views, pose and object mask of each object model (recognition structure)")
            ("input_models_dir,m", po::value<std::string>(&model_dir), "directory containing the model .pcd files")
            ("output_dir,o", po::value<std::string>(&output_dir)->default_value(output_dir), "output directory")
            ("view_prefix", po::value<std::string>(&view_prefix)->default_value(view_prefix), "training view basename prefix (e.g. cloud_, view_)")
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

    std::vector< std::string> object_models = v4r::io::getFoldersInDirectory(training_dir);
    if( object_models.empty() )
    {
        std::cerr << "No .pcd files found in given models directory " << model_dir << "! " << std::endl;
        return -1;
    }

    for (const std::string &model_training_path : object_models) {
        const std::string full_model_training_path = training_dir + "/" + model_training_path + "/";
        const std::string model = bf::basename(model_training_path);
        std::cout << "converting model " << model << "..." << std::endl;

        std::string out_model_dir = output_dir + "/" + model;
        v4r::io::createDirIfNotExist(out_model_dir);
        v4r::io::createDirIfNotExist(out_model_dir + "/views");

        const std::string model_fn = model_dir + "/" + model + ".pcd";

        if(v4r::io::existsFile(model_fn))
            bf::copy_file(model_fn, out_model_dir + "/3D_model.pcd");


        const std::string training_view_regex_pattern = ".*" + view_prefix + ".*.pcd";
        std::vector<std::string> views = v4r::io::getFilesInDirectory( full_model_training_path, training_view_regex_pattern, false);

        if(views.empty()) { // does it have classes and object identities?
            std::vector<std::string> class_models = v4r::io::getFoldersInDirectory(full_model_training_path);
            if(class_models.empty()) {
                std::cerr << "Cannot find any training views in " << full_model_training_path << "! Is the training dir and view prefix set correctly?" << std::endl;
                continue;
            }
            else {
                std::cerr << "Converting object database with class labels is not implemented yet!" << std::endl;
                return -1;
            }
        }

        std::vector<std::string> files_copied; // save copied files to avoid copying redundant files in trained descriptors (in old model database views were saved multiple times)

        for(const std::string &view : views) {
            bf::copy_file( full_model_training_path + view, out_model_dir + "/views/" + view);
            files_copied.push_back(view);


            // COPY POSE FILE
            std::string pose_fn (view);
            boost::replace_last(pose_fn, view_prefix, pose_prefix);
            boost::replace_last(pose_fn, ".pcd", ".txt");

            if(v4r::io::existsFile( full_model_training_path + pose_fn )) {
                bf::copy_file( full_model_training_path + pose_fn, out_model_dir + "/views/" + pose_fn);
                files_copied.push_back(pose_fn);
            }
            else
                std::cout << "Pose file " << pose_fn << " does not exist! " << std::endl;


            // COPY OBJECT MASK FILE
            std::string indices_fn (view);
            boost::replace_last(indices_fn, view_prefix, indices_prefix);

            if(v4r::io::existsFile(full_model_training_path + indices_fn)) {
                pcl::PointCloud<IndexPoint> obj_indices_cloud;
                pcl::io::loadPCDFile (full_model_training_path + indices_fn, obj_indices_cloud);

                const std::string object_mask_fn = out_model_dir + "/views/" + bf::basename(indices_fn) + ".txt";
                std::ofstream f ( object_mask_fn.c_str() );
                for(size_t kk=0; kk < obj_indices_cloud.points.size(); kk++)
                    f << obj_indices_cloud.points[kk].idx << std::endl;
                f.close();
                files_copied.push_back(indices_fn);
            }
            else
                std::cout << "Object indices file " << indices_fn << " does not exist! " << std::endl;


            // COPY ENTROPY FILE
            std::string entropy_fn (view);
            boost::replace_last(entropy_fn, view_prefix, entropy_prefix);
            boost::replace_last(entropy_fn, ".pcd", ".txt");

            if(v4r::io::existsFile( full_model_training_path + entropy_fn )) {
                bf::copy_file( full_model_training_path + entropy_fn, out_model_dir + "/views/" + entropy_fn);
                files_copied.push_back(entropy_fn);
            }
            else
                std::cout << "Entropy file " << entropy_fn << " does not exist! " << std::endl;
        }


        // Now copy all trained feature description files if there are any (e.g. sift, shot_omp, esf,..)
        std::vector<std::string> descriptors_folders = v4r::io::getFoldersInDirectory(full_model_training_path);
        for (const std::string &desc_folder : descriptors_folders) {
            v4r::io::createDirIfNotExist(out_model_dir + "/" + desc_folder);
            std::vector<std::string> desc_files = v4r::io::getFilesInDirectory( full_model_training_path + desc_folder, ".*", false);

            for(const std::string desc_file : desc_files) {
                bool file_already_copied = false;
                for (const std::string &check : files_copied) {
                    if ( desc_file.compare(check) == 0 ) {
                        file_already_copied = true;
                        break;
                    }
                }

                if( !file_already_copied ) {
                    bf::copy_file( full_model_training_path + desc_folder + "/" + desc_file, out_model_dir + "/" + desc_folder + "/" + desc_file);
                }
            }
        }
    }
}
