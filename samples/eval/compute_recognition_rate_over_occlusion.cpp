#include <v4r/common/miscellaneous.h>  // to extract Pose intrinsically stored in pcd file
#include <v4r/io/eigen.h>
#include <v4r/io/filesystem.h>
#include <v4r/recognition/model_only_source.h>

#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>    // std::next_permutation, std::sort

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

// -m /media/Data/datasets/TUW/models/ -t /media/Data/datasets/TUW/validation_set/ -g /media/Data/datasets/TUW/annotations/ -r /home/thomas/recognition_results_eval/

float rotation_error_threshold_deg;
float translation_error_threshold_m;
float occlusion_threshold;

typedef pcl::PointXYZRGB PointT;
typedef v4r::Model<PointT> ModelT;
typedef boost::shared_ptr<ModelT> ModelTPtr;

int
main (int argc, char ** argv)
{
    std::string out_dir = "/tmp/recognition_rates_over_occlusion/";
    std::string test_dir, gt_dir, models_dir, or_dir;

    rotation_error_threshold_deg = 30.f;
    translation_error_threshold_m = 0.05f;
    bool do_validation=false;

    std::stringstream description;
    description << "Tool to compute object instance recognition rate." << std::endl <<
                   "==================================================" << std::endl <<
                   "This will generate a text file containing:" << std::endl <<
                   "Column 1: occlusion" << std::endl <<
                   "Column 2: is recognized" << std::endl <<
                   "==================================================" << std::endl <<
                   "** Allowed options";

    po::options_description desc(description.str());
    desc.add_options()
        ("help,h", "produce help message")
        ("test_dir,t", po::value<std::string>(&test_dir)->required(), "Root directory with test scenes stored as point clouds (.pcd).")
        ("models_dir,m", po::value<std::string>(&models_dir)->required(), "Root directory containing the model files (i.e. filenames 3D_model.pcd).")
        ("groundtruth_dir,g", po::value<std::string>(&gt_dir)->required(), "Root directory containing annotation files (i.e. 4x4 ground-truth pose of each object with filename viewId_ModelId_ModelInstanceCounter.txt")
        ("rec_results_dir,r", po::value<std::string>(&or_dir)->required(), "Root directory containing the recognition results (same format as annotation files).")
        ("out_dir,o", po::value<std::string>(&out_dir)->default_value(out_dir), "Output directory where recognition results will be stored")
        ("trans_thresh", po::value<float>(&translation_error_threshold_m)->default_value(translation_error_threshold_m), "Maximal allowed translational error in metres")
        ("rot_thresh", po::value<float>(&rotation_error_threshold_deg)->default_value(rotation_error_threshold_deg), "Maximal allowed rotational error in degrees (NOT IMPLEMENTED)")
        ("do_validation", po::bool_switch(&do_validation), "if set, it interprets the given rec_results_dir as the directory containing different test runs (e.g. different parameters). For each folder, it will generate a result file. Inside each folder there is the same structure as used without this parameter.")
    ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        return false;
    }
    try  {  po::notify(vm); }
    catch( std::exception& e)  { std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl; return false; }
    v4r::io::createDirIfNotExist(out_dir);

    std::vector < std::string > model_files = v4r::io::getFilesInDirectory (models_dir, ".*3D_model.pcd",  true);
    // we are only interested in the name of the model, so remove remaing filename
    for(size_t model_id=0; model_id<model_files.size(); model_id++)
        boost::replace_last(model_files[model_id], "/3D_model.pcd", "");


    std::vector<std::string> validation_sets;
    if(do_validation)
        validation_sets = v4r::io::getFoldersInDirectory ( or_dir );
    else
        validation_sets.push_back("");

    for(size_t val_id=0; val_id<validation_sets.size(); val_id++)
    {
        const std::string out_fn = out_dir + "/results_occlusion_" + validation_sets[val_id] + ".txt";
        if(v4r::io::existsFile(out_fn))
        {
            std::cout << out_fn << " exists already. Skipping it!" << std::endl;
            continue;
        }

        std::ofstream f(out_fn);
        std::cout << "Writing results to " << out_fn << "..." << std::endl;

        std::vector < std::string > test_sets = v4r::io::getFoldersInDirectory ( test_dir );
        if(test_sets.empty())
            test_sets.push_back("");

        for(size_t set_id=0; set_id<test_sets.size(); set_id++)
        {
           const std::string test_set_dir = test_dir + "/" + test_sets[set_id];
           const std::string rec_dir = or_dir + "/" + validation_sets[val_id] + "/" + test_sets[set_id];
           const std::string anno_dir = gt_dir + "/" + test_sets[set_id];

           std::vector < std::string > view_files = v4r::io::getFilesInDirectory (test_set_dir, ".*.pcd",  true);
           for(size_t view_id=0; view_id<view_files.size(); view_id++)
           {
               boost::replace_last(view_files[view_id], ".pcd", "");
               for(size_t model_id=0; model_id<model_files.size(); model_id++)
               {
                   std::string search_pattern = view_files[view_id] + "_" + model_files[model_id] + "_";
                   std::string regex_search_pattern = ".*" + search_pattern + "*.*txt";
                   std::vector<std::string> rec_files = v4r::io::getFilesInDirectory (rec_dir, regex_search_pattern,  true);
                   std::vector<std::string> gt_files  = v4r::io::getFilesInDirectory (anno_dir, regex_search_pattern,  true);

                   for(size_t gt_id=0; gt_id<gt_files.size(); gt_id++) {
                       std::string occlusion_file = gt_files[gt_id];
                       boost::replace_first(occlusion_file, view_files[view_id], view_files[view_id] + "_occlusion");
                       occlusion_file = anno_dir + "/" + occlusion_file;
                       float occ = 0.f;
                       if (!v4r::io::readFloatFromFile(occlusion_file, occ))
                           cerr << "Did not find occlusion file " << occlusion_file << std::endl;

                       Eigen::Matrix4f gt_pose_tmp =v4r::io::readMatrixFromFile(anno_dir + "/" + gt_files[gt_id]);

                       bool is_recognized = false;
                       for(size_t r_id=0; r_id<rec_files.size(); r_id++)
                       {
                           Eigen::Matrix4f rec_tmp = v4r::io::readMatrixFromFile(rec_dir + "/" + rec_files[r_id]);
                           Eigen::Vector3f rec_trans = rec_tmp.block<3,1>(0,3);
                           Eigen::Vector3f gt_trans = gt_pose_tmp.block<3,1>(0,3);

                           if( (gt_trans-rec_trans).norm() < translation_error_threshold_m)
                               is_recognized = true;
                       }
                       f << occ << " " << is_recognized << std::endl;
                   }
               }
           }
        }
        f.close();
        std::cout << "Done!" << std::endl;
    }
}
