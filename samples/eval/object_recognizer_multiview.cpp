#include <v4r/common/miscellaneous.h>   // to extract Pose intrinsically stored in pcd file
#include <v4r/io/filesystem.h>
#include <v4r/io/eigen.h>
#include <v4r/recognition/multiview_object_recognizer.h>
#include <v4r/recognition/multiview_object_recognizer_change_detection.h>
#include <v4r/change_detection/viewport_checker.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/passthrough.h>
#include <iostream>
#include <boost/program_options.hpp>
#include <boost/format.hpp>
#include <glog/logging.h>

#define USE_CHANGE_DETECTION

namespace po = boost::program_options;

typedef pcl::PointXYZRGB PointT;
typedef v4r::Model<PointT> ModelT;
typedef boost::shared_ptr<ModelT> ModelTPtr;

bool isInsideOfView(ModelTPtr model, Eigen::Matrix4f &model_pose);

int
main (int argc, char ** argv)
{
    std::string test_dir, out_dir = "/tmp/mv_object_recognizer_results/";
    bool visualize = false;
    std::map<std::string, size_t> rec_models_per_id_;  // just to have a unique filename when writing pose of objects to disk (in case of multiple instances detected in one scene)

    google::InitGoogleLogging(argv[0]);

    po::options_description desc("Multiview Object Instance Recognizer\n======================================**Reference(s): Faeulhammer et al, ICRA / MVA 2015\n **Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("test_dir,t", po::value<std::string>(&test_dir)->required(), "Directory with test scenes stored as point clouds (.pcd). The camera pose is taken directly from the pcd header fields \"sensor_orientation_\" and \"sensor_origin_\" (if the test directory contains subdirectories, each subdirectory is considered as seperate sequence for multiview recognition)")
        ("visualize,v", po::bool_switch(&visualize), "visualize recognition results")
        ("out_dir,o", po::value<std::string>(&out_dir)->default_value(out_dir), "Output directory where recognition results will be stored.")
   ;
    po::variables_map vm;
    po::parsed_options parsed = po::command_line_parser(argc, argv).options(desc).allow_unregistered().run();
    po::store(parsed, vm);
    if (vm.count("help")) { std::cout << desc << std::endl; }
    try { po::notify(vm); }
    catch(std::exception& e) { std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl;  }

    v4r::io::createDirIfNotExist(out_dir);

     //writing parameters to file
    ofstream param_file;
    param_file.open ((out_dir + "/param.nfo").c_str());
    for(const auto& it : vm)
    {
      param_file << "--" << it.first << " ";

      auto& value = it.second.value();
      if (auto v_double = boost::any_cast<double>(&value))
        param_file << std::setprecision(3) << *v_double;
      else if (auto v_string = boost::any_cast<std::string>(&value))
        param_file << *v_string;
      else if (auto v_bool = boost::any_cast<bool>(&value))
        param_file << *v_bool;
      else if (auto v_int = boost::any_cast<int>(&value))
        param_file << *v_int;
      else if (auto v_size_t = boost::any_cast<size_t>(&value))
        param_file << *v_size_t;
      else
        param_file << "error";

      param_file << " ";
    }
    param_file.close();

#ifndef USE_CHANGE_DETECTION
    v4r::MultiviewRecognizer<PointT> mv_r(argc, argv);
    bool ignore_detections_outside_of_view = false;
#else
    v4r::MultiviewRecognizerWithChangeDetection<PointT> mv_r(argc, argv);
    bool ignore_detections_outside_of_view = true;
#endif

    // -----------TEST--------------
    std::vector< std::string> sub_folder_names = v4r::io::getFoldersInDirectory( test_dir);
    if( sub_folder_names.empty() )
        sub_folder_names.push_back("");

    for (size_t sub_folder_id=0; sub_folder_id < sub_folder_names.size(); sub_folder_id++)
    {
        const std::string sequence_path = test_dir + "/" + sub_folder_names[ sub_folder_id ];
        mv_r.set_scene_name(sequence_path);
        const std::string out_path = out_dir + "/" + sub_folder_names[ sub_folder_id ];
        v4r::io::createDirIfNotExist(out_path);

        rec_models_per_id_.clear();

        const std::string out_results_3d_fn = out_path + "/results_3d.txt";
        std::vector< std::string > views = v4r::io::getFilesInDirectory(sequence_path, ".*.pcd", false);
        for (size_t v_id=0; v_id<views.size(); v_id++)
        {
            const std::string fn = test_dir + "/" + sub_folder_names[sub_folder_id] + "/" + views[ v_id ];

            LOG(INFO) << "Recognizing file " << fn;
            typename pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
            pcl::io::loadPCDFile(fn, *cloud);

            Eigen::Matrix4f tf = v4r::RotTrans2Mat4f(cloud->sensor_orientation_, cloud->sensor_origin_);

            // reset view point otherwise pcl visualization is potentially messed up
            cloud->sensor_orientation_ = Eigen::Quaternionf::Identity();
            cloud->sensor_origin_ = Eigen::Vector4f::Zero();

            mv_r.setInputCloud (cloud);
            mv_r.setCameraPose(tf);
            pcl::StopWatch watch;
            mv_r.recognize();
            v4r::io::writeFloatToFile(out_path+"/"+views[v_id].substr(0, views[v_id].length()-4) +"_time.nfo", watch.getTimeSeconds());

            std::vector<ModelTPtr> verified_models = mv_r.getVerifiedModels();
            std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transforms_verified = mv_r.getVerifiedTransforms();

            if (visualize)
                mv_r.visualize();

            for(size_t m_id=0; m_id<verified_models.size(); m_id++)
                std::cout << "******" << verified_models[m_id]->id_ << std::endl <<  transforms_verified[m_id] << std::endl;

            std::ofstream results_3d;
            results_3d.open (out_results_3d_fn.c_str());

            for(size_t m_id=0; m_id<verified_models.size(); m_id++)
            {
                 LOG(INFO) << "******" << verified_models[m_id]->id_ << std::endl <<  transforms_verified[m_id] << std::endl;

                 if(ignore_detections_outside_of_view) {
                     if(isInsideOfView(verified_models[m_id], transforms_verified[m_id])) {
                         std::cout << "INSIDE of current view volume" << std::endl;
                     } else {
                         std::cout << "OUTSIDE of current view volume - IGNORED!" << std::endl;
                         continue;
                     }
                 }

                const std::string model_id = verified_models[m_id]->id_;
                const Eigen::Matrix4f tf_tmp = transforms_verified[m_id];
                const Eigen::Matrix4f tf2_world = tf * tf_tmp;

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
                       << model_id.substr(0, model_id.length() - 4) << "_" << num_models_per_model_id << ".txt";

                ofstream or_file (out_fn.str().c_str());
                results_3d << mv_r.getModelsDir() + "/" + model_id << " ";
                for (size_t row=0; row <4; row++)
                {
                    for(size_t col=0; col<4; col++)
                    {
                        or_file << tf_tmp(row, col) << " ";
                        results_3d << tf2_world(row, col) << " ";
                    }
                }
                or_file.close();
                results_3d << std::endl;
            }
            results_3d.close();
        }
        mv_r.cleanUp(); // delete all stored information from last sequences
    }
}

bool isInsideOfView(ModelTPtr model, Eigen::Matrix4f &model_pose) {
    static v4r::ViewVolume<PointT> view_volume = v4r::ViewVolume<PointT>::ofXtion(Eigen::Affine3f::Identity(), 0.0);

    pcl::PointCloud<PointT>::ConstPtr model_cloud = model->getAssembled(0.01);
    pcl::PointCloud<PointT>::Ptr model_transformed(new pcl::PointCloud<PointT>);
    pcl::transformPointCloud(*model_cloud, *model_transformed, model_pose);
    std::vector<bool> visible_mask(model_transformed->size(), true);

    if(view_volume.computeVisible(model_transformed, visible_mask) > 0) {
         return true;
    } else {
         return false;
    }
}
