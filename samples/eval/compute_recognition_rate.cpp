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
std::vector<std::string> coordinate_system_ids_;

typedef pcl::PointXYZRGB PointT;
typedef v4r::Model<PointT> ModelT;
typedef boost::shared_ptr<ModelT> ModelTPtr;

//#define DEBUG_VIS
#ifdef DEBUG_VIS
pcl::visualization::PCLVisualizer::Ptr vis_debug_;
std::string debug_model_;
int vp1_, vp2_;
#endif
boost::shared_ptr < v4r::ModelOnlySource<pcl::PointXYZRGBNormal, PointT> > source;

// =======  DECLARATIONS ===================
bool computeError(Eigen::Matrix4f &rec_pose, Eigen::Matrix4f &gt_pose,
                  float &trans_error, float &rot_error);

void checkMatchvector(const std::vector< std::pair<std::string, std::string> > &rec2gt,
                      std::map<std::string,float> &occlusion,
                      std::map<std::string, Eigen::Matrix4f> &rec_pose,
                      std::map<std::string, Eigen::Matrix4f> &gt_pose,
                      float &sum_translation_error,
                      int &tp, int &fp, int &fn);

void selectBestMatch (std::vector<std::string> &rec_files,
                      std::vector<std::string> &gt_files,
                      std::map<std::string,float> &occlusion,
                      std::map<std::string, Eigen::Matrix4f> &rec_pose,
                      std::map<std::string, Eigen::Matrix4f> &gt_pose,
                      int &tp, int &fp, int &fn,
                      float &sum_translation_error);

// ==========================================


bool computeError(Eigen::Matrix4f &rec_pose, Eigen::Matrix4f &gt_pose,
                  float &trans_error, float &rot_error)
{
    const Eigen::Vector3f rec_t = rec_pose.block<3,1>(0,3);
    const Eigen::Vector3f gt_t = gt_pose.block<3,1>(0,3);
//    const Eigen::Matrix3f rec_r = rec_pose.block<3,3>(0,0);
//    const Eigen::Matrix3f gt_r = gt_pose.block<3,3>(0,0);

    trans_error = (rec_t-gt_t).norm();

    rot_error = 0.f;  //not implemented yet

    if(trans_error > translation_error_threshold_m)
        return true;

    return false;
}

void checkMatchvector(const std::vector< std::pair<std::string, std::string> > &rec2gt,
                      std::map<std::string,float> &occlusion,
                      std::map<std::string, Eigen::Matrix4f> &rec_pose,
                      std::map<std::string, Eigen::Matrix4f> &gt_pose,
                      float &sum_translation_error,
                      int &tp, int &fp, int &fn)
{
    sum_translation_error = 0.f;
    tp = fp = fn = 0;
    for(size_t i=0; i<rec2gt.size(); i++)
    {
        const std::string &rec_file = rec2gt[i].first;
        const std::string &gt_file = rec2gt[i].second;

        if(gt_file.empty()) {
            fp++;
            continue;
        }

        if(rec_file.empty()) {  // only count if the gt object is not occluded

            if(occlusion[gt_file] < occlusion_threshold)
                fn++;

            continue;
        }

        float trans_error, rot_error;
        if( computeError( rec_pose[rec_file], gt_pose[gt_file], trans_error, rot_error))
        {
            fp++;

            if(occlusion[gt_file] < occlusion_threshold)
                fn++;
        }
        else
        {
            tp++;
            sum_translation_error+=trans_error;
        }
    }
}

void selectBestMatch (std::vector<std::string> &rec_files,
                      std::vector<std::string> &gt_files,
                      std::map<std::string,float> &occlusion,
                      std::map<std::string, Eigen::Matrix4f> &rec_pose,
                      std::map<std::string, Eigen::Matrix4f> &gt_pose,
                      int &tp, int &fp, int &fn,
                      float &sum_translation_error)
{
#ifdef DEBUG_VIS
    if(vis_debug_)
    {
        vis_debug_->removeAllPointClouds();
        vis_debug_->removeAllShapes();
        ModelTPtr model;
        source->getModelById( debug_model_, model );
        typename pcl::PointCloud<PointT>::ConstPtr model_cloud = model->getAssembled(0.003f);

        for(size_t r_id=0; r_id<rec_files.size(); r_id++)
        {
            typename pcl::PointCloud<PointT>::Ptr model_aligned(new pcl::PointCloud<PointT>());
            pcl::transformPointCloud(*model_cloud, *model_aligned, rec_pose[rec_files[r_id]]);
            vis_debug_->addPointCloud(model_aligned, rec_files[r_id], vp2_);
        }

        for(size_t gt_id=0; gt_id<gt_files.size(); gt_id++)
        {
            typename pcl::PointCloud<PointT>::Ptr model_aligned(new pcl::PointCloud<PointT>());
            pcl::transformPointCloud(*model_cloud, *model_aligned, gt_pose[gt_files[gt_id]]);
            if( occlusion[gt_files[gt_id]] < occlusion_threshold)
                vis_debug_->addPointCloud(model_aligned, "gt_" + gt_files[gt_id], vp1_);
            else
            {
                pcl::visualization::PointCloudColorHandlerCustom<PointT> handler_occ (model_aligned, 128, 128, 128);
                vis_debug_->addPointCloud(model_aligned, handler_occ, "gt_" + gt_files[gt_id], vp1_);
            }
        }


        // go through all possible permutations and return best match
        size_t elements_to_check = std::max(rec_files.size(), gt_files.size());

        float best_fscore = -1;
        sum_translation_error = std::numeric_limits<float>::max();
        tp=0, fp=0, fn=0;

        std::sort(rec_files.begin(), rec_files.end());
        do {
            std::sort(gt_files.begin(), gt_files.end());
            do{
                std::vector< std::pair<std::string, std::string> > rec2gt_matches (elements_to_check);
                for(size_t i=0; i<elements_to_check; i++)
                {
                    std::string rec_file = "", gt_file = "";
                    if(rec_files.size()>i)
                        rec_file = rec_files[i];
                    if(gt_files.size()>i)
                        gt_file = gt_files[i];

                    rec2gt_matches[i] = std::pair<std::string,std::string>(rec_file, gt_file);
                }
                float sum_translation_error_tmp;
                int tp_tmp, fp_tmp, fn_tmp;
                checkMatchvector(rec2gt_matches, occlusion, rec_pose, gt_pose, sum_translation_error_tmp, tp_tmp, fp_tmp, fn_tmp);

                float recall = 1.f;
                if (tp_tmp+fn_tmp) // if there are some ground-truth objects
                    recall = (float)tp_tmp / (tp_tmp + fn_tmp);

                float precision = 1.f;
                if(tp_tmp+fp_tmp)   // if there are some recognized objects
                    precision = (float)tp_tmp / (tp_tmp + fp_tmp);

                float fscore = 0.f;
                if ( precision+recall>std::numeric_limits<float>::epsilon() )
                    fscore = 2 * precision * recall / (precision + recall);

                if ( (fscore > best_fscore) || (fscore==best_fscore && sum_translation_error_tmp/tp_tmp < sum_translation_error/tp)) {
                    best_fscore = fscore;
                    sum_translation_error = sum_translation_error_tmp;
                    tp = tp_tmp;
                    fp = fp_tmp;
                    fn = fn_tmp;
                }
            } while (next_permutation(gt_files.begin(), gt_files.end()));

        } while (next_permutation(rec_files.begin(), rec_files.end()));

        vis_debug_->addText("gt objects (occluded objects in gray)", 10, 10, 20, 1.f, 1.f, 1.f, "gt_text", vp1_);
        std::stringstream rec_text;
        rec_text << "recognized objects (tp: " << best_tp << ", fp: " << best_fp << ", fn: " << best_fn << ")"; //", fscore: " << fscore << ")";
        vis_debug_->addText(rec_text.str(), 10, 10, 20, 1.f, 1.f, 1.f, "rec_text", vp2_);
        vis_debug_->spin();
    }
#endif

    // go through all possible permutations and return best match
    size_t elements_to_check = std::max(rec_files.size(), gt_files.size());

    float best_fscore = -1;
    sum_translation_error = std::numeric_limits<float>::max();
    tp=0, fp=0, fn=0;

    std::sort(rec_files.begin(), rec_files.end());
    do {
        std::sort(gt_files.begin(), gt_files.end());
        do{
            std::vector< std::pair<std::string, std::string> > rec2gt_matches (elements_to_check);
            for(size_t i=0; i<elements_to_check; i++)
            {
                std::string rec_file = "", gt_file = "";
                if(rec_files.size()>i)
                    rec_file = rec_files[i];
                if(gt_files.size()>i)
                    gt_file = gt_files[i];

                rec2gt_matches[i] = std::pair<std::string,std::string>(rec_file, gt_file);
            }
            float sum_translation_error_tmp;
            int tp_tmp, fp_tmp, fn_tmp;
            checkMatchvector(rec2gt_matches, occlusion, rec_pose, gt_pose, sum_translation_error_tmp, tp_tmp, fp_tmp, fn_tmp);

            float recall = 1.f;
            if (tp_tmp+fn_tmp) // if there are some ground-truth objects
                recall = (float)tp_tmp / (tp_tmp + fn_tmp);

            float precision = 1.f;
            if(tp_tmp+fp_tmp)   // if there are some recognized objects
                precision = (float)tp_tmp / (tp_tmp + fp_tmp);

            float fscore = 0.f;
            if ( precision+recall>std::numeric_limits<float>::epsilon() )
                fscore = 2 * precision * recall / (precision + recall);

            if ( (fscore > best_fscore) || (fscore==best_fscore && sum_translation_error_tmp/tp_tmp < sum_translation_error/tp)) {
                best_fscore = fscore;
                sum_translation_error = sum_translation_error_tmp;
                tp = tp_tmp;
                fp = fp_tmp;
                fn = fn_tmp;
            }
        } while (next_permutation(gt_files.begin(), gt_files.end()));

    } while (next_permutation(rec_files.begin(), rec_files.end()));
}

int
main (int argc, char ** argv)
{
    std::string out_dir = "/tmp/recognition_rates/";
    std::string test_dir, gt_dir, models_dir, or_dir;

    rotation_error_threshold_deg = 30.f;
    translation_error_threshold_m = 0.05f;
    occlusion_threshold = 0.95f;
    bool visualize=false, do_validation=false;

    std::stringstream description;
    description << "Tool to compute object instance recognition rate." << std::endl <<
                   "==================================================" << std::endl <<
                   "This will generate a text file containing:" << std::endl <<
                   "Column 1: test set id" << std::endl <<
                   "Column 2: view id" << std::endl <<
                   "Column 3: true positives" << std::endl <<
                   "Column 4: false positives" << std::endl <<
                   "Column 5: false negatives" << std::endl <<
                   "Column 6: accumulated translation error of all true positive objects" << std::endl <<
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
        ("visualize,v", po::bool_switch(&visualize), "visualize recognition results")
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

    pcl::visualization::PCLVisualizer::Ptr vis;
    int vp1, vp2, vp3;
    if(visualize)
    {
        vis.reset (new pcl::visualization::PCLVisualizer ("results"));
        vis->createViewPort(0, 0, 0.33, 1, vp1);
        vis->createViewPort(0.33, 0, 0.66, 1, vp2);
        vis->createViewPort(0.66, 0, 1, 1, vp3);
        source.reset(new v4r::ModelOnlySource<pcl::PointXYZRGBNormal, PointT>());
        source->setPath (models_dir);
        source->setLoadViews (false);
        source->setLoadIntoMemory(false);
        source->generate ();

#ifdef DEBUG_VIS
        vis_debug_.reset (new pcl::visualization::PCLVisualizer ("select best matches"));
        vis_debug_->createViewPort(0, 0, 0.5, 1, vp1_);
        vis_debug_->createViewPort(0.5, 0, 1, 1, vp2_);
#endif
    }


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
        const std::string out_fn = out_dir + "/results_" + validation_sets[val_id] + ".txt";
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
               int tp = 0;
               int fp = 0;
               int fn = 0;
               float sum_translation_error = 0.f;

               if(visualize)
               {
                   vis->removeAllPointClouds();
                   vis->removeAllShapes();
               }

               boost::replace_last(view_files[view_id], ".pcd", "");
               for(size_t model_id=0; model_id<model_files.size(); model_id++)
               {
    #ifdef DEBUG_VIS
                   debug_model_ = model_files[model_id];
    #endif
                   std::string search_pattern = view_files[view_id] + "_" + model_files[model_id] + "_";

                   std::string regex_search_pattern = ".*" + search_pattern + "*.*txt";
                   std::vector<std::string> rec_files = v4r::io::getFilesInDirectory (rec_dir, regex_search_pattern,  true);
                   std::vector<std::string> gt_files  = v4r::io::getFilesInDirectory (anno_dir, regex_search_pattern,  true);

                   std::map<std::string, float> occlusion;
                   std::map<std::string, Eigen::Matrix4f> gt_pose;
                   std::map<std::string, Eigen::Matrix4f> rec_pose;

                   for(size_t gt_id=0; gt_id<gt_files.size(); gt_id++) {
                       std::string occlusion_file = gt_files[gt_id];
                       boost::replace_first(occlusion_file, view_files[view_id], view_files[view_id] + "_occlusion");
                       occlusion_file = anno_dir + "/" + occlusion_file;
                       float occ = 0.f;
                       if (!v4r::io::readFloatFromFile(occlusion_file, occ))
                           cerr << "Did not find occlusion file " << occlusion_file << std::endl;

                       occlusion[gt_files[gt_id]] = occ;
                       gt_pose[gt_files[gt_id]] = v4r::io::readMatrixFromFile(anno_dir + "/" + gt_files[gt_id]);
                   }
                   for(size_t r_id=0; r_id<rec_files.size(); r_id++)
                       rec_pose[rec_files[r_id]] = v4r::io::readMatrixFromFile(rec_dir + "/" + rec_files[r_id]);

                   int tp_tmp, fp_tmp, fn_tmp;
                   float sum_translation_error_tmp;
                   selectBestMatch(rec_files, gt_files, occlusion, rec_pose, gt_pose, tp_tmp, fp_tmp, fn_tmp, sum_translation_error_tmp);

                   tp+=tp_tmp;
                   fp+=fp_tmp;
                   fn+=fn_tmp;

                   if(tp)
                       sum_translation_error += sum_translation_error_tmp;

                   if(visualize)
                   {
                       ModelTPtr model;
                       source->getModelById( model_files[model_id], model );
                       typename pcl::PointCloud<PointT>::ConstPtr model_cloud = model->getAssembled(0.003f);

                       for(size_t r_id=0; r_id<rec_files.size(); r_id++)
                       {
                           typename pcl::PointCloud<PointT>::Ptr model_aligned(new pcl::PointCloud<PointT>());
                           pcl::transformPointCloud(*model_cloud, *model_aligned, rec_pose[rec_files[r_id]]);
                           vis->addPointCloud(model_aligned, rec_files[r_id], vp3);

#if PCL_VERSION >= 100702
                           Eigen::Matrix4f tf_tmp = rec_pose[rec_files[r_id]];
                           Eigen::Matrix3f rot_tmp  = tf_tmp.block<3,3>(0,0);
                           Eigen::Vector3f trans_tmp = tf_tmp.block<3,1>(0,3);
                           Eigen::Affine3f affine_trans;
                           affine_trans.fromPositionOrientationScale(trans_tmp, rot_tmp, Eigen::Vector3f::Ones());
                           std::stringstream co_id; co_id << coordinate_system_ids_.size();
                           vis->addCoordinateSystem(0.1f, affine_trans, co_id.str(), vp3);
                           vis->setBackgroundColor(1,1,1,vp3);
#endif
                       }

                       for(size_t gt_id=0; gt_id<gt_files.size(); gt_id++)
                       {
                           typename pcl::PointCloud<PointT>::Ptr model_aligned(new pcl::PointCloud<PointT>());
                           pcl::transformPointCloud(*model_cloud, *model_aligned, gt_pose[gt_files[gt_id]]);
                           if( occlusion[gt_files[gt_id]] < occlusion_threshold)
                               vis->addPointCloud(model_aligned, "gt_" + gt_files[gt_id], vp2);
                           else
                           {
                               pcl::visualization::PointCloudColorHandlerCustom<PointT> handler_occ (model_aligned, 128, 128, 128);
                               vis->addPointCloud(model_aligned, handler_occ, "gt_" + gt_files[gt_id], vp2);
                           }

#if PCL_VERSION >= 100702
                           Eigen::Matrix4f tf_tmp = gt_pose[gt_files[gt_id]];
                           Eigen::Matrix3f rot_tmp  = tf_tmp.block<3,3>(0,0);
                           Eigen::Vector3f trans_tmp = tf_tmp.block<3,1>(0,3);
                           Eigen::Affine3f affine_trans;
                           affine_trans.fromPositionOrientationScale(trans_tmp, rot_tmp, Eigen::Vector3f::Ones());
                           std::stringstream co_id; co_id << coordinate_system_ids_.size();
                           vis->addCoordinateSystem(0.1f, affine_trans, co_id.str(), vp2);
                           vis->setBackgroundColor(1,1,1,vp2);
#endif
                       }
                   }
               }
               f << test_sets[set_id] << " " << view_files[view_id] << " " << tp << " " << fp << " " << fn << " " << sum_translation_error << std::endl;

               if(visualize)
               {
                   pcl::PointCloud<PointT> scene_cloud;
                   pcl::io::loadPCDFile( test_set_dir + "/" + view_files[view_id] + ".pcd", scene_cloud);
                   //reset view point - otherwise this messes up PCL's visualization (this does not affect recognition results)
                   scene_cloud.sensor_orientation_ = Eigen::Quaternionf::Identity();
                   scene_cloud.sensor_origin_ = Eigen::Vector4f::Zero(4);
                   vis->addPointCloud(scene_cloud.makeShared(), "scene", vp1);
                   vis->setBackgroundColor(1,1,1,vp1);
                   vis->addText(test_set_dir + "/" + view_files[view_id] + ".pcd", 10, 10, 15, 0.f, 0.f, 0.f, "scene_text", vp1);
                   vis->addText("ground-truth objects (occluded objects in gray)", 10, 10, 10, 0.f, 0.f, 0.f, "gt_text", vp2);
                   std::stringstream rec_text;
                   rec_text << "recognized objects (tp: " << tp << ", fp: " << fp << ", fn: " << fn; //", fscore: " << fscore << ")";
                   if(tp)
                       rec_text << " trans_error: " << sum_translation_error/tp;
                   rec_text << ")";
                   vis->addText(rec_text.str(), 10, 10, 15, 0.f, 0.f, 0.f, "rec_text", vp3);
                   vis->resetCamera();
                   vis->spin();
                   vis->removeCoordinateSystem(vp2);
                   vis->removeCoordinateSystem(vp3);
               }
           }
        }
        f.close();
        std::cout << "Done!" << std::endl;
    }
}
