#include <v4r/common/miscellaneous.h>  // to extract Pose intrinsically stored in pcd file
#include <v4r/io/eigen.h>
#include <v4r/io/filesystem.h>
#include <v4r/recognition/source.h>

#include <pcl/common/transforms.h>
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

using namespace v4r;

typedef pcl::PointXYZRGB PointT;

//#define DEBUG_VIS
#ifdef DEBUG_VIS
pcl::visualization::PCLVisualizer::Ptr vis_debug_;
std::string debug_model_;
int vp1_, vp2_;
#endif
Source<PointT>::Ptr source;
namespace bf = boost::filesystem;


struct Hypothesis
{
    Eigen::Matrix4f pose;
    float occlusion;
};


// =======  DECLARATIONS ===================
bool computeError(const Eigen::Matrix4f &rec_pose, const Eigen::Matrix4f &gt_pose,
                  float &trans_error, float &rot_error);

void checkMatchvector(const std::vector< std::pair<int, int> > &rec2gt,
                      const std::vector<Hypothesis> &rec_hyps,
                      const std::vector<Hypothesis> &gt_hyps,
                      double &sum_translation_error,
                      size_t &tp, size_t &fp, size_t &fn);

std::vector<std::pair<int, int> > selectBestMatch(std::vector<Hypothesis> &rec_hyps,
                                                  std::vector<Hypothesis> &gt_hyps,
                                                  size_t &tp, size_t &fp, size_t &fn,
                                                  double &sum_translation_error);

// ==========================================


bool computeError(const Eigen::Matrix4f &rec_pose, const Eigen::Matrix4f &gt_pose,
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

void checkMatchvector(const std::vector< std::pair<int, int> > &rec2gt,
                      const std::vector<Hypothesis> &rec_hyps,
                      const std::vector<Hypothesis> &gt_hyps,
                      double &sum_translation_error,
                      size_t &tp, size_t &fp, size_t &fn)
{
    sum_translation_error = 0.f;
    tp = fp = fn = 0;
    for(size_t i=0; i<rec2gt.size(); i++)
    {
        int rec_id = rec2gt[i].first;
        int gt_id = rec2gt[i].second;

        if(gt_id < 0)
        {
            fp++;
            continue;
        }

        const Hypothesis &gt_hyp = gt_hyps [ gt_id ];

        if( rec_id < 0 )
        {
            if( gt_hyp.occlusion < occlusion_threshold) // only count if the gt object is not occluded
                fn++;

            continue;
        }

        const Hypothesis &rec_hyp = rec_hyps [ rec_id ] ;

        float trans_error, rot_error;
        if( computeError( rec_hyp.pose, gt_hyp.pose, trans_error, rot_error))
        {
            fp++;

            if( gt_hyp.occlusion < occlusion_threshold)
                fn++;
        }
        else
        {
            tp++;
            sum_translation_error+=trans_error;
        }
    }
}

std::vector< std::pair<int, int> >
selectBestMatch (std::vector<Hypothesis> &rec_hyps,
                 std::vector<Hypothesis> &gt_hyps,
                 size_t &tp, size_t &fp, size_t &fn,
                 double &sum_translation_error)
{
#ifdef DEBUG_VIS
    if(vis_debug_)
    {
        vis_debug_->removeAllPointClouds();
        vis_debug_->removeAllShapes();

        bool found;
        Model<PointT>::ConstPtr model = source->getModelById("", debug_model_, found );
        if (!found)
            std::cerr << "Did not find " << model_files[model_id] << ". There is something wrong! " << std::endl;

        typename pcl::PointCloud<PointT>::ConstPtr model_cloud = model->getAssembled(3);

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
    size_t elements_to_check = std::max(rec_hyps.size(), gt_hyps.size());

    float best_fscore = -1;
    sum_translation_error = std::numeric_limits<float>::max();
    tp=0, fp=0, fn=0;
    std::vector< std::pair<int, int> > best_match;

    std::vector<int> rec_ids(rec_hyps.size());
    std::iota (std::begin(rec_ids), std::end(rec_ids), 0);
    std::vector<int> gt_ids(gt_hyps.size());
    std::iota (std::begin(gt_ids), std::end(gt_ids), 0);

    do {
        do{
            std::vector< std::pair<int, int> > rec2gt_matches (elements_to_check);
            for(size_t i=0; i<elements_to_check; i++)
            {
                int rec_id = -1, gt_id = -1;
                if( rec_hyps.size()>i )
                    rec_id = rec_ids[i];
                if( gt_hyps.size()>i )
                    gt_id = gt_ids[i];

                rec2gt_matches[i] = std::pair<int, int>(rec_id, gt_id);
            }
            double sum_translation_error_tmp;
            size_t tp_tmp, fp_tmp, fn_tmp;
            checkMatchvector(rec2gt_matches, rec_hyps, gt_hyps, sum_translation_error_tmp, tp_tmp, fp_tmp, fn_tmp);

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
                best_match = rec2gt_matches;
            }
        } while ( next_permutation( gt_ids.begin(), gt_ids.end()) );
    } while ( next_permutation( rec_ids.begin(), rec_ids.end()) );
    return best_match;
}


std::map<std::string, std::vector<Hypothesis> >
readHypothesesFromFile( const std::string &filename )
{
    std::map<std::string, std::vector<Hypothesis> > hypotheses;

    std::ifstream anno_f ( filename.c_str() );
    std::string line;
    while (std::getline(anno_f, line))
    {
        std::istringstream iss(line);
        std::string model_name, occlusion_tmp;

        Hypothesis h;
        iss >> model_name >> occlusion_tmp;
        occlusion_tmp = occlusion_tmp.substr( 1, occlusion_tmp.length() - 3 );
        h.occlusion = 1.f-std::stof( occlusion_tmp );

        for(size_t i=0; i<16; i++)
            iss >> h.pose(i / 4, i % 4);

        auto pose_it = hypotheses.find( model_name );
        if( pose_it != hypotheses.end() )
            pose_it->second.push_back( h ) ;
        else
            hypotheses[model_name] = std::vector<Hypothesis>(1, h);
    }

    return hypotheses;
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
            ("groundtruth_dir,g", po::value<std::string>(&gt_dir)->required(), "Root directory containing annotation files (i.e. 4x4 ground-truth pose of each object with filename viewId_ModelId_ModelInstanceCounter.txt")
            ("rec_results_dir,r", po::value<std::string>(&or_dir)->required(), "Root directory containing the recognition results (same format as annotation files).")
            ("out_dir,o", po::value<std::string>(&out_dir)->default_value(out_dir), "Output directory where recognition results will be stored")
            ("trans_thresh", po::value<float>(&translation_error_threshold_m)->default_value(translation_error_threshold_m), "Maximal allowed translational error in metres")
            ("rot_thresh", po::value<float>(&rotation_error_threshold_deg)->default_value(rotation_error_threshold_deg), "Maximal allowed rotational error in degrees (NOT IMPLEMENTED)")
            ("visualize,v", po::bool_switch(&visualize), "visualize recognition results")
            ("models_dir,m", po::value<std::string>(&models_dir), "Only for visualization. Root directory containing the model files (i.e. filenames 3D_model.pcd).")
            ("test_dir,t", po::value<std::string>(&test_dir), "Only for visualization. Root directory containing the scene files.")
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

    std::vector<std::string> annotation_files = v4r::io::getFilesInDirectory( gt_dir, ".*.anno", true );

    double sum_translation_error = 0.f;
    size_t tp_total = 0;
    size_t fp_total = 0;
    size_t fn_total = 0;

    pcl::visualization::PCLVisualizer::Ptr vis;

    int vp1, vp2, vp3;
    if(visualize)
    {
        vis.reset (new pcl::visualization::PCLVisualizer ("results"));
        vis->createViewPort(0, 0, 1, 0.33, vp1);
        vis->createViewPort(0, 0.33, 1, 0.66, vp2);
        vis->createViewPort(0, 0.66, 1, 1, vp3);
        source.reset( new Source<PointT> (models_dir) );
    }

    for( const std::string anno_file : annotation_files )
    {
        bf::path gt_path = gt_dir;
        gt_path /= anno_file;

        bf::path rec_path = or_dir;
        rec_path /= anno_file;

        std::map<std::string, std::vector<Hypothesis> > gt_hyps = readHypothesesFromFile( gt_path.string() );
        std::map<std::string, std::vector<Hypothesis> > rec_hyps = readHypothesesFromFile( rec_path.string() );


        std::set<std::string> model_names;  // all model names either observed from recognition or labelled
        for( const auto &tmp : gt_hyps )
            model_names.insert( tmp.first );
        for( const auto &tmp : rec_hyps)
            model_names.insert( tmp.first );


        size_t tp_view = 0;
        size_t fp_view = 0;
        size_t fn_view = 0;
        double sum_translation_error = 0.;

        if(vis)
        {
            vis->removeAllPointClouds();
            vis->removeAllShapes();
#if PCL_VERSION >= 100702
            vis->removeAllCoordinateSystems();
#endif
        }

        for( const auto &model_name : model_names )
        {
            std::vector<Hypothesis> rec_hyps_tmp, gt_hyps_tmp;

            auto it = rec_hyps.find( model_name );
            if ( it != rec_hyps.end() )
                rec_hyps_tmp = it->second;

            it = gt_hyps.find( model_name );
            if ( it != gt_hyps.end() )
                gt_hyps_tmp = it->second;


            size_t tp_tmp, fp_tmp, fn_tmp;
            double sum_translation_error_tmp;
            std::vector< std::pair<int, int> > matches = selectBestMatch(rec_hyps_tmp, gt_hyps_tmp, tp_tmp, fp_tmp, fn_tmp, sum_translation_error_tmp);

            tp_view+=tp_tmp;
            fp_view+=fp_tmp;
            fn_view+=fn_tmp;
            sum_translation_error += sum_translation_error_tmp;

            if(visualize)
            {
                size_t counter = 0;
                for ( const auto &match : matches )
                {
                    int rec_id = match.first;
                    int gt_id = match.second;

                    if ( rec_id >= 0 )
                    {
                        const Hypothesis &hyp_vis = rec_hyps_tmp[ rec_id ];

                        bool found;
                        Model<PointT>::ConstPtr model = source->getModelById("", model_name, found );
                        if (!found)
                            std::cerr << "Did not find " << model_name << ". There is something wrong! " << std::endl;

                        typename pcl::PointCloud<PointT>::ConstPtr model_cloud = model->getAssembled(3);
                        typename pcl::PointCloud<PointT>::Ptr model_aligned(new pcl::PointCloud<PointT>());
                        pcl::transformPointCloud(*model_cloud, *model_aligned, hyp_vis.pose);
                        std::stringstream unique_id; unique_id << model_name << "_" << counter;
                        vis->addPointCloud(model_aligned, unique_id.str(), vp3);

#if PCL_VERSION >= 100702
                        Eigen::Matrix4f tf_tmp = hyp_vis.pose;
                        Eigen::Matrix3f rot_tmp  = tf_tmp.block<3,3>(0,0);
                        Eigen::Vector3f trans_tmp = tf_tmp.block<3,1>(0,3);
                        Eigen::Affine3f affine_trans;
                        affine_trans.fromPositionOrientationScale(trans_tmp, rot_tmp, Eigen::Vector3f::Ones());
                        std::stringstream co_id; co_id << model_name << "_co_" << counter;
                        vis->addCoordinateSystem(0.1f, affine_trans, co_id.str(), vp3);
#endif
                        counter++;
                    }

                    if ( gt_id >= 0 )
                    {
                        const Hypothesis &hyp_vis = gt_hyps_tmp[ gt_id ];

                        bool found;
                        Model<PointT>::ConstPtr model = source->getModelById("", model_name, found );
                        if (!found)
                            std::cerr << "Did not find " << model_name << ". There is something wrong! " << std::endl;

                        typename pcl::PointCloud<PointT>::ConstPtr model_cloud = model->getAssembled(3);
                        typename pcl::PointCloud<PointT>::Ptr model_aligned(new pcl::PointCloud<PointT>());
                        pcl::transformPointCloud(*model_cloud, *model_aligned, hyp_vis.pose);
                        std::stringstream unique_id; unique_id << model_name << "_" << counter;

                        if(hyp_vis.occlusion > occlusion_threshold)
                        {
                            pcl::visualization::PointCloudColorHandlerCustom<PointT> green (model_aligned, 0, 0, 255);
                            vis->addPointCloud(model_aligned, green, unique_id.str(), vp2);
                        }
                        else
                            vis->addPointCloud(model_aligned, unique_id.str(), vp2);

#if PCL_VERSION >= 100702
                        Eigen::Matrix4f tf_tmp = hyp_vis.pose;
                        Eigen::Matrix3f rot_tmp  = tf_tmp.block<3,3>(0,0);
                        Eigen::Vector3f trans_tmp = tf_tmp.block<3,1>(0,3);
                        Eigen::Affine3f affine_trans;
                        affine_trans.fromPositionOrientationScale(trans_tmp, rot_tmp, Eigen::Vector3f::Ones());
                        std::stringstream co_id; co_id << model_name << "_co_" << counter;
                        vis->addCoordinateSystem(0.1f, affine_trans, co_id.str(), vp2);
#endif
                        counter++;
                    }
                }
            }
        }

        std::cout << anno_file << ": " << tp_view << " " << fp_view << " " << fn_view << std::endl;

        tp_total += tp_view;
        fp_total += fp_view;
        fn_total += fn_view;

        if(visualize)
        {
            std::string scene_name (anno_file);
            boost::replace_last( scene_name, ".anno", ".pcd");
            bf::path scene_path = test_dir;
            scene_path /= scene_name;
            pcl::PointCloud<PointT>::Ptr scene_cloud (new pcl::PointCloud<PointT>);
            pcl::io::loadPCDFile( scene_path.string(), *scene_cloud);
            //reset view point - otherwise this messes up PCL's visualization (this does not affect recognition results)
            scene_cloud->sensor_orientation_ = Eigen::Quaternionf::Identity();
            scene_cloud->sensor_origin_ = Eigen::Vector4f::Zero(4);
            vis->addPointCloud(scene_cloud, "scene", vp1);

            pcl::visualization::PointCloudColorHandlerCustom<PointT> gray (scene_cloud, 255, 255, 255);
            vis->addPointCloud(scene_cloud, gray, "input_vp2", vp2);
            vis->setPointCloudRenderingProperties( pcl::visualization::PCL_VISUALIZER_OPACITY, 0.2, "input_vp2");
            vis->addPointCloud(scene_cloud, gray, "input_vp3", vp3);
            vis->setPointCloudRenderingProperties( pcl::visualization::PCL_VISUALIZER_OPACITY, 0.2, "input_vp3");

            vis->addText( scene_name, 10, 10, 15, 1.f, 1.f, 1.f, "scene_text", vp1);
            vis->addText("ground-truth objects (occluded objects in blue)", 10, 10, 15, 1.f, 1.f, 1.f, "gt_text", vp2);
            std::stringstream rec_text;
            rec_text << "recognized objects (tp: " << tp_view << ", fp: " << fp_view << ", fn: " << fn_view;
            if(tp_view)
                rec_text << " trans_error: " << sum_translation_error/tp_view;
            rec_text << ")";
            vis->addText(rec_text.str(), 10, 10, 15, 1.f, 1.f, 1.f, "rec_text", vp3);
            vis->resetCamera();
            vis->spin();
        }
    }



#ifdef DEBUG_VIS
    vis_debug_.reset (new pcl::visualization::PCLVisualizer ("select best matches"));
    vis_debug_->createViewPort(0, 0, 0.5, 1, vp1_);
    vis_debug_->createViewPort(0.5, 0, 1, 1, vp2_);
#endif

#ifdef BLA
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
                for(size_t model_id=0; ;)//model_id<model_files.size(); model_id++)
                {
#ifdef DEBUG_VIS
                    debug_model_ = model_files[model_id];
#endif
                    std::string search_pattern;// = view_files[view_id] + "_" + model_files[model_id] + "_";

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

                }
                f << test_sets[set_id] << " " << view_files[view_id] << " " << tp << " " << fp << " " << fn << " " << sum_translation_error << std::endl;
            }
        }
        f.close();
        std::cout << "Done!" << std::endl;
    }
#endif
}
