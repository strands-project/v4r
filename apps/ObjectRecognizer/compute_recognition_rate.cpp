#include <v4r/common/miscellaneous.h>  // to extract Pose intrinsically stored in pcd file
#include <v4r/io/filesystem.h>

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

namespace bf = boost::filesystem;


struct Hypothesis
{
    Eigen::Matrix4f pose;
    float occlusion;
};


// =======  DECLARATIONS ===================
/**
 * @brief computeError compute translation error for a given pose
 * @param[in] pose_a
 * @param[in] pose_b
 * @param[out] trans_error translation error
 * @param[out] rot_error rotation error
 * @return boolean indicitating if error is outside given threshold
 */
bool computeError(const Eigen::Matrix4f &pose_a, const Eigen::Matrix4f &pose_b,
                  float &trans_error, float &rot_error);


/**
 * @brief checkMatchvector check a certain match configuration
 * @param rec2gt matches indicated by first index corresponding
 * to model id and second index to ground-truth id
 * @param rec_hyps recognition hypotheses
 * @param gt_hyps ground-truth hypotheses
 * @param sum_translation_error
 * @param tp true positives
 * @param fp flase positives
 * @param fn false negatives
 */
void checkMatchvector(const std::vector< std::pair<int, int> > &rec2gt,
                      const std::vector<Hypothesis> &rec_hyps,
                      const std::vector<Hypothesis> &gt_hyps,
                      double &sum_translation_error,
                      size_t &tp, size_t &fp, size_t &fn);

/**
 * @brief selectBestMatch computes the best matches for a set of hypotheses. This
 * tackles the problem when multiple object instances are present in the scene and
 * the question is which ground-truth object belongs to which recognized object. By
 * permuating through all possible solutions, it finds the match which gives the
 * best f-score taking into account to be neglected hypotheses due to occlusion.
 * @param rec_hyps recognized hypotheses
 * @param gt_hyps ground-truth hypotheses
 * @param tp true positives for best match
 * @param fp false positives for best match
 * @param fn false negatives for best match
 * @param sum_translation_error accumulated translation error for best match
 * @return best match for the given hypotheses. First index corresponds to element in
 * given recognition hypothesis, second index to ground-truth hypothesis
 */
std::vector<std::pair<int, int> > selectBestMatch(const std::vector<Hypothesis> &rec_hyps,
                                                  const std::vector<Hypothesis> &gt_hyps,
                                                  size_t &tp, size_t &fp, size_t &fn,
                                                  double &sum_translation_error);


/**
 * @brief readHypothesesFromFile reads annotations from a text file
 * @param filename filename
 * @return stores hypotheses into Hypothesis class for each object model
 */
std::map<std::string, std::vector<Hypothesis> >
readHypothesesFromFile( const std::string &filename );
// ==========================================


bool computeError(const Eigen::Matrix4f &pose_a, const Eigen::Matrix4f &pose_b,
                  float &trans_error, float &rot_error)
{
    const Eigen::Vector3f trans_a = pose_a.block<3,1>(0,3);
    const Eigen::Vector3f trans_b = pose_b.block<3,1>(0,3);
    //    const Eigen::Matrix3f rec_r = rec_pose.block<3,3>(0,0);
    //    const Eigen::Matrix3f gt_r = gt_pose.block<3,3>(0,0);

    trans_error = (trans_a-trans_b).norm();

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
selectBestMatch (const std::vector<Hypothesis> &rec_hyps,
                 const std::vector<Hypothesis> &gt_hyps,
                 size_t &tp, size_t &fp, size_t &fn,
                 double &sum_translation_error)
{
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

//    rotation_error_threshold_deg = 30.f;
    translation_error_threshold_m = 0.05f;
    occlusion_threshold = 0.95f;
    bool visualize=false;
    bool use_generated_hypotheses = false;

    std::stringstream description;
    description << "Tool to compute object instance recognition rate." << std::endl <<
                   "==================================================" << std::endl <<
                   "This will generate a text file containing:" << std::endl <<
                   "Column 1: annotation file" << std::endl <<
                   "Column 2: true positives" << std::endl <<
                   "Column 3: false positives" << std::endl <<
                   "Column 4: false negatives" << std::endl <<
                   "Column 5: accumulated translation error of all true positive objects" << std::endl <<
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
            ("use_generated_hypotheses", po::bool_switch(&use_generated_hypotheses), "if true, computes recognition rate for all generated hypotheses instead of verified ones.")
            ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) { std::cout << desc << std::endl; return false;}
    try  {  po::notify(vm); }
    catch( std::exception& e)  { std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl; return false; }

    bf::path out_path = out_dir;
    out_path /= "recognition_results.txt";
    v4r::io::createDirForFileIfNotExist(out_path.string());
    std::ofstream of ( out_path.string().c_str() );


    pcl::visualization::PCLVisualizer::Ptr vis;
    int vp1, vp2, vp3;

    // load models (only used for visualization)
    std::vector<std::string> model_filenames = io::getFilesInDirectory( models_dir, "3D_model.pcd", true );
    std::map<std::string, pcl::PointCloud<PointT>::Ptr> models;

    if(visualize)
    {
        for(const std::string &model_fn : model_filenames)
        {
            pcl::PointCloud<PointT>::Ptr model_cloud (new pcl::PointCloud<PointT>);
            bf::path model_full_path = models_dir;
            model_full_path /= model_fn;
            pcl::io::loadPCDFile( model_full_path.string(), *model_cloud );

            // model identity is equal folder name -> remove \"/3D_model.pcd\" from filename
            bf::path model_path = model_fn;
            models[ model_path.parent_path().string() ] = model_cloud;
        }

        vis.reset (new pcl::visualization::PCLVisualizer ("results"));
        vis->createViewPort(0, 0, 1, 0.33, vp1);
        vis->createViewPort(0, 0.33, 1, 0.66, vp2);
        vis->createViewPort(0, 0.66, 1, 1, vp3);
    }


    std::vector<std::string> annotation_files = v4r::io::getFilesInDirectory( gt_dir, ".*.anno", true );


    for( const std::string anno_file : annotation_files )
    {
        bf::path gt_path = gt_dir;
        gt_path /= anno_file;

        std::string rec_file = anno_file;
        if(use_generated_hypotheses)
            boost::replace_last( rec_file, ".anno", ".generated_hyps");

        bf::path rec_path = or_dir;
        rec_path /= rec_file;

        std::map<std::string, std::vector<Hypothesis> > gt_hyps = readHypothesesFromFile( gt_path.string() );
        std::map<std::string, std::vector<Hypothesis> > rec_hyps = readHypothesesFromFile( rec_path.string() );

        size_t tp_view = 0;
        size_t fp_view = 0;
        size_t fn_view = 0;
        double sum_translation_error_view = 0.;

        if(vis)
        {
            vis->removeAllPointClouds();
            vis->removeAllShapes();
#if PCL_VERSION >= 100800
            vis->removeAllCoordinateSystems();
#endif
        }

        for( const auto &m : models )
        {
            std::vector<Hypothesis> rec_hyps_tmp, gt_hyps_tmp;

            auto it = rec_hyps.find( m.first );
            if ( it != rec_hyps.end() )
                rec_hyps_tmp = it->second;

            it = gt_hyps.find( m.first );
            if ( it != gt_hyps.end() )
                gt_hyps_tmp = it->second;

            size_t tp_tmp, fp_tmp, fn_tmp;
            double sum_translation_error_tmp;
            std::vector< std::pair<int, int> > matches = selectBestMatch(rec_hyps_tmp, gt_hyps_tmp, tp_tmp, fp_tmp, fn_tmp, sum_translation_error_tmp);

            tp_view+=tp_tmp;
            fp_view+=fp_tmp;
            fn_view+=fn_tmp;
            sum_translation_error_view += sum_translation_error_tmp;

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
                        typename pcl::PointCloud<PointT>::ConstPtr model_cloud = m.second;
                        typename pcl::PointCloud<PointT>::Ptr model_aligned(new pcl::PointCloud<PointT>());
                        pcl::transformPointCloud(*model_cloud, *model_aligned, hyp_vis.pose);
                        std::stringstream unique_id; unique_id << m.first << "_" << counter;
                        vis->addPointCloud(model_aligned, unique_id.str(), vp3);

#if PCL_VERSION >= 100800
                        Eigen::Matrix4f tf_tmp = hyp_vis.pose;
                        Eigen::Matrix3f rot_tmp  = tf_tmp.block<3,3>(0,0);
                        Eigen::Vector3f trans_tmp = tf_tmp.block<3,1>(0,3);
                        Eigen::Affine3f affine_trans;
                        affine_trans.fromPositionOrientationScale(trans_tmp, rot_tmp, Eigen::Vector3f::Ones());
                        std::stringstream co_id; co_id << m.first << "_co_" << counter;
                        vis->addCoordinateSystem(0.1f, affine_trans, co_id.str(), vp3);
#endif
                        counter++;
                    }

                    if ( gt_id >= 0 )
                    {
                        const Hypothesis &hyp_vis = gt_hyps_tmp[ gt_id ];
                        typename pcl::PointCloud<PointT>::ConstPtr model_cloud = m.second;
                        typename pcl::PointCloud<PointT>::Ptr model_aligned(new pcl::PointCloud<PointT>());
                        pcl::transformPointCloud(*model_cloud, *model_aligned, hyp_vis.pose);
                        std::stringstream unique_id; unique_id << m.first << "_" << counter;

                        if(hyp_vis.occlusion > occlusion_threshold)
                        {
                            pcl::visualization::PointCloudColorHandlerCustom<PointT> green (model_aligned, 0, 0, 255);
                            vis->addPointCloud(model_aligned, green, unique_id.str(), vp2);
                        }
                        else
                            vis->addPointCloud(model_aligned, unique_id.str(), vp2);

#if PCL_VERSION >= 100800
                        Eigen::Matrix4f tf_tmp = hyp_vis.pose;
                        Eigen::Matrix3f rot_tmp  = tf_tmp.block<3,3>(0,0);
                        Eigen::Vector3f trans_tmp = tf_tmp.block<3,1>(0,3);
                        Eigen::Affine3f affine_trans;
                        affine_trans.fromPositionOrientationScale(trans_tmp, rot_tmp, Eigen::Vector3f::Ones());
                        std::stringstream co_id; co_id << m.first << "_co_" << counter;
                        vis->addCoordinateSystem(0.1f, affine_trans, co_id.str(), vp2);
#endif
                        counter++;
                    }
                }
            }
        }

        std::cout << anno_file << ": " << tp_view << " " << fp_view << " " << fn_view << " " << sum_translation_error_view << std::endl;
        of << anno_file << " " << tp_view << " " << fp_view << " " << fn_view << " " << sum_translation_error_view << std::endl;

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
                rec_text << " trans_error: " << sum_translation_error_view/tp_view;
            rec_text << ")";
            vis->addText(rec_text.str(), 10, 10, 15, 1.f, 1.f, 1.f, "rec_text", vp3);
            vis->resetCamera();
            vis->spin();
        }
    }
    of.close();
}
