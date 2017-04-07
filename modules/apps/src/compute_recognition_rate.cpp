#include <v4r/apps/compute_recognition_rate.h>
#include <v4r/io/filesystem.h>

#include <pcl/common/centroid.h>
#include <pcl/common/angles.h>
#include <pcl/common/time.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <algorithm>    // std::next_permutation, std::sort
#include <glog/logging.h>

namespace po = boost::program_options;


namespace v4r
{
namespace apps
{


/**
 * @brief readHypothesesFromFile reads annotations from a text file
 * @param filename filename
 * @return stores hypotheses into Hypothesis class for each object model
 */
std::map<std::string, std::vector<Hypothesis> >
readHypothesesFromFile( const std::string &filename );
// ==========================================


bool
RecognitionEvaluator::computeError(const Eigen::Matrix4f &pose_a, const Eigen::Matrix4f &pose_b, const Eigen::Vector4f& centroid_model,
                  float &trans_error, float &rot_error, bool is_rotation_invariant, bool is_rotational_symmetric)
{
    const Eigen::Vector4f centroid_a = pose_a * centroid_model;
    const Eigen::Vector4f centroid_b = pose_b * centroid_model;

    const Eigen::Matrix3f rot_a = pose_a.block<3,3>(0,0);
    const Eigen::Matrix3f rot_b = pose_b.block<3,3>(0,0);

    const Eigen::Vector3f rotX_a = rot_a * Eigen::Vector3f::UnitX();
    const Eigen::Vector3f rotX_b = rot_b * Eigen::Vector3f::UnitX();
//    const Eigen::Vector3f rotY_a = rot_a * Eigen::Vector3f::UnitY();
//    const Eigen::Vector3f rotY_b = rot_b * Eigen::Vector3f::UnitY();
    const Eigen::Vector3f rotZ_a = rot_a * Eigen::Vector3f::UnitZ();
    const Eigen::Vector3f rotZ_b = rot_b * Eigen::Vector3f::UnitZ();


//    float angleX = pcl::rad2deg( acos( rotX_a.dot(rotX_b) ) );
//    float angleY = pcl::rad2deg( acos( rotY_a.dot(rotY_b) ) );
    float angleZ = pcl::rad2deg( acos( rotZ_a.dot(rotZ_b) ) );

    float angleXY = 0.f;
    if( !is_rotation_invariant )
    {
        Eigen::Vector2f rotXY_a = rotX_a.head(2);
        rotXY_a.normalize();
        Eigen::Vector2f rotXY_b = rotX_b.head(2);
        rotXY_b.normalize();
        angleXY = pcl::rad2deg ( acos (rotXY_a.dot(rotXY_b ) ) );

        if( is_rotational_symmetric )
            angleXY = std::min<float>(angleXY, fabs( 180.f - angleXY) );
    }

//    std::cout << " error_rotxy: " << angleXY << " error_rotx: " << angleX << " error_roty: " << angleY << " error_rotz: " << angleZ << std::endl;

    trans_error = (centroid_a.head(3)-centroid_b.head(3)).norm();
    rot_error = std::max<float>(angleXY, angleZ);

    if(trans_error > translation_error_threshold_m || rot_error > rotation_error_threshold_deg)
        return true;

    return false;
}

void
RecognitionEvaluator::checkMatchvector(const std::vector< std::pair<int, int> > &rec2gt,
                      const std::vector<Hypothesis> &rec_hyps,
                      const std::vector<Hypothesis> &gt_hyps,
                      const Eigen::Vector4f &model_centroid,
                      double &sum_translation_error, double &sum_rotational_error,
                      size_t &tp, size_t &fp, size_t &fn, bool is_rotation_invariant, bool is_rotational_symmetric)
{
    sum_translation_error = 0.;
    sum_rotational_error = 0.;
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
        if( computeError( rec_hyp.pose, gt_hyp.pose, model_centroid, trans_error, rot_error, is_rotation_invariant, is_rotational_symmetric))
        {
            fp++;

            if( gt_hyp.occlusion < occlusion_threshold)
                fn++;
        }
        else
        {
            tp++;
            sum_translation_error += trans_error;
            sum_rotational_error += rot_error;
        }
    }
}

std::vector< std::pair<int, int> >
RecognitionEvaluator::selectBestMatch (const std::vector<Hypothesis> &rec_hyps,
                 const std::vector<Hypothesis> &gt_hyps,
                 const Eigen::Vector4f &model_centroid,
                 size_t &tp, size_t &fp, size_t &fn,
                 double &sum_translation_error, double &sum_rotational_error, bool is_rotation_invariant, bool is_rotational_symmetric)
{
    // go through all possible permutations and return best match
    size_t elements_to_check = std::max(rec_hyps.size(), gt_hyps.size());
    size_t min_elements = std::min(rec_hyps.size(), gt_hyps.size());
    size_t max_offset = elements_to_check - min_elements;

    float best_fscore = -1;
    sum_translation_error = std::numeric_limits<float>::max();
    sum_rotational_error = std::numeric_limits<float>::max();
    tp=0, fp=0, fn=0;

    std::vector< std::pair<int, int> > best_match;

    /*
     * example:
     * a = [0 1 2 3 4 5]
     * b = [0 1 2]
     *
     * now we permutate through all possible combinations of b (smaller vector)
     * and slide the vector through the elements of a for each permuation iteration
     *
     * e.g.
     * b = [-1 -1 0 1 2 -1]
     * b = [-1 -1 -1 0 1 2]
     * b = [2 1 0 -1 -1 -1]
     * b = [-1 2 1 0 -1 -1]
     */
    std::vector<int> ids( min_elements );
    std::iota (std::begin(ids), std::end(ids), 0);

    bool gt_is_smaller = gt_hyps.size() < rec_hyps.size();

    do
    {
        for(size_t offset=0; offset<=max_offset; offset++)
        {
            std::vector< std::pair<int, int> > rec2gt_matches (elements_to_check);

            // initialize all b's to -1 (b = [-1 -1 -1 -1 -1 -1] )
            for(size_t i=0; i<elements_to_check; i++)
            {
                int rec_id, gt_id;

                if(gt_is_smaller)
                {
                    rec_id = i;
                    gt_id = -1;
                }
                else
                {
                    rec_id = -1;
                    gt_id = i;
                }

                rec2gt_matches[i] = std::pair<int, int>(rec_id, gt_id);
            }

            // now set the corresponding b values to their current permutation
            for(size_t i=0; i<min_elements; i++)
            {
                if(gt_is_smaller)
                    rec2gt_matches[i+offset].second = ids[i];
                else
                    rec2gt_matches[i+offset].first = ids[i];
            }

            double sum_translation_error_tmp;
            double sum_rotational_error_tmp;
            size_t tp_tmp, fp_tmp, fn_tmp;
            checkMatchvector(rec2gt_matches, rec_hyps, gt_hyps, model_centroid, sum_translation_error_tmp, sum_rotational_error_tmp,
                             tp_tmp, fp_tmp, fn_tmp, is_rotation_invariant, is_rotational_symmetric);

            float recall = 1.f;
            if (tp_tmp+fn_tmp) // if there are some ground-truth objects
                recall = (float)tp_tmp / (tp_tmp + fn_tmp);

            float precision = 1.f;
            if(tp_tmp+fp_tmp)   // if there are some recognized objects
                precision = (float)tp_tmp / (tp_tmp + fp_tmp);

            float fscore = 0.f;
            if ( precision+recall>std::numeric_limits<float>::epsilon() )
                fscore = 2 * precision * recall / (precision + recall);

            if ( (fscore > best_fscore) || (fscore==best_fscore && sum_translation_error_tmp/tp_tmp < sum_translation_error/tp))
            {
                best_fscore = fscore;
                sum_translation_error = sum_translation_error_tmp;
                sum_rotational_error = sum_rotational_error_tmp;
                tp = tp_tmp;
                fp = fp_tmp;
                fn = fn_tmp;
                best_match = rec2gt_matches;
            }
        }

    } while ( next_permutation( ids.begin(), ids.end()) );
    return best_match;
}


std::map<std::string, std::vector<Hypothesis> >
RecognitionEvaluator::readHypothesesFromFile( const std::string &filename )
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

void
RecognitionEvaluator::visualizeResults(const typename pcl::PointCloud<PointT>::Ptr &input_cloud, const bf::path & gt_path, const bf::path &recognition_results_path)
{
    std::map<std::string, std::vector<Hypothesis> > gt_hyps = readHypothesesFromFile( gt_path.string() );
    std::map<std::string, std::vector<Hypothesis> > rec_hyps = readHypothesesFromFile( recognition_results_path.string() );

    static pcl::visualization::PCLVisualizer::Ptr vis;
    static int vp1, vp2, vp3;
    if(!vis)
    {
        vis.reset (new pcl::visualization::PCLVisualizer ("results"));
        vis->setBackgroundColor(vis_params_->bg_color_(0), vis_params_->bg_color_(1), vis_params_->bg_color_(2));
        vis->createViewPort(0, 0, 1, 0.33, vp1);
        vis->createViewPort(0, 0.33, 1, 0.66, vp2);
        vis->createViewPort(0, 0.66, 1, 1, vp3);
    }

    vis->removeAllPointClouds();
    vis->removeAllShapes();
#if PCL_VERSION >= 100800
    vis->removeAllCoordinateSystems();
#endif

    input_cloud->sensor_orientation_ = Eigen::Quaternionf::Identity();
    input_cloud->sensor_origin_ = Eigen::Vector4f::Zero(4);
    vis->addPointCloud(input_cloud, "scene", vp1);
    vis->addText("scene", 10, 10, 14, 1., 1., 1., "scene", vp1);
    vis->addText("ground-truth", 10, 10, 14, 1., 1., 1., "gt", vp2);
    vis->addText("recognition results", 10, 10, 14, 1., 1., 1., "rec", vp3);

    pcl::visualization::PointCloudColorHandlerCustom<PointT> gray (input_cloud, 255, 255, 255);
    vis->addPointCloud(input_cloud, gray, "input_vp2", vp2);
    vis->setPointCloudRenderingProperties( pcl::visualization::PCL_VISUALIZER_OPACITY, 0.2, "input_vp2");
    vis->addPointCloud(input_cloud, gray, "input_vp3", vp3);
    vis->setPointCloudRenderingProperties( pcl::visualization::PCL_VISUALIZER_OPACITY, 0.2, "input_vp3");


    for( const auto &m : models )
    {

        auto it = rec_hyps.find( m.first );
        if ( it != rec_hyps.end() )
        {
            typename pcl::PointCloud<PointT>::ConstPtr model_cloud = m.second.cloud;
            size_t counter = 0;
            for( const Hypothesis &hyp_vis : it->second )
            {
                typename pcl::PointCloud<PointT>::Ptr model_aligned(new pcl::PointCloud<PointT>());
                pcl::transformPointCloud(*model_cloud, *model_aligned, hyp_vis.pose);
                std::stringstream unique_id; unique_id << "rec_" << m.first << "_" << counter++;
                vis->addPointCloud(model_aligned, unique_id.str(), vp3);
            }
        }

        it = gt_hyps.find( m.first );
        if ( it != gt_hyps.end() )
        {
            typename pcl::PointCloud<PointT>::ConstPtr model_cloud = m.second.cloud;
            size_t counter = 0;
            for( const Hypothesis &hyp_vis : it->second )
            {
                typename pcl::PointCloud<PointT>::Ptr model_aligned(new pcl::PointCloud<PointT>());
                pcl::transformPointCloud(*model_cloud, *model_aligned, hyp_vis.pose);
                std::stringstream unique_id; unique_id << "gt_" << m.first << "_" << counter++;
                vis->addPointCloud(model_aligned, unique_id.str(), vp2);
            }
        }
    }

    vis->spin();

}

void
RecognitionEvaluator::compute_recognition_rate (size_t &total_tp, size_t &total_fp, size_t &total_fn)
{
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

    bf::path out_path = out_dir;
    out_path /= "recognition_results.txt";
    v4r::io::createDirForFileIfNotExist(out_path.string());
    std::ofstream of ( out_path.string().c_str() );

    std::vector<std::string> annotation_files = v4r::io::getFilesInDirectory( gt_dir, ".*.anno", true );

    total_tp = 0;
    total_fp = 0;
    total_fn = 0;

    for( const std::string anno_file : annotation_files )
    {
        bf::path gt_path = gt_dir;
        gt_path /= anno_file;

        std::string rec_file = anno_file;
        if(use_generated_hypotheses)
            boost::replace_last( rec_file, ".anno", ".generated_hyps");

        bf::path rec_path = or_dir;
        rec_path /= rec_file;

        if(!v4r::io::existsFile(rec_path.string()))
            continue;

        std::map<std::string, std::vector<Hypothesis> > gt_hyps = readHypothesesFromFile( gt_path.string() );
        std::map<std::string, std::vector<Hypothesis> > rec_hyps = readHypothesesFromFile( rec_path.string() );

        size_t tp_view = 0;
        size_t fp_view = 0;
        size_t fn_view = 0;
        double sum_translation_error_view = 0.;
        double sum_rotational_error_view = 0.;

        if(vis_)
        {
            vis_->removeAllPointClouds();
            vis_->removeAllShapes();
#if PCL_VERSION >= 100800
            vis_->removeAllCoordinateSystems();
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

            size_t tp_tmp=0, fp_tmp=0, fn_tmp=0;
            double sum_translation_error_tmp=0.;
            double sum_rotational_error_tmp=0.;
            std::vector< std::pair<int, int> > matches;

//            if( gt_hyps_tmp.empty() && rec_hyps_tmp.empty() )
//                continue;
//            else if( gt_hyps_tmp.empty() )
//            {
//                fp_tmp = rec_hyps_tmp.size();
//                matches = std::vector<std::pair<int,int> > (rec_hyps_tmp.size());
//                for(size_t r_id=0; r_id<rec_hyps_tmp.size(); r_id++)
//                    matches[r_id] = std::pair<int,int>(r_id, -1);
//            }
//            else if( rec_hyps_tmp.empty() )
//            {
//                for(const Hypothesis &gt_hyp : gt_hyps_tmp )
//                {
//                    if( gt_hyp.occlusion < occlusion_threshold) // only count if the gt object is not occluded
//                        fn_tmp++;

//                    matches = std::vector<std::pair<int,int> > (gt_hyps_tmp.size());
//                    for(size_t gt_id=0; gt_id<gt_hyps_tmp.size(); gt_id++)
//                        matches[gt_id] = std::pair<int,int>(-1, gt_id);
//                }
//            }
//            else
            {
                const Eigen::Vector4f &centroid = m.second.centroid;
                matches = selectBestMatch(rec_hyps_tmp, gt_hyps_tmp, centroid, tp_tmp, fp_tmp, fn_tmp,
                                          sum_translation_error_tmp, sum_rotational_error_tmp,
                                          m.second.is_rotation_invariant_, m.second.is_rotational_symmetric_);
            }


            tp_view+=tp_tmp;
            fp_view+=fp_tmp;
            fn_view+=fn_tmp;
            sum_translation_error_view += sum_translation_error_tmp;
            sum_rotational_error_view += sum_rotational_error_tmp;

            if(visualize_)
            {
                if(!vis_)
                {
                    vis_.reset (new pcl::visualization::PCLVisualizer ("results"));
                    vis_->createViewPort(0, 0, 1, 0.33, vp1_);
                    vis_->createViewPort(0, 0.33, 1, 0.66, vp2_);
                    vis_->createViewPort(0, 0.66, 1, 1, vp3_);
                    vis_->setBackgroundColor(vis_params_->bg_color_(0), vis_params_->bg_color_(1), vis_params_->bg_color_(2));
                    vis_->setBackgroundColor(vis_params_->bg_color_(0), vis_params_->bg_color_(1), vis_params_->bg_color_(2), vp1_);
                    vis_->setBackgroundColor(vis_params_->bg_color_(0), vis_params_->bg_color_(1), vis_params_->bg_color_(2), vp2_);
                    vis_->setBackgroundColor(vis_params_->bg_color_(0), vis_params_->bg_color_(1), vis_params_->bg_color_(2), vp3_);

                }

                size_t counter = 0;
                for ( const auto &match : matches )
                {
                    int rec_id = match.first;
                    int gt_id = match.second;

                    if ( rec_id >= 0 )
                    {
                        const Hypothesis &hyp_vis = rec_hyps_tmp[ rec_id ];
                        typename pcl::PointCloud<PointT>::ConstPtr model_cloud = m.second.cloud;
                        typename pcl::PointCloud<PointT>::Ptr model_aligned(new pcl::PointCloud<PointT>());
                        pcl::transformPointCloud(*model_cloud, *model_aligned, hyp_vis.pose);
                        std::stringstream unique_id; unique_id << m.first << "_" << counter;
                        vis_->addPointCloud(model_aligned, unique_id.str(), vp3_);

#if PCL_VERSION >= 100800
                        Eigen::Matrix4f tf_tmp = hyp_vis.pose;
                        Eigen::Matrix3f rot_tmp  = tf_tmp.block<3,3>(0,0);
                        Eigen::Vector3f trans_tmp = tf_tmp.block<3,1>(0,3);
                        Eigen::Affine3f affine_trans;
                        affine_trans.fromPositionOrientationScale(trans_tmp, rot_tmp, Eigen::Vector3f::Ones());
                        std::stringstream co_id; co_id << m.first << "_co_" << counter;
                        vis_->addCoordinateSystem(0.1f, affine_trans, co_id.str(), vp3_);
#endif
                        counter++;
                    }

                    if ( gt_id >= 0 )
                    {
                        const Hypothesis &hyp_vis = gt_hyps_tmp[ gt_id ];
                        typename pcl::PointCloud<PointT>::ConstPtr model_cloud = m.second.cloud;
                        typename pcl::PointCloud<PointT>::Ptr model_aligned(new pcl::PointCloud<PointT>());
                        pcl::transformPointCloud(*model_cloud, *model_aligned, hyp_vis.pose);
                        std::stringstream unique_id; unique_id << m.first << "_" << counter;

                        if(hyp_vis.occlusion > occlusion_threshold)
                        {
                            pcl::visualization::PointCloudColorHandlerCustom<PointT> green (model_aligned, 0, 0, 255);
                            vis_->addPointCloud(model_aligned, green, unique_id.str(), vp2_);
                        }
                        else
                            vis_->addPointCloud(model_aligned, unique_id.str(), vp2_);

#if PCL_VERSION >= 100800
                        Eigen::Matrix4f tf_tmp = hyp_vis.pose;
                        Eigen::Matrix3f rot_tmp  = tf_tmp.block<3,3>(0,0);
                        Eigen::Vector3f trans_tmp = tf_tmp.block<3,1>(0,3);
                        Eigen::Affine3f affine_trans;
                        affine_trans.fromPositionOrientationScale(trans_tmp, rot_tmp, Eigen::Vector3f::Ones());
                        std::stringstream co_id; co_id << m.first << "_co_" << counter;
                        vis_->addCoordinateSystem(0.1f, affine_trans, co_id.str(), vp2_);
#endif
                        counter++;
                    }
                }
            }
        }

        std::cout << anno_file << ": " << tp_view << " " << fp_view << " " << fn_view << " " << sum_translation_error_view << " " << sum_rotational_error_view << std::endl;
        of << anno_file << " " << tp_view << " " << fp_view << " " << fn_view << " " << sum_translation_error_view << " " << sum_rotational_error_view << std::endl;

        total_tp += tp_view;
        total_fp += fp_view;
        total_fn += fn_view;

        if(visualize_)
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
            vis_->addPointCloud(scene_cloud, "scene", vp1_);

//            pcl::visualization::PointCloudColorHandlerCustom<PointT> gray (scene_cloud, 255, 255, 255);
//            vis_->addPointCloud(scene_cloud, gray, "input_vp2", vp2_);
//            vis_->setPointCloudRenderingProperties( pcl::visualization::PCL_VISUALIZER_OPACITY, 0.2, "input_vp2");
//            vis_->addPointCloud(scene_cloud, gray, "input_vp3", vp3_);
//            vis_->setPointCloudRenderingProperties( pcl::visualization::PCL_VISUALIZER_OPACITY, 0.2, "input_vp3");

            vis_->addText( scene_name, 10, 10, 15, 1.f, 1.f, 1.f, "scene_text", vp1_);
            vis_->addText("ground-truth objects (occluded objects in blue)", 10, 10, 15, 1.f, 1.f, 1.f, "gt_text", vp2_);
            std::stringstream rec_text;
            rec_text << "recognized objects (tp: " << tp_view << ", fp: " << fp_view << ", fn: " << fn_view;
            if(tp_view)
            {
                rec_text << " trans_error: " << sum_translation_error_view/tp_view;
                rec_text << " rot_error: " << sum_rotational_error_view/tp_view;
            }
            rec_text << ")";
            vis_->addText(rec_text.str(), 10, 10, 15, 1.f, 1.f, 1.f, "rec_text", vp3_);
//            vis->resetCamera();
            vis_->spin();
            vis_.reset();
        }
    }
    of.close();
}

std::string RecognitionEvaluator::getModels_dir() const
{
    return models_dir;
}

void RecognitionEvaluator::setModels_dir(const std::string &value)
{
    models_dir = value;
    loadModels();
}

std::string RecognitionEvaluator::getTest_dir() const
{
    return test_dir;
}

void RecognitionEvaluator::setTest_dir(const std::string &value)
{
    test_dir = value;
}

std::string RecognitionEvaluator::getOr_dir() const
{
    return or_dir;
}

void RecognitionEvaluator::setOr_dir(const std::string &value)
{
    or_dir = value;
}

std::string RecognitionEvaluator::getGt_dir() const
{
    return gt_dir;
}

void RecognitionEvaluator::setGt_dir(const std::string &value)
{
    gt_dir = value;
}

bool RecognitionEvaluator::getUse_generated_hypotheses() const
{
    return use_generated_hypotheses;
}

void RecognitionEvaluator::setUse_generated_hypotheses(bool value)
{
    use_generated_hypotheses = value;
}

bool RecognitionEvaluator::getVisualize() const
{
    return visualize_;
}

void RecognitionEvaluator::setVisualize(bool value)
{
    visualize_ = value;
}

std::string RecognitionEvaluator::getOut_dir() const
{
    return out_dir;
}

void RecognitionEvaluator::setOut_dir(const std::string &value)
{
    out_dir = value;
}

void RecognitionEvaluator::loadModels()
{
    std::vector<std::string> model_filenames = v4r::io::getFilesInDirectory( models_dir, "3D_model.pcd", true );
    for(const std::string &model_fn : model_filenames)
    {
        pcl::PointCloud<PointT>::Ptr model_cloud (new pcl::PointCloud<PointT>);
        bf::path model_full_path = models_dir;
        model_full_path /= model_fn;
        pcl::io::loadPCDFile( model_full_path.string(), *model_cloud );

        bf::path model_path = model_fn;
        const std::string model_name = model_path.parent_path().string();
        Model m;
        m.cloud = model_cloud;
        m.is_rotational_symmetric_ = std::find(rotational_symmetric_objects_.begin(), rotational_symmetric_objects_.end(), model_name) != rotational_symmetric_objects_.end();
        m.is_rotation_invariant_ = std::find(rotational_invariant_objects_.begin(), rotational_invariant_objects_.end(), model_name) != rotational_invariant_objects_.end();
        pcl::compute3DCentroid(*m.cloud, m.centroid);

        // model identity is equal folder name -> remove \"/3D_model.pcd\" from filename
        models[ model_name ] = m;
    }
}

std::vector<std::string>
RecognitionEvaluator::init(const std::vector<std::string> &params)
{
    po::options_description desc("Evaluation of object recognition\n==========================================\nAllowed options:\n");
    desc.add_options()
            ("help,h", "produce help message")
            ("groundtruth_dir,g", po::value<std::string>(&gt_dir), "Root directory containing annotation files (i.e. 4x4 ground-truth pose of each object with filename viewId_ModelId_ModelInstanceCounter.txt")
            ("rec_results_dir,r", po::value<std::string>(&or_dir), "Root directory containing the recognition results (same format as annotation files).")
            ("out_dir,o", po::value<std::string>(&out_dir)->default_value(out_dir), "Output directory where recognition results will be stored")
            ("trans_thresh", po::value<float>(&translation_error_threshold_m)->default_value(translation_error_threshold_m), "Maximal allowed translational error in metres")
            ("rot_thresh", po::value<float>(&rotation_error_threshold_deg)->default_value(rotation_error_threshold_deg), "Maximal allowed rotational error in degrees")
            ("occlusion_thresh", po::value<float>(&occlusion_threshold)->default_value(occlusion_threshold), "Occlusion threshold. Object with higher occlusion will be ignored in the evaluation")
            ("visualize,v", po::bool_switch(&visualize_), "visualize recognition results")
            ("models_dir,m", po::value<std::string>(&models_dir), "Only for visualization. Root directory containing the model files (i.e. filenames 3D_model.pcd).")
            ("test_dir,t", po::value<std::string>(&test_dir), "Only for visualization. Root directory containing the scene files.")
            ("use_generated_hypotheses", po::bool_switch(&use_generated_hypotheses), "if true, computes recognition rate for all generated hypotheses instead of verified ones.")
            ;
    po::variables_map vm;
    po::parsed_options parsed = po::command_line_parser(params).options(desc).allow_unregistered().run();
    std::vector<std::string> unused_params = po::collect_unrecognized(parsed.options, po::include_positional);
    po::store(parsed, vm);
    if (vm.count("help")) { std::cout << desc << std::endl;}
    try  {  po::notify(vm); }
    catch( std::exception& e)  { std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl; }

    loadModels();

    return unused_params;
}

float
RecognitionEvaluator::compute_recognition_rate_over_occlusion()
{
    std::stringstream description;
    description << "Tool to compute object instance recognition rate." << std::endl <<
                   "==================================================" << std::endl <<
                   "This will generate a text file containing:" << std::endl <<
                   "Column 1: occlusion" << std::endl <<
                   "Column 2: is recognized" << std::endl <<
                   "==================================================" << std::endl <<
                   "** Allowed options";


    bf::path out_path = out_dir;
    out_path /= "results_occlusion.txt";

    v4r::io::createDirForFileIfNotExist( out_path.string() );
    std::ofstream f( out_path.string() );
    std::cout << "Writing results to " << out_path.string() << "..." << std::endl;

    std::vector<std::string> annotation_files = v4r::io::getFilesInDirectory( gt_dir, ".*.anno", true );

    size_t num_recognized = 0;
    size_t num_total = 0;
    for( const std::string anno_file : annotation_files )
    {
        bf::path gt_path = gt_dir;
        gt_path /= anno_file;

        std::string rec_file = anno_file;
        if( use_generated_hypotheses )
            boost::replace_last( rec_file, ".anno", ".generated_hyps");

        bf::path rec_path = or_dir;
        rec_path /= rec_file;

        if(!v4r::io::existsFile(rec_path.string()))
            continue;

        std::map<std::string, std::vector<Hypothesis> > gt_hyps = readHypothesesFromFile( gt_path.string() );
        std::map<std::string, std::vector<Hypothesis> > rec_hyps = readHypothesesFromFile( rec_path.string() );

        for(auto const &gt_model_hyps : gt_hyps)
        {
            const std::string &model_name_gt = gt_model_hyps.first;
            const Model &m = models[ model_name_gt ];
            const std::vector<Hypothesis> &hyps = gt_model_hyps.second;

            for(const Hypothesis &h_gt : hyps)
            {
                bool is_recognized = false;

                const Eigen::Matrix4f &gt_pose = h_gt.pose;

                float occlusion = h_gt.occlusion;

                const auto it = rec_hyps.find( model_name_gt );
                if (it != rec_hyps.end())
                {
                    const std::vector<Hypothesis> &rec_model_hyps = it->second;
                    for(const Hypothesis &h_rec: rec_model_hyps)
                    {
                        const Eigen::Matrix4f &rec_pose = h_rec.pose;
                        float trans_error, rot_error;
                        if(! computeError( gt_pose, rec_pose, m.centroid, trans_error, rot_error, m.is_rotation_invariant_, m.is_rotational_symmetric_ ) )
                            is_recognized = true;
                    }
                }
                num_total++;

                if(is_recognized)
                    num_recognized++;

                f << occlusion << " " << is_recognized << std::endl;
            }
        }
    }
    f.close();
    std::cout << "Done!" << std::endl;

    return (float)num_recognized/num_total;
}


void
RecognitionEvaluator::checkIndividualHypotheses()
{

    std::vector<std::string> annotation_files = io::getFilesInDirectory( gt_dir, ".*.anno", true );

    pcl::visualization::PCLVisualizer::Ptr vis;
    int vp1, vp2;
    if(visualize_)
    {
        vis.reset (new pcl::visualization::PCLVisualizer ("results"));
        vis->setBackgroundColor(vis_params_->bg_color_(0), vis_params_->bg_color_(1), vis_params_->bg_color_(2));
        vis->createViewPort(0, 0, 1, 0.5, vp1);
        vis->createViewPort(0, 0.5, 1, 1, vp2);
    }

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


        if(visualize_)
        {
            vis->removeAllPointClouds();
            vis->removeAllShapes(vp1);
#if PCL_VERSION >= 100800
            vis->removeAllCoordinateSystems(vp1);
#endif
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
            vis->addText( scene_name, 10, 10, 15, 1.f, 1.f, 1.f, "scene_text", vp1);
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

            for( const Hypothesis &h : rec_hyps_tmp )
            {
                bool is_correct = false;
                float translation_error = std::numeric_limits<double>::max();
                float rotational_error = std::numeric_limits<double>::max();
                int best_matching_gt_id = -1;

                if( !gt_hyps_tmp.empty() )
                {
                    best_matching_gt_id = -1;
                    translation_error = std::numeric_limits<float>::max();
                    rotational_error = std::numeric_limits<float>::max();

                    for(size_t gt_id = 0; gt_id<gt_hyps_tmp.size(); gt_id++)
                    {
                        const Hypothesis &gt_hyp = gt_hyps_tmp[gt_id];
                        float trans_error_tmp, rot_error_tmp;
                        if( !computeError( h.pose, gt_hyp.pose, m.second.centroid, trans_error_tmp, rot_error_tmp, m.second.is_rotation_invariant_, m.second.is_rotational_symmetric_))
                            is_correct = true;

                        if(trans_error_tmp < translation_error)
                        {
                            translation_error = trans_error_tmp;
                            rotational_error = rot_error_tmp;
                            best_matching_gt_id = gt_id;
                        }
                    }
                }

                if(visualize_)
                {
                    vis->removePointCloud("model_cloud", vp2);
                    vis->removeAllShapes(vp2);
        #if PCL_VERSION >= 100800
                    vis->removeAllCoordinateSystems(vp2);
        #endif
                    pcl::PointCloud<PointT>::Ptr model_cloud = m.second.cloud;
                    pcl::PointCloud<PointT>::Ptr model_aligned(new pcl::PointCloud<PointT>());
                    pcl::transformPointCloud(*model_cloud, *model_aligned, h.pose);

                    vis->addPointCloud(model_aligned, "model_cloud", vp2);

                    if(is_correct)
                        vis->setBackgroundColor( 0, 255, 0, vp2);
                    else
                        vis->setBackgroundColor( 255, 0, 0, vp2);

#if PCL_VERSION >= 100800
                    Eigen::Matrix4f tf_tmp = h.pose;
                    Eigen::Matrix3f rot_tmp  = tf_tmp.block<3,3>(0,0);
                    Eigen::Vector3f trans_tmp = tf_tmp.block<3,1>(0,3);
                    Eigen::Affine3f affine_trans;
                    affine_trans.fromPositionOrientationScale(trans_tmp, rot_tmp, Eigen::Vector3f::Ones());
                    vis->addCoordinateSystem(0.1f, affine_trans, "model_co", vp2);
#endif
                    if(best_matching_gt_id>=0)
                    {
                        const Hypothesis &gt_hyp = gt_hyps_tmp[best_matching_gt_id];
                        pcl::PointXYZ center_rec, center_gt;
                        center_rec.getVector4fMap() = h.pose * m.second.centroid;
                        vis->addSphere(center_rec, 0.01, 0,0,255, "center_rec", vp2);
                        center_gt.getVector4fMap() = gt_hyp.pose * m.second.centroid;
                        vis->addSphere(center_gt, 0.01, 125,125,255, "center_gt", vp2);
                        vis->addLine(center_rec, center_gt, 0, 0, 255, "distance", vp2);
                        std::stringstream model_txt;
                        model_txt.precision(2);
                        model_txt << "Transl. error: " << translation_error*100.f << "cm; rotational error: " <<  rotational_error << "deg; occlusion: " << gt_hyp.occlusion;
                        vis->addText( model_txt.str(), 10, 10, 15, 1.f, 1.f, 1.f, "model_text", vp2);
                    }
//                    vis->resetCamera();
                    vis->spin();
                }
            }
        }
    }
}


Eigen::MatrixXi
RecognitionEvaluator::compute_confusion_matrix()
{
//    std::stringstream description;
//    description << "Tool to compute object instance recognition rate." << std::endl <<
//                   "==================================================" << std::endl <<
//                   "This will generate a text file containing:" << std::endl <<
//                   "Column 1: occlusion" << std::endl <<
//                   "Column 2: is recognized" << std::endl <<
//                   "==================================================" << std::endl <<
//                   "** Allowed options";


    bf::path out_path = out_dir;
    out_path /= "confusion_matrix.txt";

    v4r::io::createDirForFileIfNotExist( out_path.string() );
    std::ofstream f( out_path.string() );
    std::cout << "Writing results to " << out_path.string() << "..." << std::endl;

    std::vector<std::string> annotation_files = v4r::io::getFilesInDirectory( gt_dir, ".*.anno", true );

    Eigen::MatrixXi confusion_matrix = Eigen::MatrixXi::Zero( models.size(), models.size());

    std::map<std::string, int> modelname2modelid;
    size_t id=0;
    for(const auto &m:models)
    {
        modelname2modelid[m.first] = id;
//        f << id << ": " << m.first << std::endl;
        id++;
    }

    for( const std::string anno_file : annotation_files )
    {
        bf::path gt_path = gt_dir;
        gt_path /= anno_file;

        std::string rec_file = anno_file;
        if( use_generated_hypotheses )
            boost::replace_last( rec_file, ".anno", ".generated_hyps");

        bf::path rec_path = or_dir;
        rec_path /= rec_file;

        if(!v4r::io::existsFile(rec_path.string()))
        {
            LOG(WARNING) << "Recognition path " << rec_path.string() << " does not exist!";
            continue;
        }

        Eigen::MatrixXi tmp_confusion_matrix = Eigen::MatrixXi::Zero( models.size(), models.size());

        std::map<std::string, std::vector<Hypothesis> > gt_hyps_all_models = readHypothesesFromFile( gt_path.string() );
        std::map<std::string, std::vector<Hypothesis> > rec_hyps = readHypothesesFromFile( rec_path.string() );

        for(auto const &gt_model_hyps : gt_hyps_all_models)
        {
            const std::string &model_name_gt = gt_model_hyps.first;
            const Model &m = models[ model_name_gt ];

            for(const Hypothesis &h_gt : gt_model_hyps.second)
            {
                float occlusion = h_gt.occlusion;
                float lowest_trans_error = std::numeric_limits<float>::max();
                std::string best_match = "";

                for(auto const &rec_model_hyps : rec_hyps)
                {
                    const std::string &rec_model_name = rec_model_hyps.first;
                    for (const Hypothesis &h_rec : rec_model_hyps.second)
                    {
                        const Eigen::Vector4f centroid_a = h_gt.pose * m.centroid;
                        const Eigen::Vector4f centroid_b = h_rec.pose * m.centroid;

                        //ignore z

                        float trans_error = (h_gt.pose.block<2,1>(0,3)-h_rec.pose.block<2,1>(0,3)).norm();

                        if(trans_error < lowest_trans_error)
                        {
                            best_match = rec_model_name;
                            lowest_trans_error = trans_error;
                        }
                    }
                }

                if( !best_match.empty() && lowest_trans_error < translation_error_threshold_m)
                {
                    tmp_confusion_matrix( modelname2modelid[model_name_gt], modelname2modelid[best_match] ) ++;
                }
            }
        }

        std::cout << tmp_confusion_matrix << std::endl << std::endl
                  << "view accuracy: " << tmp_confusion_matrix.trace() << " / " <<
                     tmp_confusion_matrix.sum() << " (" <<
                     (float)tmp_confusion_matrix.trace() / tmp_confusion_matrix.sum() << ")" << std::endl;

        if(visualize_)
        {
            std::string scene_name (anno_file);
            boost::replace_last( scene_name, ".anno", ".pcd");

            bf::path scene_path = test_dir;
            scene_path /= scene_name;
            pcl::PointCloud<PointT>::Ptr scene_cloud (new pcl::PointCloud<PointT>);
            pcl::io::loadPCDFile( scene_path.string(), *scene_cloud);

            visualizeResults( scene_cloud, gt_path, rec_path);
        }

        confusion_matrix += tmp_confusion_matrix;
    }

    std::cout << confusion_matrix << std::endl << std::endl
              << "view accuracy: " << confusion_matrix.trace() << " / " <<
                 confusion_matrix.sum() << " (" <<
                 (float)confusion_matrix.trace() / confusion_matrix.sum() << ")" << std::endl;

    f << confusion_matrix;
    f.close();
    std::cout << "Done!" << std::endl;

    return confusion_matrix;
}

}
}
