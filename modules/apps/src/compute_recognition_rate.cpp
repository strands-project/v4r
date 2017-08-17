#include <v4r/apps/compute_recognition_rate.h>
#include <v4r/common/pcl_opencv.h>
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
    float dotpz = rotZ_a.dot(rotZ_b);
    dotpz = std::min( 0.9999999f, std::max( -0.999999999f, dotpz) );
    float angleZ = pcl::rad2deg( acos( dotpz ) );

    float angleXY = 0.f;
    if( !is_rotation_invariant )
    {
        float dotpxy = rotX_a.dot(rotX_b );
        dotpxy = std::min( 0.9999999f, std::max( -0.999999999f, dotpxy) );
        angleXY = pcl::rad2deg ( acos ( dotpxy ) );

        if( is_rotational_symmetric )
            angleXY = std::min<float>(angleXY, fabs( 180.f - angleXY) );
    }

//    std::cout << " error_rotxy: " << angleXY << " error_rotx: " << angleX << " error_roty: " << angleY << " error_rotz: " << angleZ << std::endl;

    trans_error = (centroid_a.head(3)-centroid_b.head(3)).norm();
    rot_error = std::max<float>(angleXY, angleZ);

    VLOG(1) << "translation error: " << trans_error << ", rotational error: " << angleXY << "(xy), " << angleZ << "(z), is rotational invariant? " << is_rotation_invariant << ", is rotational symmetric? " << is_rotational_symmetric;

    if(trans_error > translation_error_threshold_m || rot_error > rotation_error_threshold_deg)
        return true;

    return false;
}

void
RecognitionEvaluator::checkMatchvector(const std::vector< std::pair<int, int> > &rec2gt,
                                       const std::vector<Hypothesis> &rec_hyps,
                                       const std::vector<Hypothesis> &gt_hyps,
                                       const Eigen::Vector4f &model_centroid,
                                       std::vector<float> &translation_error,
                                       std::vector<float> &rotational_error,
                                       size_t &tp, size_t &fp, size_t &fn,
                                       bool is_rotation_invariant,
                                       bool is_rotational_symmetric)
{
    translation_error = std::vector<float>(rec2gt.size(), -1000.f);
    rotational_error = std::vector<float>(rec2gt.size(), -1000.f);
    tp = fp = fn = 0;

    for(size_t i=0; i<rec2gt.size(); i++)
    {
        int rec_id = rec2gt[i].first;
        int gt_id = rec2gt[i].second;

        VLOG(1) << "Checking rec_id " << rec_id << " and gt_id " << gt_id;

        if(gt_id < 0)
        {
            VLOG(1) << "Adding false positve because there is not a single ground-truth object with the same instance name";
            fp++;
            continue;
        }

        const Hypothesis &gt_hyp = gt_hyps [ gt_id ];

        if( rec_id < 0 )
        {
            if( gt_hyp.occlusion < occlusion_threshold) // only count if the gt object is not occluded
            {
                fn++;
                VLOG(1) << "Adding false negative because there is not a single detected object with the same instance name";
            }

            continue;
        }

        const Hypothesis &rec_hyp = rec_hyps [ rec_id ] ;

        if( computeError( rec_hyp.pose, gt_hyp.pose, model_centroid, translation_error[i], rotational_error[i], is_rotation_invariant, is_rotational_symmetric))
        {
            fp++;

            if( gt_hyp.occlusion < occlusion_threshold)
            {
                fn++;
                VLOG(1) << "Adding false negative";
            }
            else
                VLOG(1) << "Adding false positve";
        }
        else
        {
            VLOG(1) << "Adding true positve";
            tp++;
        }
    }
}

std::vector<std::vector<int> >
PermGenerator(int n, int k)
{
    std::vector<std::vector<int> > possible_permutations;

    std::vector<int> d(n);
    std::iota(d.begin(),d.end(),0);
    do
    {
        std::vector<int> p (k);
        for (int i = 0; i < k; i++)
            p[i] = d[i];

        possible_permutations.push_back(p);
        std::reverse(d.begin()+k,d.end());
    } while (next_permutation(d.begin(),d.end()));

    return possible_permutations;
}

std::vector< std::pair<int, int> >
RecognitionEvaluator::selectBestMatch (const std::vector<Hypothesis> &rec_hyps,
                                       const std::vector<Hypothesis> &gt_hyps,
                                       const Eigen::Vector4f &model_centroid,
                                       size_t &tp, size_t &fp, size_t &fn,
                                       std::vector<float> &translation_errors,
                                       std::vector<float> &rotational_errors,
                                       bool is_rotation_invariant,
                                       bool is_rotational_symmetric)
{
    float best_fscore = -1;
    tp=0, fp=0, fn=0;

    std::vector< std::pair<int, int> > best_match;

    size_t k = std::min(rec_hyps.size(), gt_hyps.size());
    size_t n = std::max(rec_hyps.size(), gt_hyps.size());

    std::vector<std::vector<int> > perms = PermGenerator( n, k );

    for(const std::vector<int> &perm : perms)
    {
        std::vector< std::pair<int, int> > rec2gt_matches ( k );


        size_t tp_tmp=0, fp_tmp=0, fn_tmp=0;

        if( rec_hyps.size() < n)
        {
            boost::dynamic_bitset<> taken_gts(n,0);

            for( size_t i=0; i< k; i++)
            {
                std::pair<int, int> &p = rec2gt_matches[i];
                p.first = i;
                p.second = perm[i];
                taken_gts.set(perm[i]);
            }

            for(size_t i=0; i<n; i++)
            {
                if(!taken_gts[i] && gt_hyps [i].occlusion < occlusion_threshold)
                    fn_tmp++;
            }
        }
        else
        {
            boost::dynamic_bitset<> taken_recs(n,0);
            for( size_t i=0; i< k; i++)
            {
                std::pair<int, int> &p = rec2gt_matches[i];
                p.first = perm[i];
                p.second = i;
                taken_recs.set(perm[i]);
            }
            for(size_t i=0; i<n; i++)
            {
                if(!taken_recs[i])
                    fp_tmp++;
            }
        }

        std::vector<float> translation_errors_tmp(k);
        std::vector<float> rotational_errors_tmp(k);
        for( size_t i=0; i< k; i++)
        {
            const std::pair<int,int> &x = rec2gt_matches[i];
            VLOG(1) <<x.first<<"/"<<x.second<< " " << std::endl;

          const Hypothesis &rec_hyp = rec_hyps [ x.first ] ;
          const Hypothesis &gt_hyp = gt_hyps [ x.second ] ;

          if( computeError( rec_hyp.pose, gt_hyp.pose, model_centroid, translation_errors_tmp[i], rotational_errors_tmp[i], is_rotation_invariant, is_rotational_symmetric))
          {

              if( gt_hyp.occlusion < occlusion_threshold)
              {
                  fn_tmp++;
                  fp_tmp++;
                  VLOG(1) << "Adding false negative and false positive";
              }
              else
              {
                  if(translation_errors_tmp[i] > translation_error_threshold_m )    //ignore rotation erros for occluded objects
                  {
                      fp_tmp++;
                      VLOG(1) << "Adding false positve but ignoring false negative due to occlusion";
                  }
                  else
                      VLOG(1) << "Ignoring due to occlusion";
              }
          }
          else
          {
              VLOG(1) << "Adding true positve";
              tp_tmp++;
          }

        }


        float recall = 1.f;
        if (tp_tmp+fn_tmp) // if there are some ground-truth objects
            recall = (float)tp_tmp / (tp_tmp + fn_tmp);

        float precision = 1.f;
        if(tp_tmp+fp_tmp)   // if there are some recognized objects
            precision = (float)tp_tmp / (tp_tmp + fp_tmp);

        float fscore = 0.f;
        if ( precision+recall>std::numeric_limits<float>::epsilon() )
            fscore = 2.f * precision * recall / (precision + recall);

        if ( (fscore > best_fscore) ) // || (fscore==best_fscore && translation_errors_tmp/tp_tmp < translation_errors/tp))
        {
            best_fscore = fscore;
            translation_errors = translation_errors_tmp;
            rotational_errors = rotational_errors_tmp;
            tp = tp_tmp;
            fp = fp_tmp;
            fn = fn_tmp;
            best_match = rec2gt_matches;
        }


//        std::vector<float> translation_errors_tmp;
//        std::vector<float> rotational_errors_tmp;
//        size_t tp_tmp, fp_tmp, fn_tmp;
//        checkMatchvector(rec2gt_matches, rec_hyps, gt_hyps, model_centroid, translation_errors_tmp, rotational_errors_tmp,
//                         tp_tmp, fp_tmp, fn_tmp, is_rotation_invariant, is_rotational_symmetric);

//        float recall = 1.f;
//        if (tp_tmp+fn_tmp) // if there are some ground-truth objects
//            recall = (float)tp_tmp / (tp_tmp + fn_tmp);

//        float precision = 1.f;
//        if(tp_tmp+fp_tmp)   // if there are some recognized objects
//            precision = (float)tp_tmp / (tp_tmp + fp_tmp);

//        float fscore = 0.f;
//        if ( precision+recall>std::numeric_limits<float>::epsilon() )
//            fscore = 2.f * precision * recall / (precision + recall);

//        if ( (fscore > best_fscore) ) // || (fscore==best_fscore && translation_errors_tmp/tp_tmp < translation_errors/tp))
//        {
//            best_fscore = fscore;
//            translation_errors = translation_errors_tmp;
//            rotational_errors = rotational_errors_tmp;
//            tp = tp_tmp;
//            fp = fp_tmp;
//            fn = fn_tmp;
//            best_match = rec2gt_matches;
//        }
    }


    VLOG(1) << "BEST MATCH: ";
    for(auto &x:best_match)
    {
      VLOG(1)<<x.first<<"/"<<x.second<< " ";
    }
    VLOG(1) << std::endl;

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
        std::istringstream os(occlusion_tmp);
        float visible;
        os >> visible;
        h.occlusion = 1.f-visible;

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
    vis->addText("scene", 10, 10, 14, vis_params_->text_color_(0), vis_params_->text_color_(1), vis_params_->text_color_(2), "scene", vp1);
    vis->addText("ground-truth", 10, 10, 14, vis_params_->text_color_(0), vis_params_->text_color_(1), vis_params_->text_color_(2), "gt", vp2);
    vis->addText("recognition results", 10, 10, 14, vis_params_->text_color_(0), vis_params_->text_color_(1), vis_params_->text_color_(2), "rec", vp3);

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
                   "Column 6: accumulated rotational error of all true positive objects" << std::endl <<
                   "Column 7: elapsed time in ms" << std::endl <<
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
        if(use_generated_hypotheses_)
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

        pcl::PointCloud<PointT>::Ptr all_hypotheses;
        pcl::PointCloud<PointT>::Ptr all_groundtruth_objects;

        if (save_images_to_disk_)
        {
            all_hypotheses.reset (new pcl::PointCloud<PointT>);
            all_groundtruth_objects.reset(new pcl::PointCloud<PointT>);
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
            std::vector<float> translation_errors_tmp;
            std::vector<float> rotational_errors_tmp;
            std::vector< std::pair<int, int> > matches;

            if( gt_hyps_tmp.empty() && rec_hyps_tmp.empty() )
                continue;
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
                                          translation_errors_tmp, rotational_errors_tmp,
                                          m.second.is_rotation_invariant_, m.second.is_rotational_symmetric_);
            }


            tp_view+=tp_tmp;
            fp_view+=fp_tmp;
            fn_view+=fn_tmp;

            float sum_translation_error_tmp = 0.f;
            float sum_rotational_error_tmp = 0.f;

            for(size_t t_id=0; t_id<translation_errors_tmp.size(); t_id++)
            {
                if( translation_errors_tmp[t_id]>0.f )
                {
                    sum_translation_error_tmp += translation_errors_tmp[t_id];
                    sum_rotational_error_tmp += rotational_errors_tmp[t_id];
                }
            }
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
                for( const Hypothesis &rec_hyp : rec_hyps_tmp )
                {
                    typename pcl::PointCloud<PointT>::ConstPtr model_cloud = m.second.cloud;
                    typename pcl::PointCloud<PointT>::Ptr model_aligned(new pcl::PointCloud<PointT>());
                    pcl::transformPointCloud(*model_cloud, *model_aligned, rec_hyp.pose);

                    std::stringstream unique_id; unique_id << m.first << "_" << counter++;
#if PCL_VERSION >= 100800
                        Eigen::Matrix4f tf_tmp = rec_hyp.pose;
                        Eigen::Matrix3f rot_tmp  = tf_tmp.block<3,3>(0,0);
                        Eigen::Vector3f trans_tmp = tf_tmp.block<3,1>(0,3);
                        Eigen::Affine3f affine_trans;
                        affine_trans.fromPositionOrientationScale(trans_tmp, rot_tmp, Eigen::Vector3f::Ones());
                        std::stringstream co_id; co_id << m.first << "_co_" << counter;
                        vis_->addCoordinateSystem(0.1f, affine_trans, co_id.str(), vp3_);
#endif
                    vis_->addPointCloud(model_aligned, unique_id.str(), vp3_);
                }

                for( const Hypothesis &gt_hyp : gt_hyps_tmp )
                {
                    typename pcl::PointCloud<PointT>::ConstPtr model_cloud = m.second.cloud;
                    typename pcl::PointCloud<PointT>::Ptr model_aligned(new pcl::PointCloud<PointT>());
                    pcl::transformPointCloud(*model_cloud, *model_aligned, gt_hyp.pose);

                    std::stringstream unique_id; unique_id << m.first << "_" << counter++;
#if PCL_VERSION >= 100800
                        Eigen::Matrix4f tf_tmp = gt_hyp.pose;
                        Eigen::Matrix3f rot_tmp  = tf_tmp.block<3,3>(0,0);
                        Eigen::Vector3f trans_tmp = tf_tmp.block<3,1>(0,3);
                        Eigen::Affine3f affine_trans;
                        affine_trans.fromPositionOrientationScale(trans_tmp, rot_tmp, Eigen::Vector3f::Ones());
                        std::stringstream co_id; co_id << m.first << "_co_" << counter;
                        vis_->addCoordinateSystem(0.1f, affine_trans, co_id.str(), vp3_);
#endif
                    vis_->addPointCloud(model_aligned, unique_id.str(), vp2_);
                }
            }
        }

        if(save_images_to_disk_)
        {
            if( visualize_errors_only_ && fp_view == 0)
                continue;

            const std::string img_output_dir = "/tmp/recognition_output_images/";
            v4r::Camera::Ptr kinect (new v4r::Camera(525.f, 525.f, 640, 480, 319.5, 239.5));

            std::string scene_name (anno_file);
            boost::replace_last( scene_name, ".anno", ".pcd");
            bf::path scene_path = test_dir;
            scene_path /= scene_name;
            pcl::PointCloud<PointT>::Ptr scene_cloud (new pcl::PointCloud<PointT>);
            pcl::io::loadPCDFile( scene_path.string(), *scene_cloud);
            boost::replace_last( scene_name, ".pcd", "");

            v4r::PCLOpenCVConverter<PointT> ocv;
            ocv.setCamera(kinect);
            ocv.setInputCloud( all_hypotheses );
            ocv.setBackgroundColor(255, 255, 255);
            ocv.setRemoveBackground(false);
            cv::Mat all_hypotheses_img = ocv.getRGBImage();
            bf::path img_path = img_output_dir; img_path /= scene_name; img_path /= "all_hypotheses.jpg";
            v4r::io::createDirForFileIfNotExist(img_path.string());
            cv::imwrite( img_path.string(), all_hypotheses_img);
            ocv.setInputCloud( all_groundtruth_objects );
            cv::Mat all_groundtruth_objects_img = ocv.getRGBImage();
            cv::imwrite( img_output_dir + "/" + scene_name + "/all_groundtruth_objects.jpg", all_groundtruth_objects_img);

            ocv.setInputCloud( scene_cloud );
            cv::Mat scene_cloud_img = ocv.getRGBImage();
            cv::imwrite( img_output_dir + "/" + scene_name + "/scene.jpg", scene_cloud_img);
        }


        // get time measurements
        size_t time_view = 0;
        {
            std::map<std::string, size_t> time_measurements;
            std::string time_file = anno_file;
            boost::replace_last(time_file, ".anno", ".times");
            bf::path time_path = or_dir;
            time_path /= time_file;

            std::ifstream time_f ( time_path.string() );
            std::string line;
            while (std::getline(time_f, line))
            {
                size_t elapsed_time;
                std::istringstream iss(line);
                iss >> elapsed_time;
                std::stringstream elapsed_time_ss; elapsed_time_ss << elapsed_time;

                const std::string time_description = line.substr( elapsed_time_ss.str().length() + 1 );
                time_measurements[time_description] = elapsed_time;
            }
            time_f.close();

            for(const auto &t_map:time_measurements)
            {
                VLOG(1) << t_map.first << ": " << t_map.second;
                if ( (t_map.first == "Computing normals" ) ||
                     (t_map.first == "Removing planes" ) ||
                     (t_map.first == "Generation of object hypotheses" ) ||
                     (t_map.first == "Computing noise model" ) ||
                     (t_map.first == "Noise model based cloud integration" ) ||
                     (t_map.first == "Verification of object hypotheses" ) )
                {
                    VLOG(1) << "count!";
                    time_view += t_map.second;
                }
            }
        }

        std::cout << anno_file << ": " << tp_view << " " << fp_view << " " << fn_view << " " << sum_translation_error_view << " " << sum_rotational_error_view << " " << time_view <<std::endl;
        of << anno_file << " " << tp_view << " " << fp_view << " " << fn_view << " " << sum_translation_error_view << " " << sum_rotational_error_view << " " << time_view << std::endl;

        total_tp += tp_view;
        total_fp += fp_view;
        total_fn += fn_view;

        if(visualize_)
        {
            if( visualize_errors_only_ && fp_view == 0)
                continue;

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

            vis_->addText( scene_name, 10, 10, 15, vis_params_->text_color_(0), vis_params_->text_color_(1), vis_params_->text_color_(2), "scene_text", vp1_);
            vis_->addText("ground-truth objects (occluded objects in blue, false ones in red, pose errors green)", 10, 10, 15, vis_params_->text_color_(0), vis_params_->text_color_(1), vis_params_->text_color_(2), "gt_text", vp2_);
            std::stringstream rec_text;
            rec_text << "recognized objects (tp: " << tp_view << ", fp: " << fp_view << ", fn: " << fn_view;
            if(tp_view)
            {
                rec_text << " trans_error: " << sum_translation_error_view/tp_view;
                rec_text << " rot_error: " << sum_rotational_error_view/tp_view;
            }
            rec_text << ")";
            vis_->addText(rec_text.str(), 10, 10, 15, vis_params_->text_color_(0), vis_params_->text_color_(1), vis_params_->text_color_(2), "rec_text", vp3_);
//            vis->resetCamera();
            vis_->spin();
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
    return use_generated_hypotheses_;
}

void RecognitionEvaluator::setUse_generated_hypotheses(bool value)
{
    use_generated_hypotheses_ = value;
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
    int verbosity = 0;

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
            ("visualize_errors_only", po::bool_switch(&visualize_errors_only_), "visualize only if there are errors (visualization must be on)")
            ("save_images_to_disk", po::bool_switch(&save_images_to_disk_), "if true, saves images to disk (visualization must be on)")
            ("highlight_errors", po::bool_switch(&highlight_errors_), "if true, highlights errors in the visualization")
            ("verbosity", po::value<int>(&verbosity)->default_value(verbosity), "verbosity level")
            ("models_dir,m", po::value<std::string>(&models_dir), "Only for visualization. Root directory containing the model files (i.e. filenames 3D_model.pcd).")
            ("test_dir,t", po::value<std::string>(&test_dir), "Only for visualization. Root directory containing the scene files.")
            ("use_generated_hypotheses", po::bool_switch(&use_generated_hypotheses_), "if true, computes recognition rate for all generated hypotheses instead of verified ones.")
            ;
    po::variables_map vm;
    po::parsed_options parsed = po::command_line_parser(params).options(desc).allow_unregistered().run();
    std::vector<std::string> unused_params = po::collect_unrecognized(parsed.options, po::include_positional);
    po::store(parsed, vm);
    if (vm.count("help")) { std::cout << desc << std::endl;}
    try  {  po::notify(vm); }
    catch( std::exception& e)  { std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl; }

    if(verbosity>=0)
    {
        FLAGS_logtostderr = 1;
        FLAGS_v = verbosity;
        std::cout << "Enabling verbose logging." << std::endl;
    }

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
        if( use_generated_hypotheses_ )
            boost::replace_last( rec_file, ".anno", ".generated_hyps");

        bf::path rec_path = or_dir;
        rec_path /= rec_file;

        if(!v4r::io::existsFile(rec_path.string()))
        {
            LOG(INFO) << "File " << rec_path.string() << " not found.";
            continue;
        }

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
        if(use_generated_hypotheses_)
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
            vis->addText( scene_name, 10, 10, 15, vis_params_->text_color_(0), vis_params_->text_color_(1), vis_params_->text_color_(2), "scene_text", vp1);
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
                        vis->addText( model_txt.str(), 10, 10, 15, vis_params_->text_color_(0), vis_params_->text_color_(1), vis_params_->text_color_(2), "model_text", vp2);
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
        if( use_generated_hypotheses_ )
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
                        VLOG(1) << h_gt.pose;
                        VLOG(1) << h_rec.pose;
                        VLOG(1) << trans_error;

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

        VLOG(1) << tmp_confusion_matrix << std::endl << std::endl
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
