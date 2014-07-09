/*
 * local_recognition_mian_dataset.cpp
 *
 *  Created on: Mar 24, 2012
 *      Author: aitor
 */

#include <pcl/console/parse.h>
#include <faat_pcl/3d_rec_framework/pc_source/model_only_source.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <faat_pcl/recognition/hv/hv_go_1.h>
#include <pcl/filters/passthrough.h>
#include <faat_pcl/3d_rec_framework/tools/or_evaluator.h>
#include <faat_pcl/3d_rec_framework/segmentation/multiplane_segmentation.h>
#include <pcl/features/organized_edge_detection.h>
#include <faat_pcl/utils/miscellaneous.h>
#include <faat_pcl/utils/filesystem_utils.h>
#include <faat_pcl/utils/noise_models.h>
#include <faat_pcl/registration/visibility_reasoning.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_point_to_plane_lls.h>
#include <pcl/common/angles.h>

float VX_SIZE_ICP_ = 0.005f;
bool PLAY_ = false;
std::string go_log_file_ = "test.txt";
float Z_DIST_ = 1.5f;
std::string GT_DIR_;
std::string MODELS_DIR_;
std::string MODELS_DIR_FOR_VIS_;
std::string HYPOTHESES_DIR_;
float model_scale = 1.f;
bool SHOW_GT_ = true;
int use_hv = 1;
float MAX_OCCLUSION_ = 1.01f;
bool UPPER_BOUND_ = false;
std::string HYPOTHESES_DIR_OUT_;
int SCENE_STEP_ = 1;
bool PRE_FILTER_ = false;
bool DO_ICP_ = false;
bool VISUALIZE_OUTLIERS_ = false;
bool VIS_TEXT_ = false;
int ICP_ITERATIONS_ = 5;
float ICP_CORRESP_DIST_ = 0.02f;
bool FORCE_UNORGANIZED_ = false;
bool FORCE_UNORGANIZED_PLANES_ = false;
float nwt_ = 0.9;
int MAX_THREADS_ = 4;

#define VISUALIZE_

inline void
getScenesInDirectory (bf::path & dir, std::string & rel_path_so_far, std::vector<std::string> & relative_paths)
{
  //list models in MODEL_FILES_DIR_ and return list
  bf::directory_iterator end_itr;
  for (bf::directory_iterator itr (dir); itr != end_itr; ++itr)
  {
    //check if its a directory, then get models in it
    if (bf::is_directory (*itr))
    {

#if BOOST_FILESYSTEM_VERSION == 3
      std::string so_far = rel_path_so_far + (itr->path ().filename ()).string () + "/";
#else
      std::string so_far = rel_path_so_far + (itr->path ()).filename () + "/";
#endif
      bf::path curr_path = itr->path ();
      getScenesInDirectory (curr_path, so_far, relative_paths);
    }
    else
    {
      //check that it is a ply file and then add, otherwise ignore..
      std::vector<std::string> strs;
#if BOOST_FILESYSTEM_VERSION == 3
      std::string file = (itr->path ().filename ()).string ();
#else
      std::string file = (itr->path ().filename ());
#endif

      boost::split (strs, file, boost::is_any_of ("."));
      std::string extension = strs[strs.size () - 1];

      if (extension == "pcd")
      {

#if BOOST_FILESYSTEM_VERSION == 3
        std::string path = rel_path_so_far + (itr->path ().filename ()).string ();
#else
        std::string path = rel_path_so_far + (itr->path ()).filename ();
#endif

        relative_paths.push_back (path);
      }
    }
  }
}

void
getModelsInDirectory (bf::path & dir, std::string & rel_path_so_far, std::vector<std::string> & relative_paths, std::string & ext)
{
  bf::directory_iterator end_itr;
  for (bf::directory_iterator itr (dir); itr != end_itr; ++itr)
  {
    //check if its a directory, then get models in it
    if (bf::is_directory (*itr))
    {
#if BOOST_FILESYSTEM_VERSION == 3
      std::string so_far = rel_path_so_far + (itr->path ().filename ()).string () + "/";
#else
      std::string so_far = rel_path_so_far + (itr->path ()).filename () + "/";
#endif

      bf::path curr_path = itr->path ();
      getModelsInDirectory (curr_path, so_far, relative_paths, ext);
    }
    else
    {
      //check that it is a ply file and then add, otherwise ignore..
      std::vector<std::string> strs;
#if BOOST_FILESYSTEM_VERSION == 3
      std::string file = (itr->path ().filename ()).string ();
#else
      std::string file = (itr->path ()).filename ();
#endif

      boost::split (strs, file, boost::is_any_of ("."));
      std::string extension = strs[strs.size () - 1];

      if (extension.compare (ext) == 0)
      {
#if BOOST_FILESYSTEM_VERSION == 3
        std::string path = rel_path_so_far + (itr->path ().filename ()).string ();
#else
        std::string path = rel_path_so_far + (itr->path ()).filename ();
#endif

        relative_paths.push_back (path);
      }
    }
  }
}

class go_params
{
public:
  float go_resolution;
  float go_iterations;
  float go_inlier_thres;
  float radius_clutter;
  float regularizer;
  float clutter_regularizer;
  bool go_use_replace_moves;
  int go_opt_type;
  float init_temp;
  float radius_normals_go_;
  float require_normals;
  bool go_init;
  bool detect_clutter;
  bool LS_short_circuit_;
  int visualize_cues_;
  float color_sigma_ab_;
  float color_sigma_l_;
  bool use_model_normals_;
  float min_contribution_;
  float hyp_penalty_;
  bool use_plane_hypotheses_;
  bool use_histogram_specification_;
  bool use_smooth_faces_;
  bool use_planes_on_different_sides_;
  int color_space_;
  bool visualize_accepted_;
  float duplicity_cm_weight_;
  bool go_use_supervoxels_;
  bool ignore_color_;
  float best_color_weight_;
  float duplicity_curvature_max_;
  float duplicity_weight_test_;
  bool use_normals_from_visible_;
  float weight_for_bad_normals_;
  bool use_clutter_exp_;

  void
  writeParamsToFile (std::ofstream & of)
  {
    of << "Params: \t" << go_opt_type << "\t";
    of << static_cast<int> (go_init) << "\t";
    of << radius_clutter << "\t" << clutter_regularizer << "\t";
    of << radius_normals_go_ << "\t" << static_cast<int> (go_use_replace_moves);
    of << std::endl;
  }
};

class TimingStructure
{
    std::vector<int> num_hypotheses_;
    std::vector<float> cues_computation_time_;
    std::vector<float> optimization_time_;
    std::vector<int> num_visible_model_points_;

    public:
        void addResult(int n_hyp, float t_cues, float t_opt, int num_points)
        {
            num_hypotheses_.push_back(n_hyp);
            cues_computation_time_.push_back(t_cues);
            optimization_time_.push_back(t_opt);
            num_visible_model_points_.push_back(num_points);
        }

        void writeToFile(std::string & file)
        {
            std::ofstream of;
            of.open (file.c_str ());

            for(size_t i=0; i < num_hypotheses_.size(); i++)
            {
                of << num_hypotheses_[i] << "\t";
                of << num_visible_model_points_[i] << "\t";
                of << cues_computation_time_[i] << "\t";
                of << optimization_time_[i] << "\t";
                of << std::endl;
            }
        }
};

go_params parameters_for_go;

std::string STATISTIC_OUTPUT_FILE_ = "stats.txt";
std::string POSE_STATISTICS_OUTPUT_FILE_ = "pose_translation.txt";
std::string POSE_STATISTICS_ANGLE_OUTPUT_FILE_ = "pose_error.txt";
std::string TIMING_OUTPUT_FILE_ = "timing.txt";

struct results_and_parameters
{
    float sigma_l_;
    float sigma_ab_;
    faat_pcl::rec_3d_framework::or_evaluator::RecognitionStatisticsResults results_;
};

float TRANSLATION_ERROR_ALLOWED_ = 0.03f;
bool FILTER_DUPLICATES_ = false;

template<typename PointTModel, typename PointT>
  faat_pcl::rec_3d_framework::or_evaluator::RecognitionStatisticsResults
  recognizeAndVisualize (typename boost::shared_ptr<faat_pcl::rec_3d_framework::ModelOnlySource<PointTModel, PointT> > & source,
                         std::string & scene_file,
                         std::string model_ext = "ply")
  {

    TimingStructure timing;

    faat_pcl::rec_3d_framework::or_evaluator::OREvaluator<PointT> or_eval;
    or_eval.setGTDir(GT_DIR_);
    or_eval.setModelsDir(MODELS_DIR_);
    or_eval.setCheckPose(true);
    or_eval.setScenesDir(scene_file);
    or_eval.setModelFileExtension(model_ext);
    or_eval.setReplaceModelExtension(false);
    or_eval.setDataSource(source);
    or_eval.setCheckPose(true);
    or_eval.setMaxCentroidDistance(TRANSLATION_ERROR_ALLOWED_);
    or_eval.setMaxOcclusion(MAX_OCCLUSION_);
    or_eval.useMaxOcclusion(true);

    faat_pcl::rec_3d_framework::or_evaluator::OREvaluator<PointT> or_hypotheses;
    or_hypotheses.setGTDir(HYPOTHESES_DIR_);
    or_hypotheses.setModelsDir(MODELS_DIR_);
    or_hypotheses.setModelFileExtension(model_ext);
    or_hypotheses.setReplaceModelExtension(false);
    or_hypotheses.setDataSource(source);

    std::ofstream logfile_stream;
    logfile_stream.open (go_log_file_.c_str ());
    parameters_for_go.writeParamsToFile (logfile_stream);

    boost::shared_ptr<faat_pcl::GlobalHypothesesVerification_1<PointT, PointT> > go (new faat_pcl::GlobalHypothesesVerification_1<PointT, PointT>);
    go->setResolution (parameters_for_go.go_resolution);
    go->setMaxIterations (parameters_for_go.go_iterations);
    go->setInlierThreshold (parameters_for_go.go_inlier_thres);
    go->setRadiusClutter (parameters_for_go.radius_clutter);
    go->setRegularizer (parameters_for_go.regularizer);
    go->setClutterRegularizer (parameters_for_go.clutter_regularizer);
    go->setDetectClutter (parameters_for_go.detect_clutter);
    go->setOcclusionThreshold (0.01f);
    go->setOptimizerType (parameters_for_go.go_opt_type);
    go->setUseReplaceMoves (parameters_for_go.go_use_replace_moves);
    go->setInitialTemp (parameters_for_go.init_temp);
    go->setRadiusNormals (parameters_for_go.radius_normals_go_);
    go->setRequiresNormals (parameters_for_go.require_normals);
    go->setInitialStatus (parameters_for_go.go_init);
    go->setLSShortCircuit(parameters_for_go.LS_short_circuit_);
    go->setVisualizeGoCues(parameters_for_go.visualize_cues_);
    go->setHypPenalty(parameters_for_go.hyp_penalty_);
    go->setMinContribution(parameters_for_go.min_contribution_);
    go->setIgnoreColor(parameters_for_go.ignore_color_);
    go->setColorSigma(parameters_for_go.color_sigma_l_, parameters_for_go.color_sigma_ab_);
    go->setUsePointsOnPlaneSides(parameters_for_go.use_planes_on_different_sides_);
    go->setColorSpace(parameters_for_go.color_space_);
    go->setVisualizeAccepted(parameters_for_go.visualize_accepted_);
    go->setDuplicityCMWeight(parameters_for_go.duplicity_cm_weight_);
    go->setUseSuperVoxels(parameters_for_go.go_use_supervoxels_);
    go->setBestColorWeight(parameters_for_go.best_color_weight_);
    go->setDuplicityMaxCurvature(parameters_for_go.duplicity_curvature_max_);
    go->setDuplicityWeightTest(parameters_for_go.duplicity_weight_test_);
    go->setUseNormalsFromVisible(parameters_for_go.use_normals_from_visible_);
    go->setMaxThreads(MAX_THREADS_);
    go->setWeightForBadNormals(parameters_for_go.weight_for_bad_normals_);
    go->setUseClutterExp(parameters_for_go.use_clutter_exp_);

    boost::shared_ptr<faat_pcl::HypothesisVerification<PointT, PointT> > cast_hv_alg;
    cast_hv_alg = boost::static_pointer_cast<faat_pcl::HypothesisVerification<PointT, PointT> > (go);

    typedef typename pcl::PointCloud<PointT>::ConstPtr ConstPointInTPtr;
    typedef faat_pcl::rec_3d_framework::Model<PointT> ModelT;
    typedef boost::shared_ptr<ModelT> ModelTPtr;

    bf::path input = scene_file;
    std::vector<std::string> files_to_recognize;

    if (bf::is_directory (input))
    {
      std::vector<std::string> files;
      std::string start = "";
      std::string ext = std::string ("pcd");
      bf::path dir = input;
      getModelsInDirectory (dir, start, files, ext);
      std::cout << "Number of scenes in directory is:" << files.size () << std::endl;
      for (size_t i = 0; i < files.size (); i++)
      {
        std::cout << files[i] << std::endl;
        std::stringstream filestr;
        filestr << scene_file << files[i];
        std::string file = filestr.str ();
        files_to_recognize.push_back (file);
      }

      std::sort (files_to_recognize.begin (), files_to_recognize.end ());

      if(SCENE_STEP_ > 1)
      {
          std::map<std::string, bool> ignore_list;

          //some scenes will not be recognized, modify files to recognize accordingly
          std::vector<std::string> files_to_recognize_step;
          for(size_t i=0; i < files_to_recognize.size(); i++)
          {
              if( ((int)(i) % SCENE_STEP_) == 0)
              {
                  files_to_recognize_step.push_back(files_to_recognize[i]);
              }
              else
              {
                  std::string file_to_recognize = files_to_recognize[i];
                  boost::replace_all (file_to_recognize, scene_file, "");
                  ignore_list.insert(std::make_pair(file_to_recognize, true));
              }
          }

          std::cout << files_to_recognize.size() << " " << files_to_recognize_step.size() << std::endl;
          files_to_recognize = files_to_recognize_step;
          or_eval.setIgnoreList(ignore_list);
          or_hypotheses.setIgnoreList(ignore_list);
      }

      or_eval.setScenesDir(scene_file);
      or_eval.loadGTData();

      or_hypotheses.setScenesDir(scene_file);
      or_hypotheses.loadGTData();
    }
    else
    {
      files_to_recognize.push_back (scene_file);
    }

#ifdef VISUALIZE_
    pcl::visualization::PCLVisualizer vis ("Recognition results");
    int v1, v2, v3, v4, v5;

    if(SHOW_GT_)
    {
      vis.createViewPort (0.0, 0.5, 0.33, 1.0, v1);
      vis.createViewPort (0.33, 0.5, 0.66, 1.0, v2);
      vis.createViewPort (0.0, 0.0, 0.5, 0.5, v3);
      vis.createViewPort (0.5, 0.0, 1.0, 0.5, v4);
      vis.createViewPort (0.66, 0.5, 1.0, 1.0, v5);

      if(VIS_TEXT_)
      {
          vis.addText ("Ground truth", 1, 30, 18, 1, 0, 0, "gt_text", v4);
          vis.addText ("Recognition hypotheses", 1, 30, 18, 1, 0, 0, "recog_hyp_text", v2);
          vis.addText ("Recognition results", 1, 30, 18, 1, 0, 0, "recog_res_text", v3);
      }
    }
    else
    {
      vis.createViewPort (0.0, 0.0, 0.33, 1.0, v1);
      vis.createViewPort (0.33, 0, 0.66, 1.0, v2);
      vis.createViewPort (0.66, 0, 1.0, 1.0, v3);

      if(VIS_TEXT_)
      {
        vis.addText ("Recognition results", 1, 30, 18, 1, 0, 0, "recog_res_text", v2);
      }
    }
#endif

    for (size_t i = 0; i < files_to_recognize.size (); i++)
    {

      std::cout << parameters_for_go.color_sigma_ab_ << " " << parameters_for_go.color_sigma_l_ << std::endl;

      std::string file_to_recognize(files_to_recognize[i]);
      boost::replace_all (file_to_recognize, scene_file, "");

      boost::replace_all (file_to_recognize, ".pcd", "");

      std::string id_1 = file_to_recognize;

      typename pcl::PointCloud<PointT>::Ptr scene (new pcl::PointCloud<PointT>);
      pcl::io::loadPCDFile (files_to_recognize[i], *scene);

      pcl::PointCloud<pcl::Normal>::Ptr normal_cloud (new pcl::PointCloud<pcl::Normal>);

      if(scene->isOrganized() && (model_ext.compare("ply") != 0) && !FORCE_UNORGANIZED_) //ATTENTION!
      {
          pcl::NormalEstimationOMP<PointT, pcl::Normal> ne;
          ne.setRadiusSearch(0.02f);
          ne.setInputCloud (scene);
          ne.compute (*normal_cloud);

          faat_pcl::utils::noise_models::NguyenNoiseModel<PointT> nm;
          nm.setInputCloud(scene);
          nm.setInputNormals(normal_cloud);
          nm.setLateralSigma(0.001f);
          nm.setMaxAngle(70.f);
          nm.setUseDepthEdges(true);
          nm.compute();
          std::vector<float> weights;
          nm.getWeights(weights);
          nm.getFilteredCloudRemovingPoints(scene, nwt_);
      }

      typename pcl::PointCloud<PointT>::Ptr occlusion_cloud (new pcl::PointCloud<PointT>(*scene));



      if (Z_DIST_ > 0)
      {

        if(FORCE_UNORGANIZED_)
        {
          typename pcl::PointCloud<PointT>::Ptr voxelized (new pcl::PointCloud<PointT>());
          float VOXEL_SIZE_ICP_ = 0.001f;
          pcl::VoxelGrid<PointT> voxel_grid_icp;
          voxel_grid_icp.setInputCloud (scene);
          voxel_grid_icp.setLeafSize (VOXEL_SIZE_ICP_, VOXEL_SIZE_ICP_, VOXEL_SIZE_ICP_);
          voxel_grid_icp.filter (*voxelized);
          scene = voxelized;
        }

        pcl::PassThrough<PointT> pass_;
        pass_.setFilterLimits (0.f, Z_DIST_);
        pass_.setFilterFieldName ("z");
        pass_.setInputCloud (scene);
        pass_.setKeepOrganized (!FORCE_UNORGANIZED_);
        pass_.filter (*scene);

        std::cout << "cloud is organized:" << scene->isOrganized() << std::endl;
        std::cout << scene->width << " " << scene->height << " " << FORCE_UNORGANIZED_ << " " << Z_DIST_ << std::endl;
      }


      //Multiplane segmentation
      faat_pcl::MultiPlaneSegmentation<PointT> mps;
      mps.setInputCloud(scene);
      mps.setMinPlaneInliers(1000);
      mps.setResolution(parameters_for_go.go_resolution);
      mps.setMergePlanes(true);
      mps.segment(FORCE_UNORGANIZED_PLANES_);
      std::vector<faat_pcl::PlaneModel<PointT> > planes_found;
      planes_found = mps.getModels();

#ifdef VISUALIZE_
      float rgb_m;
      bool exists_m;

      typedef pcl::PointCloud<PointT> CloudM;
      typedef typename pcl::traits::fieldList<typename CloudM::PointType>::type FieldListM;
      pcl::for_each_type<FieldListM> (pcl::CopyIfFieldExists<typename CloudM::PointType, float> (scene->points[0],"rgb", exists_m, rgb_m));

      if(!exists_m)
      {
          pcl::visualization::PointCloudColorHandlerGenericField<PointT> scene_handler(scene, "z");
          vis.addPointCloud<PointT> (scene, scene_handler, "scene_cloud", v1);
      }
      else
      {
          pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_rgb(new pcl::PointCloud<pcl::PointXYZRGB>());
          pcl::copyPointCloud(*scene, *scene_rgb);
          pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> scene_handler(scene_rgb);
          vis.addPointCloud<pcl::PointXYZRGB> (scene_rgb, scene_handler, "scene_cloud", v1);
      }

      for(size_t kk=0; kk < planes_found.size(); kk++)
      {
        std::stringstream pname;
        pname << "plane_" << kk;

        pcl::visualization::PointCloudColorHandlerRandom<PointT> scene_handler(planes_found[kk].plane_cloud_);
        vis.addPointCloud<PointT> (planes_found[kk].plane_cloud_, scene_handler, pname.str(), v2);

        pname << "chull";
        vis.addPolygonMesh (*planes_found[kk].convex_hull_, pname.str(), v2);
      }

      std::vector < std::string > strs;
      boost::split (strs, files_to_recognize[i], boost::is_any_of ("/"));

      if(VIS_TEXT_)
          vis.addText (strs[strs.size() - 1], 1, 30, 18, 1, 0, 0, "scene_text", v1);
#endif

      boost::shared_ptr<std::vector<ModelTPtr> > models;
      boost::shared_ptr<std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > > transforms;
      models.reset(new std::vector<ModelTPtr>);
      transforms.reset(new std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >);

      or_hypotheses.getGroundTruthModelsAndPoses(id_1, models, transforms);

      if(use_hv)
      {
          //visualize results


          if(FILTER_DUPLICATES_)
          {

              //filter equal hypotheses...
              float max_centroid_diff_ = 0.01f;
              float max_rotation_ = 5;

              boost::shared_ptr<std::vector<ModelTPtr> > models_filtered;
              boost::shared_ptr<std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > > transforms_filtered;
              models_filtered.reset(new std::vector<ModelTPtr>);
              transforms_filtered.reset(new std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >);

              Eigen::Vector4f x,y,z;
              x = y = z = Eigen::Vector4f::Zero();
              x = Eigen::Vector4f::UnitX();
              y = Eigen::Vector4f::UnitY();
              z = Eigen::Vector4f::UnitZ();

              for (size_t kk = 0; kk < models->size (); kk++)
              {

                  Eigen::Vector4f centroid_kk = models->at(kk)->getCentroid();
                  centroid_kk[3] = 1.f;
                  centroid_kk = transforms->at(kk) * centroid_kk;
                  centroid_kk[3] = 0.f;
                  std::string id_kk = models->at(kk)->id_;

                  Eigen::Matrix4f trans = transforms->at(kk);
                  Eigen::Vector4f x_gt,y_gt,z_gt;
                  x_gt = trans * x; y_gt = trans * y; z_gt = trans * z;


                  bool found = false;
                  for(size_t jj=(kk+1); (jj < models->size() && !found); jj++)
                  {

                      std::string id_jj = models->at(jj)->id_;
                      if(id_kk.compare(id_jj) != 0)
                          continue;

                      Eigen::Vector4f centroid_jj = models->at(jj)->getCentroid();
                      centroid_jj[3] = 1.f;
                      centroid_jj = transforms->at(jj) * centroid_jj;
                      centroid_jj[3] = 0.f;

                      //is pose correct? compare centroids and rotation

                      float diff = ( centroid_jj - centroid_kk).norm();

                      if(diff < max_centroid_diff_)
                      {
                        std::cout << "Hypotheses are similar" << std::endl;

                        //rotional error, how?

                        Eigen::Vector4f x_eval,y_eval,z_eval;
                        x_eval = transforms->at(jj) * x;
                        y_eval = transforms->at(jj) * y;
                        z_eval = transforms->at(jj) * z;

                        float dotx, doty, dotz;
                        dotx = x_eval.dot(x_gt);
                        doty = y_eval.dot(y_gt);
                        dotz = z_eval.dot(z_gt);

                        if(dotx >= 1.f)
                            dotx = 0.9999f;

                        if(doty >= 1.f)
                            doty = 0.9999f;

                        if(dotz >= 1.f)
                            dotz = 0.9999f;

                        if(dotx <= -1.f)
                            dotx = -0.9999f;

                        if(doty <= -1.f)
                            doty = -0.9999f;

                        if(dotz <= -1.f)
                            dotz = -0.9999f;

                        float angle_x, angle_y, angle_z;
                        angle_x = std::abs(pcl::rad2deg(acos(dotx)));
                        angle_y = std::abs(pcl::rad2deg(acos(doty)));
                        angle_z = std::abs(pcl::rad2deg(acos(dotz)));

                        float avg_angle = (angle_x + angle_y + angle_z) / 3.f;
                        //std::cout << "angles:" << angle_x << " " << angle_y << " " << angle_z << "avg:" << avg_angle << std::endl;
                        //std::cout << "max rotation:" << max_rotation_ << std::endl;

                        if(avg_angle < max_rotation_)
                        {
                          found = true;
                          continue;
                        }
                      }
                  }

                  if(!found)
                  {
                      models_filtered->push_back(models->at(kk));
                      transforms_filtered->push_back(transforms->at(kk));
                  }
              }

              std::cout << "models size:" << models->size() << " filtered size:" << models_filtered->size() << std::endl;
              models = models_filtered;
              transforms = transforms_filtered;
          }

          std::vector<std::string> model_ids;

          std::vector<typename pcl::PointCloud<PointT>::ConstPtr> aligned_models;
          std::vector<pcl::PointCloud<pcl::Normal>::ConstPtr> aligned_normals;
          std::vector<pcl::PointCloud<pcl::PointXYZL>::Ptr> aligned_smooth_faces;

          aligned_models.resize (models->size ());
          aligned_smooth_faces.resize (models->size ());

          if(parameters_for_go.use_model_normals_)
            aligned_normals.resize (models->size ());

          int kept = 0;
          float fsv_threshold = 0.2f;

          for (size_t kk = 0; kk < models->size (); kk++, kept++)
          {
               ConstPointInTPtr model_cloud = models->at (kk)->getAssembled (parameters_for_go.go_resolution);
               typename pcl::PointCloud<PointT>::Ptr model_aligned (new pcl::PointCloud<PointT>);
               pcl::transformPointCloud (*model_cloud, *model_aligned, transforms->at (kk));

               if(PRE_FILTER_)
               {
                   if(scene->isOrganized())
                   {
                       //compute FSV for the model and occlusion_cloud
                       faat_pcl::registration::VisibilityReasoning<PointT> vr (525.f, 640, 480);
                       vr.setThresholdTSS (0.01f);
                       Eigen::Matrix4f identity = Eigen::Matrix4f::Identity();

                       float fsv_ij = 0;

                       if(parameters_for_go.use_model_normals_)
                       {
                           pcl::PointCloud<pcl::Normal>::ConstPtr normal_cloud = models->at (kk)->getNormalsAssembled (parameters_for_go.go_resolution);
                           typename pcl::PointCloud<pcl::Normal>::Ptr normal_aligned (new pcl::PointCloud<pcl::Normal>);
                           faat_pcl::utils::miscellaneous::transformNormals(normal_cloud, normal_aligned, transforms->at (kk));

                           if(models->at(kk)->getFlipNormalsBasedOnVP())
                           {
                               Eigen::Vector3f viewpoint = Eigen::Vector3f(0,0,0);

                               for(size_t i=0; i < model_aligned->points.size(); i++)
                               {
                                   Eigen::Vector3f n = normal_aligned->points[i].getNormalVector3fMap();
                                   n.normalize();
                                   Eigen::Vector3f p = model_aligned->points[i].getVector3fMap();
                                   Eigen::Vector3f d = viewpoint - p;
                                   d.normalize();
                                   if(n.dot(d) < 0)
                                   {
                                       normal_aligned->points[i].getNormalVector3fMap() = normal_aligned->points[i].getNormalVector3fMap() * -1;
                                   }
                               }
                           }

                           fsv_ij = vr.computeFSVWithNormals (occlusion_cloud, model_aligned, normal_aligned);
                       }
                       else
                       {
                           fsv_ij = vr.computeFSV (occlusion_cloud, model_aligned, identity);
                       }

                       //std::cout << "fsv_kk" << kk << " " << fsv_ij << std::endl;
                       if(fsv_ij > fsv_threshold)
                       {
                           kept--;
                           continue;
                       }
                   }
               }

               models->at(kept) = models->at(kk);
               transforms->at(kept) = transforms->at(kk);

               aligned_models[kept] = model_aligned;

               if(parameters_for_go.use_model_normals_)
               {
                   pcl::PointCloud<pcl::Normal>::ConstPtr normal_cloud = models->at (kk)->getNormalsAssembled (parameters_for_go.go_resolution);
                   typename pcl::PointCloud<pcl::Normal>::Ptr normal_aligned (new pcl::PointCloud<pcl::Normal>);
                   faat_pcl::utils::miscellaneous::transformNormals(normal_cloud, normal_aligned, transforms->at (kk));

                   if(models->at(kk)->getFlipNormalsBasedOnVP())
                   {
                       Eigen::Vector3f viewpoint = Eigen::Vector3f(0,0,0);

                       int flip = 0;
                       for(size_t i=0; i < model_aligned->points.size(); i++)
                       {
                           Eigen::Vector3f n = normal_aligned->points[i].getNormalVector3fMap();
                           n.normalize();
                           Eigen::Vector3f p = model_aligned->points[i].getVector3fMap();
                           Eigen::Vector3f d = viewpoint - p;
                           d.normalize();
                           if(n.dot(d) < 0)
                           {
                               normal_aligned->points[i].getNormalVector3fMap() = normal_aligned->points[i].getNormalVector3fMap() * -1;
                               flip++;
                           }
                       }

                       /*std::cout << "Number of flipped normals:" << flip << " total:" << model_aligned->points.size() << std::endl;
                       pcl::visualization::PCLVisualizer vis("test");
                       pcl::visualization::PointCloudColorHandlerRandom<PointT> random_handler (model_aligned);
                       vis.addPointCloud<PointT>(model_aligned, random_handler, "model");
                       vis.addPointCloudNormals<PointT, pcl::Normal>(model_aligned, normal_aligned, 10, 0.02, "normals");
                       vis.addCoordinateSystem(0.1f);
                       vis.spin();*/
                   }

                   aligned_normals[kept] = normal_aligned;
               }

               pcl::PointCloud<pcl::PointXYZL>::Ptr faces = models->at(kk)->getAssembledSmoothFaces(parameters_for_go.go_resolution);
               pcl::PointCloud<pcl::PointXYZL>::Ptr faces_aligned(new pcl::PointCloud<pcl::PointXYZL>);
               pcl::transformPointCloud (*faces, *faces_aligned, transforms->at (kk));
               aligned_smooth_faces[kept] = faces_aligned;

               model_ids.push_back(models->at (kk)->id_);
          }

          models->resize(kept);
          transforms->resize(kept);
          aligned_models.resize(kept);
          aligned_smooth_faces.resize (kept);

          if(parameters_for_go.use_model_normals_)
            aligned_normals.resize (kept);
          std::cout << "kept hypotheses by FSV:" << kept << " " << models->size() << " " << fsv_threshold << std::endl;
          std::cout << "use normals:" << parameters_for_go.use_model_normals_ << std::endl;
          //DO ICP
          if(DO_ICP_)
          {
              pcl::ScopeTime t("ICP");
              float VOXEL_SIZE_ICP_ = parameters_for_go.go_resolution;
              typename pcl::PointCloud<PointT>::Ptr cloud_voxelized_icp (new pcl::PointCloud<PointT> ());
              pcl::VoxelGrid<PointT> voxel_grid_icp;
              voxel_grid_icp.setInputCloud (occlusion_cloud);
              voxel_grid_icp.setLeafSize (VOXEL_SIZE_ICP_, VOXEL_SIZE_ICP_, VOXEL_SIZE_ICP_);
              voxel_grid_icp.filter (*cloud_voxelized_icp);

              std::vector<ConstPointInTPtr> model_clouds_for_icp;
              model_clouds_for_icp.resize(models->size());
              for (size_t kk = 0; kk < models->size (); kk++)
              {
                  model_clouds_for_icp[kk] = models->at (kk)->getAssembled (VOXEL_SIZE_ICP_);
              }

              typename pcl::search::KdTree<PointT>::Ptr kdtree_scene(new pcl::search::KdTree<PointT>);
              kdtree_scene->setInputCloud(cloud_voxelized_icp);

#pragma omp parallel for schedule(dynamic, 1) num_threads(MAX_THREADS_)
              for (size_t kk = 0; kk < models->size (); kk++)
              {
                  //ConstPointInTPtr model_cloud = models->at (kk)->getAssembled (VOXEL_SIZE_ICP_);
                  ConstPointInTPtr model_cloud = model_clouds_for_icp[kk];
                  typename pcl::PointCloud<PointT>::Ptr model_aligned (new pcl::PointCloud<PointT>);
                  pcl::transformPointCloud (*model_cloud, *model_aligned, transforms->at (kk));

                  typename pcl::registration::CorrespondenceRejectorSampleConsensus<PointT>::Ptr
                                          rej (new pcl::registration::CorrespondenceRejectorSampleConsensus<PointT> ());

                  rej->setInputTarget (cloud_voxelized_icp);
                  rej->setMaximumIterations (1000);
                  rej->setInlierThreshold (0.01f);
                  rej->setInputSource (model_aligned);

                  pcl::IterativeClosestPoint<PointT, PointT> reg;
                  reg.addCorrespondenceRejector (rej);
                  reg.setInputTarget (cloud_voxelized_icp); //scene
                  reg.setInputSource (model_aligned); //model
                  reg.setSearchMethodTarget(kdtree_scene, true);
                  reg.setMaximumIterations (ICP_ITERATIONS_);
                  reg.setMaxCorrespondenceDistance (ICP_CORRESP_DIST_);
                  reg.setEuclideanFitnessEpsilon(1e-9);

                  typename pcl::PointCloud<PointT>::Ptr output_ (new pcl::PointCloud<PointT> ());
                  reg.align (*output_);

                  Eigen::Matrix4f icp_trans = reg.getFinalTransformation ();
                  transforms->at (kk) = icp_trans * transforms->at (kk);

                  {
                      ConstPointInTPtr model_cloud = models->at (kk)->getAssembled (parameters_for_go.go_resolution);
                      typename pcl::PointCloud<PointT>::Ptr model_aligned (new pcl::PointCloud<PointT>);
                      pcl::transformPointCloud (*model_cloud, *model_aligned, transforms->at (kk));

                      aligned_models[kk] = model_aligned;

                      /*if(parameters_for_go.use_model_normals_)
                      {
                          pcl::PointCloud<pcl::Normal>::ConstPtr normal_cloud = models->at (kk)->getNormalsAssembled (parameters_for_go.go_resolution);
                          typename pcl::PointCloud<pcl::Normal>::Ptr normal_aligned (new pcl::PointCloud<pcl::Normal>);
                          faat_pcl::utils::miscellaneous::transformNormals(normal_cloud, normal_aligned, transforms->at (kk));
                          aligned_normals[kk] = normal_aligned;
                      }*/

                      if(parameters_for_go.use_model_normals_)
                      {
                          //pcl::PointCloud<pcl::Normal>::ConstPtr normal_cloud = models->at (kk)->getNormalsAssembled (parameters_for_go.go_resolution);

                          pcl::PointCloud<pcl::Normal>::ConstPtr normal_cloud;

                          normal_cloud = models->at (kk)->getNormalsAssembled (parameters_for_go.go_resolution);

                          typename pcl::PointCloud<pcl::Normal>::Ptr normal_aligned (new pcl::PointCloud<pcl::Normal>);
                          faat_pcl::utils::miscellaneous::transformNormals(normal_cloud, normal_aligned, transforms->at (kk));

                          if(models->at(kk)->getFlipNormalsBasedOnVP())
                          {
                              Eigen::Vector3f viewpoint = Eigen::Vector3f(0,0,0);

                              for(size_t i=0; i < model_aligned->points.size(); i++)
                              {
                                  Eigen::Vector3f n = normal_aligned->points[i].getNormalVector3fMap();
                                  n.normalize();
                                  Eigen::Vector3f p = model_aligned->points[i].getVector3fMap();
                                  Eigen::Vector3f d = viewpoint - p;
                                  d.normalize();
                                  if(n.dot(d) < 0)
                                  {
                                      normal_aligned->points[i].getNormalVector3fMap() = normal_aligned->points[i].getNormalVector3fMap() * -1;
                                  }
                              }
                          }

                          aligned_normals[kk] = normal_aligned;
                      }
                  }
              }
          }

            if(model_ext.compare("ply") == 0)
            {
                std::vector<std::string> ply_paths_for_go_;
                std::vector< vtkSmartPointer < vtkTransform > > poses_ply_;

                for (size_t j = 0; j < transforms->size (); j++)
                {
                    std::stringstream pathPly;
                    pathPly << MODELS_DIR_FOR_VIS_ << "/" << models->at (j)->id_;

                    vtkSmartPointer < vtkTransform > poseTransform = vtkSmartPointer<vtkTransform>::New ();
                    vtkSmartPointer < vtkTransform > scale_models = vtkSmartPointer<vtkTransform>::New ();
                    scale_models->Scale(model_scale, model_scale, model_scale);

                    vtkSmartPointer < vtkMatrix4x4 > mat = vtkSmartPointer<vtkMatrix4x4>::New ();
                    for (size_t kk = 0; kk < 4; kk++)
                    {
                     for (size_t k = 0; k < 4; k++)
                     {
                       mat->SetElement (kk, k, transforms->at (j) (kk, k));
                     }
                    }

                    poseTransform->SetMatrix (mat);
                    poseTransform->Modified ();
                    poseTransform->Concatenate(scale_models);
                    ply_paths_for_go_.push_back(pathPly.str ());
                    poses_ply_.push_back(poseTransform);
                }

                go->setPlyPathsAndPoses(ply_paths_for_go_, poses_ply_);
            }

#ifdef VISUALIZE_
          std::stringstream name;
          for (size_t kk = 0; kk < models->size (); kk++)
          {
              ConstPointInTPtr model_cloud = models->at (kk)->getAssembled (0.005f);
              typename pcl::PointCloud<PointT>::Ptr model_aligned (new pcl::PointCloud<PointT>);
              pcl::transformPointCloud (*model_cloud, *model_aligned, transforms->at (kk));

              name << "hypotheses_" << kk;

              if(!exists_m)
              {
                pcl::visualization::PointCloudColorHandlerRandom<PointT> random_handler (model_aligned);
                vis.addPointCloud<PointT> (model_aligned, random_handler, name.str (), v2);
              }
              else
              {
                  pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_rgb(new pcl::PointCloud<pcl::PointXYZRGB>());
                  pcl::copyPointCloud(*model_aligned, *scene_rgb);
                  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> scene_handler(scene_rgb);
                  vis.addPointCloud<pcl::PointXYZRGB> (scene_rgb, scene_handler, name.str (), v2);
              }
          }

#endif

          if(scene->isOrganized() && !FORCE_UNORGANIZED_)
          {
              /*pcl::PointCloud<pcl::Normal>::Ptr normal_cloud (new pcl::PointCloud<pcl::Normal>);
              pcl::NormalEstimationOMP<PointT, pcl::Normal> ne;
              ne.setRadiusSearch(0.02f);
              ne.setInputCloud (scene);
              ne.compute (*normal_cloud);

              pcl::OrganizedEdgeBase<PointT, pcl::Label> oed;
              oed.setDepthDisconThreshold (0.02f); //at 1m, adapted linearly with depth
              oed.setMaxSearchNeighbors(100);
              oed.setEdgeType (pcl::OrganizedEdgeBase<PointT, pcl::Label>::EDGELABEL_OCCLUDING
              | pcl::OrganizedEdgeBase<pcl::PointXYZRGB, pcl::Label>::EDGELABEL_OCCLUDED
              | pcl::OrganizedEdgeBase<pcl::PointXYZRGB, pcl::Label>::EDGELABEL_NAN_BOUNDARY
              );
              oed.setInputCloud (occlusion_cloud);

              pcl::PointCloud<pcl::Label>::Ptr labels (new pcl::PointCloud<pcl::Label>);
              std::vector<pcl::PointIndices> indices2;
              oed.compute (*labels, indices2);

              pcl::PointCloud<pcl::PointXYZ>::Ptr occ_edges_full(new pcl::PointCloud<pcl::PointXYZ>);
              occ_edges_full->points.resize(occlusion_cloud->points.size());
              occ_edges_full->width = occlusion_cloud->width;
              occ_edges_full->height = occlusion_cloud->height;
              occ_edges_full->is_dense = occlusion_cloud->is_dense;

              for(size_t ik=0; ik < occ_edges_full->points.size(); ik++)
              {
                  occ_edges_full->points[ik].x =
                  occ_edges_full->points[ik].y =
                  occ_edges_full->points[ik].z = std::numeric_limits<float>::quiet_NaN();
              }

              for (size_t j = 0; j < indices2.size (); j++)
              {
                for (size_t i = 0; i < indices2[j].indices.size (); i++)
                {
                  occ_edges_full->points[indices2[j].indices[i]].getVector3fMap() = occlusion_cloud->points[indices2[j].indices[i]].getVector3fMap();
                }
              }

              go->setOcclusionEdges(occ_edges_full);*/
              go->setNormalsForClutterTerm(normal_cloud);
              go->setOcclusionCloud (occlusion_cloud);
              std::cout << "occlusion cloud being set" << std::endl;
          }

          if(parameters_for_go.use_model_normals_)
          {
              go->setRequiresNormals(true);
              go->addNormalsClouds(aligned_normals);
          }

          go->setSceneCloud (scene);
          //addModels
          go->addModels (aligned_models, true);

          if(parameters_for_go.use_plane_hypotheses_)
          {
              //append planar models
              go->addPlanarModels(planes_found);

              for(size_t kk=0; kk < planes_found.size(); kk++)
              {
                  std::stringstream plane_id;
                  plane_id << "plane_" << kk;
                  model_ids.push_back(plane_id.str());
              }
          }

          if(parameters_for_go.use_histogram_specification_)
          {
            go->setHistogramSpecification(true);
            if(parameters_for_go.use_smooth_faces_)
            {
                go->setSmoothFaces(aligned_smooth_faces);
            }
          }
          else
          {
            go->setHistogramSpecification(false);
          }

          go->setObjectIds(model_ids);

          std::cout << aligned_models.size() << " " << planes_found.size() << std::endl;

          //verify
          go->verify ();
          std::vector<bool> mask_hv;
          go->getMask (mask_hv);

          float t_cues = go->getCuesComputationTime();
          float t_opt = go->getOptimizationTime();
          int num_p = go->getNumberOfVisiblePoints();

          timing.addResult(static_cast<int>(aligned_models.size() + planes_found.size()), t_cues, t_opt, num_p);

          std::vector<int> coming_from;
          coming_from.resize(aligned_models.size() + planes_found.size());
          for(size_t j=0; j < aligned_models.size(); j++)
           coming_from[j] = 0;

          for(size_t j=0; j < planes_found.size(); j++)
           coming_from[aligned_models.size() + j] = 1;

#ifdef VISUALIZE_
          if(SHOW_GT_)
          {

            if(!exists_m)
            {
              pcl::visualization::PointCloudColorHandlerGenericField<PointT> scene_handler(scene, "z");
              vis.addPointCloud<PointT> (scene, scene_handler, "scene_cloud_V4", v4);
            }
            else
            {
              pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_rgb(new pcl::PointCloud<pcl::PointXYZRGB>());
              pcl::copyPointCloud(*scene, *scene_rgb);
              pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> scene_handler(scene_rgb);
              vis.addPointCloud<pcl::PointXYZRGB> (scene_rgb, scene_handler, "scene_cloud_V4", v4);
            }

            or_eval.visualizeGroundTruth(vis, id_1, v4, false);

            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr smooth_cloud_ =  go->getSmoothClustersRGBCloud();
            if(smooth_cloud_)
            {
              pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> random_handler (smooth_cloud_);
              vis.addPointCloud<pcl::PointXYZRGBA> (smooth_cloud_, random_handler, "smooth_cloud", v5);
            }

          }

#endif

          boost::shared_ptr<std::vector<ModelTPtr> > verified_models(new std::vector<ModelTPtr>);
          boost::shared_ptr<std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > > verified_transforms;
          verified_transforms.reset(new std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >);

          for (size_t j = 0; j < mask_hv.size (); j++)
          {
            if(!mask_hv[j])
              continue;

            std::stringstream name;
            name << "cloud_" << j;

            if(coming_from[j] == 0)
            {

              verified_models->push_back(models->at(j));
              verified_transforms->push_back(transforms->at(j));

              ConstPointInTPtr model_cloud = models->at (j)->getAssembled (0.001f);
              typename pcl::PointCloud<PointT>::Ptr model_aligned (new pcl::PointCloud<PointT>);
              pcl::transformPointCloud (*model_cloud, *model_aligned, transforms->at (j));

              std::cout << models->at (j)->id_ << std::endl;

#ifdef VISUALIZE_
              if(!exists_m)
              {
                //pcl::visualization::PointCloudColorHandlerRandom<PointT> random_handler (model_aligned);
                //vis.addPointCloud<PointT> (model_aligned, random_handler, name.str (), v3);

                if(model_ext.compare("ply") == 0)
                {
                    std::stringstream pathPly;
                    pathPly << MODELS_DIR_FOR_VIS_ << "/" << models->at (j)->id_;

                    std::cout << pathPly.str() << std::endl;

                    vtkSmartPointer < vtkTransform > poseTransform = vtkSmartPointer<vtkTransform>::New ();
                    vtkSmartPointer < vtkTransform > scale_models = vtkSmartPointer<vtkTransform>::New ();
                    scale_models->Scale(model_scale, model_scale, model_scale);

                    vtkSmartPointer < vtkMatrix4x4 > mat = vtkSmartPointer<vtkMatrix4x4>::New ();
                    for (size_t kk = 0; kk < 4; kk++)
                    {
                     for (size_t k = 0; k < 4; k++)
                     {
                       mat->SetElement (kk, k, transforms->at (j) (kk, k));
                     }
                    }

                    poseTransform->SetMatrix (mat);
                    poseTransform->Modified ();
                    poseTransform->Concatenate(scale_models);

                    std::stringstream cluster_name;
                    cluster_name << "_ply_model_" << j;
                    vis.addModelFromPLYFile (pathPly.str (), poseTransform, cluster_name.str (), v3);
                }
              }
              else
              {
                  pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_rgb(new pcl::PointCloud<pcl::PointXYZRGB>());
                  pcl::copyPointCloud(*model_aligned, *scene_rgb);
                  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> scene_handler(scene_rgb);
                  vis.addPointCloud<pcl::PointXYZRGB> (scene_rgb, scene_handler, name.str (), v3);
              }
#endif

            }
            else
            {
#ifdef VISUALIZE_
              std::stringstream pname;
              pname << "plane_" << j;

              /*pcl::visualization::PointCloudColorHandlerRandom<pcl::PointXYZ> scene_handler(planes_found[j - models->size()].plane_cloud_);
              vis.addPointCloud<pcl::PointXYZ> (planes_found[j - models->size()].plane_cloud_, scene_handler, pname.str(), v3);*/

              pname << "chull";
              vis.addPolygonMesh (*planes_found[j - models->size()].convex_hull_, pname.str(), v3);
#endif
            }
          }

#ifdef VISUALIZE_

          if(VISUALIZE_OUTLIERS_)
          {
              std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> outlier_clouds_;
              std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> outlier_clouds_color, outlier_clouds_3d;
              go->getOutliersForAcceptedModels(outlier_clouds_);
              go->getOutliersForAcceptedModels(outlier_clouds_color, outlier_clouds_3d);

              for (size_t j = 0; j < outlier_clouds_.size (); j++)
              {
                  std::cout << outlier_clouds_3d[j]->points.size() << " " << outlier_clouds_color[j]->points.size() << std::endl;

                  {
                      std::stringstream name;
                      name << "cloud_outliers" << j;
                      pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> random_handler (outlier_clouds_3d[j], 255, 255, 0);
                      vis.addPointCloud<pcl::PointXYZ> (outlier_clouds_3d[j], random_handler, name.str (), v3);
                      vis.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, name.str());
                  }

                  {
                      std::stringstream name;
                      name << "cloud_outliers_color" << j;
                      pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> random_handler (outlier_clouds_color[j], 255, 0, 255);
                      vis.addPointCloud<pcl::PointXYZ> (outlier_clouds_color[j], random_handler, name.str (), v3);
                      vis.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, name.str());
                  }
              }
          }
#endif

          or_eval.addRecognitionResults(id_1, verified_models, verified_transforms);

          //log results into file
          logfile_stream << files_to_recognize[i] << "\t";
          go->writeToLog (logfile_stream, false);

          if(HYPOTHESES_DIR_OUT_.compare("") != 0)
          {
              //save hypotheses that were selected into folder
              bf::path or_path = HYPOTHESES_DIR_OUT_;
              if(!bf::exists(or_path))
              {
                  bf::create_directory(or_path);
              }

              std::string seq_id;
              std::vector < std::string > strs_2;
              boost::split (strs_2, id_1, boost::is_any_of ("/\\"));
              seq_id = strs_2[0];

              if(strs_2.size() > 1)
              {
                  std::string dir_without_scene_name;
                  for(size_t j=0; j < (strs_2.size() - 1); j++)
                  {
                      dir_without_scene_name.append(strs_2[j]);
                      dir_without_scene_name.append("/");
                  }

                  std::stringstream dir;
                  dir << HYPOTHESES_DIR_OUT_ << "/" << dir_without_scene_name;
                  bf::path dir_sequence = dir.str();
                  bf::create_directories(dir_sequence);
              }

              std::map<std::string, int>::iterator instances_per_scene_it;
              std::map<std::string, int> instances_per_scene;

              std::stringstream model_ext_str;
              model_ext_str << "." << model_ext;
              for(size_t k=0; k < verified_models->size(); k++)
              {
                  std::string model_id = verified_models->at(k)->id_;
                  boost::replace_all (model_id, model_ext_str.str(), "");

                  instances_per_scene_it = instances_per_scene.find(model_id);
                  int inst = 0;
                  if(instances_per_scene_it != instances_per_scene.end())
                  {
                      inst = ++instances_per_scene_it->second;
                  }
                  else
                  {
                      instances_per_scene.insert(std::make_pair(model_id, 0));
                  }

                  std::stringstream pose_file_name;
                  pose_file_name << HYPOTHESES_DIR_OUT_ << "/" << id_1 << "_" << model_id << "_" << inst << ".txt";
                  std::cout << pose_file_name.str() << std::endl;
                  faat_pcl::utils::writeMatrixToFile (pose_file_name.str (),  verified_transforms->at(k));
              }
          }

      }
      else
      {
          or_eval.addRecognitionResults(id_1, models, transforms);
      }

#ifdef VISUALIZE_
      vis.setBackgroundColor (1.0, 1.0, 1.0);
      if (PLAY_)
      {
        vis.spinOnce (500.f, true);
      }
      else
      {
        vis.spin ();
      }

      vis.removePointCloud ("scene_cloud");
      vis.removeShape ("scene_text");
      for (size_t j = 0; j < models->size (); j++)
      {
        std::stringstream name;
        name << "cloud_" << j;
        vis.removePointCloud (name.str ());

        std::stringstream cluster_name;
        cluster_name << "_ply_model_" << j;
        vis.removeShape(cluster_name.str(), v3);
      }

      vis.removeAllPointClouds(v3);
      vis.removeAllShapes(v3);
      vis.removeAllPointClouds(v2);
      if(SHOW_GT_)
      {
        vis.removeAllPointClouds(v4);
      }

      vis.removeAllPointClouds(v5);
#endif
    }

    if(UPPER_BOUND_)
    {
        or_eval.computeStatisticsUpperBound();
    }
    else
    {
        or_eval.computeStatistics();
    }

    or_eval.saveStatistics(STATISTIC_OUTPUT_FILE_);
    or_eval.savePoseStatistics(POSE_STATISTICS_OUTPUT_FILE_);
    or_eval.savePoseStatisticsRotation(POSE_STATISTICS_ANGLE_OUTPUT_FILE_);
    faat_pcl::rec_3d_framework::or_evaluator::RecognitionStatisticsResults rsr;
    or_eval.getRecognitionStatisticsResults(rsr);
    logfile_stream.close ();

    timing.writeToFile(TIMING_OUTPUT_FILE_);

    return rsr;
  }

typedef pcl::ReferenceFrame RFType;

int CG_SIZE_ = 3;
float CG_THRESHOLD_ = 0.005f;

/*

Each optimizer method (except B&B) should be run with -go_init 0/1.
For local search, in addition -LS_short_circuit 0/1

Mian (without rhino)
-----

With LOCAL Search and replace moves 1

./bin/eval_hv_opt_strategies -models_dir /home/aitor/data/Mians_dataset/models/ -pcd_file /home/aitor/data/Mians_dataset/scenes/pcl_scenes -hv_method 0 -model_scale 0.001 -go_opt_type 0 -go_iterations 10000 -go_resolution 0.0025 -go_inlier_thres 0.005 -go_initial_temp 500 -PLAY 1 -go_require_normals 0 -go_init 0 -GT_DIR /home/aitor/data/Mians_dataset/gt_or_format_rhino_with_occ -models_dir_vis /home/aitor/data/Mians_dataset/models_with_rhino_vis/ -LS_short_circuit 0 -go_use_replace_moves 1 -vis_cues_ 0 -hypotheses_dir /home/aitor/data/Mians_dataset/hypotheses_before_hv_without_rhino -go_log_file /home/aitor/data/Mians_dataset/hv_eval_no_rhino/LS_RM1_init0.txt
./bin/eval_hv_opt_strategies -models_dir /home/aitor/data/Mians_dataset/models/ -pcd_file /home/aitor/data/Mians_dataset/scenes/pcl_scenes -hv_method 0 -model_scale 0.001 -go_opt_type 0 -go_iterations 10000 -go_resolution 0.0025 -go_inlier_thres 0.005 -go_initial_temp 500 -PLAY 1 -go_require_normals 0 -go_init 1 -GT_DIR /home/aitor/data/Mians_dataset/gt_or_format_rhino_with_occ -models_dir_vis /home/aitor/data/Mians_dataset/models_with_rhino_vis/ -LS_short_circuit 0 -go_use_replace_moves 1 -vis_cues_ 0 -hypotheses_dir /home/aitor/data/Mians_dataset/hypotheses_before_hv_without_rhino -go_log_file /home/aitor/data/Mians_dataset/hv_eval_no_rhino/LS_RM1_init1.txt
./bin/eval_hv_opt_strategies -models_dir /home/aitor/data/Mians_dataset/models/ -pcd_file /home/aitor/data/Mians_dataset/scenes/pcl_scenes -hv_method 0 -model_scale 0.001 -go_opt_type 0 -go_iterations 10000 -go_resolution 0.0025 -go_inlier_thres 0.005 -go_initial_temp 500 -PLAY 1 -go_require_normals 0 -go_init 0 -GT_DIR /home/aitor/data/Mians_dataset/gt_or_format_rhino_with_occ -models_dir_vis /home/aitor/data/Mians_dataset/models_with_rhino_vis/ -LS_short_circuit 1 -go_use_replace_moves 1 -vis_cues_ 0 -hypotheses_dir /home/aitor/data/Mians_dataset/hypotheses_before_hv_without_rhino -go_log_file /home/aitor/data/Mians_dataset/hv_eval_no_rhino/LS_SC_RM1_init0.txt
./bin/eval_hv_opt_strategies -models_dir /home/aitor/data/Mians_dataset/models/ -pcd_file /home/aitor/data/Mians_dataset/scenes/pcl_scenes -hv_method 0 -model_scale 0.001 -go_opt_type 0 -go_iterations 10000 -go_resolution 0.0025 -go_inlier_thres 0.005 -go_initial_temp 500 -PLAY 1 -go_require_normals 0 -go_init 1 -GT_DIR /home/aitor/data/Mians_dataset/gt_or_format_rhino_with_occ -models_dir_vis /home/aitor/data/Mians_dataset/models_with_rhino_vis/ -LS_short_circuit 1 -go_use_replace_moves 1 -vis_cues_ 0 -hypotheses_dir /home/aitor/data/Mians_dataset/hypotheses_before_hv_without_rhino -go_log_file /home/aitor/data/Mians_dataset/hv_eval_no_rhino/LS_SC_RM1_init1.txt

With LOCAL Search and replace moves 0

./bin/eval_hv_opt_strategies -models_dir /home/aitor/data/Mians_dataset/models/ -pcd_file /home/aitor/data/Mians_dataset/scenes/pcl_scenes -hv_method 0 -model_scale 0.001 -go_opt_type 0 -go_iterations 10000 -go_resolution 0.0025 -go_inlier_thres 0.005 -go_initial_temp 500 -PLAY 1 -go_require_normals 0 -go_init 0 -GT_DIR /home/aitor/data/Mians_dataset/gt_or_format_rhino_with_occ -models_dir_vis /home/aitor/data/Mians_dataset/models_with_rhino_vis/ -LS_short_circuit 0 -go_use_replace_moves 0 -vis_cues_ 0 -hypotheses_dir /home/aitor/data/Mians_dataset/hypotheses_before_hv_without_rhino -go_log_file /home/aitor/data/Mians_dataset/hv_eval_no_rhino/LS_RM0_init0.txt
./bin/eval_hv_opt_strategies -models_dir /home/aitor/data/Mians_dataset/models/ -pcd_file /home/aitor/data/Mians_dataset/scenes/pcl_scenes -hv_method 0 -model_scale 0.001 -go_opt_type 0 -go_iterations 10000 -go_resolution 0.0025 -go_inlier_thres 0.005 -go_initial_temp 500 -PLAY 1 -go_require_normals 0 -go_init 1 -GT_DIR /home/aitor/data/Mians_dataset/gt_or_format_rhino_with_occ -models_dir_vis /home/aitor/data/Mians_dataset/models_with_rhino_vis/ -LS_short_circuit 0 -go_use_replace_moves 0 -vis_cues_ 0 -hypotheses_dir /home/aitor/data/Mians_dataset/hypotheses_before_hv_without_rhino -go_log_file /home/aitor/data/Mians_dataset/hv_eval_no_rhino/LS_RM0_init1.txt
./bin/eval_hv_opt_strategies -models_dir /home/aitor/data/Mians_dataset/models/ -pcd_file /home/aitor/data/Mians_dataset/scenes/pcl_scenes -hv_method 0 -model_scale 0.001 -go_opt_type 0 -go_iterations 10000 -go_resolution 0.0025 -go_inlier_thres 0.005 -go_initial_temp 500 -PLAY 1 -go_require_normals 0 -go_init 0 -GT_DIR /home/aitor/data/Mians_dataset/gt_or_format_rhino_with_occ -models_dir_vis /home/aitor/data/Mians_dataset/models_with_rhino_vis/ -LS_short_circuit 1 -go_use_replace_moves 0 -vis_cues_ 0 -hypotheses_dir /home/aitor/data/Mians_dataset/hypotheses_before_hv_without_rhino -go_log_file /home/aitor/data/Mians_dataset/hv_eval_no_rhino/LS_SC_RM0_init0.txt
./bin/eval_hv_opt_strategies -models_dir /home/aitor/data/Mians_dataset/models/ -pcd_file /home/aitor/data/Mians_dataset/scenes/pcl_scenes -hv_method 0 -model_scale 0.001 -go_opt_type 0 -go_iterations 10000 -go_resolution 0.0025 -go_inlier_thres 0.005 -go_initial_temp 500 -PLAY 1 -go_require_normals 0 -go_init 1 -GT_DIR /home/aitor/data/Mians_dataset/gt_or_format_rhino_with_occ -models_dir_vis /home/aitor/data/Mians_dataset/models_with_rhino_vis/ -LS_short_circuit 1 -go_use_replace_moves 0 -vis_cues_ 0 -hypotheses_dir /home/aitor/data/Mians_dataset/hypotheses_before_hv_without_rhino -go_log_file /home/aitor/data/Mians_dataset/hv_eval_no_rhino/LS_SC_RM0_init1.txt

With SA

./bin/eval_hv_opt_strategies -models_dir /home/aitor/data/Mians_dataset/models/ -pcd_file /home/aitor/data/Mians_dataset/scenes/pcl_scenes -hv_method 0 -model_scale 0.001 -go_opt_type 3 -go_iterations 10000 -go_resolution 0.0025 -go_inlier_thres 0.005 -go_initial_temp 500 -PLAY 1 -go_require_normals 0 -go_init 0 -GT_DIR /home/aitor/data/Mians_dataset/gt_or_format_rhino_with_occ -models_dir_vis /home/aitor/data/Mians_dataset/models_with_rhino_vis/ -LS_short_circuit 0 -go_use_replace_moves 0 -vis_cues_ 0 -hypotheses_dir /home/aitor/data/Mians_dataset/hypotheses_before_hv_without_rhino -go_log_file /home/aitor/data/Mians_dataset/hv_eval_no_rhino/SA_RM0_init0.txt
./bin/eval_hv_opt_strategies -models_dir /home/aitor/data/Mians_dataset/models/ -pcd_file /home/aitor/data/Mians_dataset/scenes/pcl_scenes -hv_method 0 -model_scale 0.001 -go_opt_type 3 -go_iterations 10000 -go_resolution 0.0025 -go_inlier_thres 0.005 -go_initial_temp 500 -PLAY 1 -go_require_normals 0 -go_init 0 -GT_DIR /home/aitor/data/Mians_dataset/gt_or_format_rhino_with_occ -models_dir_vis /home/aitor/data/Mians_dataset/models_with_rhino_vis/ -LS_short_circuit 0 -go_use_replace_moves 1 -vis_cues_ 0 -hypotheses_dir /home/aitor/data/Mians_dataset/hypotheses_before_hv_without_rhino -go_log_file /home/aitor/data/Mians_dataset/hv_eval_no_rhino/SA_RM1_init0.txt

With TS

./bin/eval_hv_opt_strategies -models_dir /home/aitor/data/Mians_dataset/models/ -pcd_file /home/aitor/data/Mians_dataset/scenes/pcl_scenes -hv_method 0 -model_scale 0.001 -go_opt_type 1 -go_iterations 10000 -go_resolution 0.0025 -go_inlier_thres 0.005 -go_initial_temp 500 -PLAY 1 -go_require_normals 0 -go_init 0 -GT_DIR /home/aitor/data/Mians_dataset/gt_or_format_rhino_with_occ -models_dir_vis /home/aitor/data/Mians_dataset/models_with_rhino_vis/ -LS_short_circuit 0 -go_use_replace_moves 0 -vis_cues_ 0 -hypotheses_dir /home/aitor/data/Mians_dataset/hypotheses_before_hv_without_rhino -go_log_file /home/aitor/data/Mians_dataset/hv_eval_no_rhino/TS_RM0_init0.txt
./bin/eval_hv_opt_strategies -models_dir /home/aitor/data/Mians_dataset/models/ -pcd_file /home/aitor/data/Mians_dataset/scenes/pcl_scenes -hv_method 0 -model_scale 0.001 -go_opt_type 1 -go_iterations 10000 -go_resolution 0.0025 -go_inlier_thres 0.005 -go_initial_temp 500 -PLAY 1 -go_require_normals 0 -go_init 0 -GT_DIR /home/aitor/data/Mians_dataset/gt_or_format_rhino_with_occ -models_dir_vis /home/aitor/data/Mians_dataset/models_with_rhino_vis/ -LS_short_circuit 0 -go_use_replace_moves 1 -vis_cues_ 0 -hypotheses_dir /home/aitor/data/Mians_dataset/hypotheses_before_hv_without_rhino -go_log_file /home/aitor/data/Mians_dataset/hv_eval_no_rhino/TS_RM1_init0.txt

With BB
./bin/eval_hv_opt_strategies -models_dir /home/aitor/data/Mians_dataset/models/ -pcd_file /home/aitor/data/Mians_dataset/scenes/pcl_scenes -hv_method 0 -model_scale 0.001 -go_opt_type 2 -go_iterations 10000 -go_resolution 0.0025 -go_inlier_thres 0.005 -go_initial_temp 500 -PLAY 1 -go_require_normals 0 -go_init 0 -GT_DIR /home/aitor/data/Mians_dataset/gt_or_format_rhino_with_occ -models_dir_vis /home/aitor/data/Mians_dataset/models_with_rhino_vis/ -LS_short_circuit 0 -go_use_replace_moves 0 -vis_cues_ 0 -hypotheses_dir /home/aitor/data/Mians_dataset/hypotheses_before_hv_without_rhino -go_log_file /home/aitor/data/Mians_dataset/hv_eval_no_rhino/Branch_Bound.txt

Queens
-----

LS
./bin/eval_hv_opt_strategies -models_dir /home/aitor/data/queens_dataset/pcd_models/ -pcd_file /home/aitor/data/queens_dataset/hard_scenes -hv_method 0 -model_scale 1 -go_opt_type 0 -go_iterations 5000 -go_resolution 0.005 -go_inlier_thres 0.01 -go_initial_temp 500 -PLAY 1 -go_require_normals 0 -go_init 0 -GT_DIR /home/aitor/data/queens_dataset/gt_or_format_all_with_occ/ -models_dir_vis /home/aitor/data/queens_dataset/models_for_visualization/ -LS_short_circuit 0 -go_use_replace_moves 0 -vis_cues_ 0 -hypotheses_dir /home/aitor/data/queens_dataset/hypotheses_before_hv -go_log_file /home/aitor/data/queens_dataset/hv_eval/LS_RM0_init0.txt
./bin/eval_hv_opt_strategies -models_dir /home/aitor/data/queens_dataset/pcd_models/ -pcd_file /home/aitor/data/queens_dataset/hard_scenes -hv_method 0 -model_scale 1 -go_opt_type 0 -go_iterations 5000 -go_resolution 0.005 -go_inlier_thres 0.01 -go_initial_temp 500 -PLAY 1 -go_require_normals 0 -go_init 0 -GT_DIR /home/aitor/data/queens_dataset/gt_or_format_all_with_occ/ -models_dir_vis /home/aitor/data/queens_dataset/models_for_visualization/ -LS_short_circuit 0 -go_use_replace_moves 1 -vis_cues_ 0 -hypotheses_dir /home/aitor/data/queens_dataset/hypotheses_before_hv -go_log_file /home/aitor/data/queens_dataset/hv_eval/LS_RM1_init0.txt

LS_SC
./bin/eval_hv_opt_strategies -models_dir /home/aitor/data/queens_dataset/pcd_models/ -pcd_file /home/aitor/data/queens_dataset/hard_scenes -hv_method 0 -model_scale 1 -go_opt_type 0 -go_iterations 5000 -go_resolution 0.005 -go_inlier_thres 0.01 -go_initial_temp 500 -PLAY 1 -go_require_normals 0 -go_init 0 -GT_DIR /home/aitor/data/queens_dataset/gt_or_format_all_with_occ/ -models_dir_vis /home/aitor/data/queens_dataset/models_for_visualization/ -LS_short_circuit 1 -go_use_replace_moves 0 -vis_cues_ 0 -hypotheses_dir /home/aitor/data/queens_dataset/hypotheses_before_hv -go_log_file /home/aitor/data/queens_dataset/hv_eval/LS_SC_RM0_init0.txt
./bin/eval_hv_opt_strategies -models_dir /home/aitor/data/queens_dataset/pcd_models/ -pcd_file /home/aitor/data/queens_dataset/hard_scenes -hv_method 0 -model_scale 1 -go_opt_type 0 -go_iterations 5000 -go_resolution 0.005 -go_inlier_thres 0.01 -go_initial_temp 500 -PLAY 1 -go_require_normals 0 -go_init 0 -GT_DIR /home/aitor/data/queens_dataset/gt_or_format_all_with_occ/ -models_dir_vis /home/aitor/data/queens_dataset/models_for_visualization/ -LS_short_circuit 1 -go_use_replace_moves 1 -vis_cues_ 0 -hypotheses_dir /home/aitor/data/queens_dataset/hypotheses_before_hv -go_log_file /home/aitor/data/queens_dataset/hv_eval/LS_SC_RM1_init0.txt

SA
./bin/eval_hv_opt_strategies -models_dir /home/aitor/data/queens_dataset/pcd_models/ -pcd_file /home/aitor/data/queens_dataset/hard_scenes -hv_method 0 -model_scale 1 -go_opt_type 3 -go_iterations 10000 -go_resolution 0.005 -go_inlier_thres 0.01 -go_initial_temp 5000 -PLAY 1 -go_require_normals 0 -go_init 0 -GT_DIR /home/aitor/data/queens_dataset/gt_or_format_all_with_occ/ -models_dir_vis /home/aitor/data/queens_dataset/models_for_visualization/ -LS_short_circuit 0 -go_use_replace_moves 0 -vis_cues_ 0 -hypotheses_dir /home/aitor/data/queens_dataset/hypotheses_before_hv -go_log_file /home/aitor/data/queens_dataset/hv_eval/SA_RM0_init0.txt
./bin/eval_hv_opt_strategies -models_dir /home/aitor/data/queens_dataset/pcd_models/ -pcd_file /home/aitor/data/queens_dataset/hard_scenes -hv_method 0 -model_scale 1 -go_opt_type 3 -go_iterations 10000 -go_resolution 0.005 -go_inlier_thres 0.01 -go_initial_temp 5000 -PLAY 1 -go_require_normals 0 -go_init 0 -GT_DIR /home/aitor/data/queens_dataset/gt_or_format_all_with_occ/ -models_dir_vis /home/aitor/data/queens_dataset/models_for_visualization/ -LS_short_circuit 0 -go_use_replace_moves 1 -vis_cues_ 0 -hypotheses_dir /home/aitor/data/queens_dataset/hypotheses_before_hv -go_log_file /home/aitor/data/queens_dataset/hv_eval/SA_RM1_init0.txt

TS
./bin/eval_hv_opt_strategies -models_dir /home/aitor/data/queens_dataset/pcd_models/ -pcd_file /home/aitor/data/queens_dataset/hard_scenes -hv_method 0 -model_scale 1 -go_opt_type 1 -go_iterations 5000 -go_resolution 0.005 -go_inlier_thres 0.01 -go_initial_temp 500 -PLAY 1 -go_require_normals 0 -go_init 0 -GT_DIR /home/aitor/data/queens_dataset/gt_or_format_all_with_occ/ -models_dir_vis /home/aitor/data/queens_dataset/models_for_visualization/ -LS_short_circuit 0 -go_use_replace_moves 0 -vis_cues_ 0 -hypotheses_dir /home/aitor/data/queens_dataset/hypotheses_before_hv -go_log_file /home/aitor/data/queens_dataset/hv_eval/TS_RM0_init0.txt

BB
./bin/eval_hv_opt_strategies -models_dir /home/aitor/data/queens_dataset/pcd_models/ -pcd_file /home/aitor/data/queens_dataset/hard_scenes -hv_method 0 -model_scale 1 -go_opt_type 2 -go_iterations 5000 -go_resolution 0.005 -go_inlier_thres 0.01 -go_initial_temp 500 -PLAY 1 -go_require_normals 0 -go_init 0 -GT_DIR /home/aitor/data/queens_dataset/gt_or_format_all_with_occ/ -models_dir_vis /home/aitor/data/queens_dataset/models_for_visualization/ -LS_short_circuit 0 -go_use_replace_moves 0 -vis_cues_ 0 -hypotheses_dir /home/aitor/data/queens_dataset/hypotheses_before_hv -go_log_file /home/aitor/data/queens_dataset/hv_eval/Branch_Bound.txt

Willow
------

LS
./bin/eval_hv_opt_strategies -pcd_file /home/aitor/data/willow_dataset/ -models_dir /home/aitor/data/willow/models/ -go_require_normals 0 -GT_DIR /home/aitor/data/willow/willow_dataset_gt/ -model_scale 1 -go_opt_type 0 -Z_DIST 1.5 -detect_clutter 1 -go_resolution 0.005 -go_regularizer 2 -go_inlier_thres 0.01 -PLAY 1 -go_use_supervoxels 0 -LS_short_circuit 0 -go_use_replace_moves 0 -vis_cues_ 0 -hypotheses_dir /home/aitor/data/willow/hypotheses_before_hv -source_models 1

*/
int
main (int argc, char ** argv)
{
  std::string path = "";
  std::string pcd_file = "";
  int detect_clutter = 1;
  int hv_method = 0;
  int go_opt_type = 2;
  int go_iterations = 7000;
  bool go_use_replace_moves = true;
  float go_inlier_thres = 0.01f;
  float go_resolution = 0.005f;
  float init_temp = 1000.f;
  float radius_normals_go_ = 0.015f;
  bool go_require_normals = false;
  bool go_log = true;
  bool go_init = false;
  bool LS_short_circuit_ = false;
  int vis_cues_ = 0;
  float go_color_sigma_ab_ = 0.25f;
  float go_color_sigma_l_ = 0.25f;
  float go_hyp_penalty = 0.2f;
  float go_min_contribution = 100;
  float go_outlier_regularizer = 5;
  float go_radius_clutter = 0.03f;
  float go_clutter_regularizer = 10.f;

  bool use_model_normals = false;
  bool go_use_planes = true;
  bool specify_histograms = false;
  bool use_smooth_faces = true;
  bool use_planes_on_different_sides = false;
  int color_space = 0;
  bool visualize_accepted = false;
  float dup_cm_weight = 2.f;

  bool go_use_supervoxels = false;
  bool go_ignore_color = false;
  float go_bcw = 0.8;

  float go_duplicity_weight_test = 1.f;
  float go_duplicity_max_curvature = 0.03f;
  bool use_normals_from_visible = false;
  float weight_for_bad_normals = 0.1f;


  pcl::console::parse_argument (argc, argv, "-filter_duplicates", FILTER_DUPLICATES_);
  pcl::console::parse_argument (argc, argv, "-weight_for_bad_normals", weight_for_bad_normals);
  pcl::console::parse_argument (argc, argv, "-radius_normals_go", radius_normals_go_);
  pcl::console::parse_argument (argc, argv, "-use_normals_from_visible", use_normals_from_visible);
  pcl::console::parse_argument (argc, argv, "-nwt", nwt_);
  pcl::console::parse_argument (argc, argv, "-force_unorganized", FORCE_UNORGANIZED_);
  pcl::console::parse_argument (argc, argv, "-force_unorganized_planes", FORCE_UNORGANIZED_PLANES_);
  pcl::console::parse_argument (argc, argv, "-go_duplicity_weight_test", go_duplicity_weight_test);
  pcl::console::parse_argument (argc, argv, "-go_duplicity_max_curvature", go_duplicity_max_curvature);

  pcl::console::parse_argument (argc, argv, "-icp_correspondence_distance", ICP_CORRESP_DIST_);
  pcl::console::parse_argument (argc, argv, "-icp_iterations", ICP_ITERATIONS_);
  pcl::console::parse_argument (argc, argv, "-vis_text", VIS_TEXT_);
  pcl::console::parse_argument (argc, argv, "-vis_outliers", VISUALIZE_OUTLIERS_);
  pcl::console::parse_argument (argc, argv, "-go_bcw", go_bcw);
  pcl::console::parse_argument (argc, argv, "-max_trans_error", TRANSLATION_ERROR_ALLOWED_);
  pcl::console::parse_argument (argc, argv, "-DO_ICP", DO_ICP_);
  pcl::console::parse_argument (argc, argv, "-PRE_FILTER", PRE_FILTER_);
  pcl::console::parse_argument (argc, argv, "-go_use_supervoxels", go_use_supervoxels);
  pcl::console::parse_argument (argc, argv, "-go_ignore_color", go_ignore_color);
  pcl::console::parse_argument (argc, argv, "-go_radius_clutter", go_radius_clutter);
  pcl::console::parse_argument (argc, argv, "-go_clutter_regularizer", go_clutter_regularizer);
  pcl::console::parse_argument (argc, argv, "-scene_step", SCENE_STEP_);
  pcl::console::parse_argument (argc, argv, "-dup_cm_weight", dup_cm_weight);
  pcl::console::parse_argument (argc, argv, "-visualize_accepted", visualize_accepted);
  pcl::console::parse_argument (argc, argv, "-color_space", color_space);
  pcl::console::parse_argument (argc, argv, "-use_planes_on_different_sides", use_planes_on_different_sides);
  pcl::console::parse_argument (argc, argv, "-use_smooth_faces", use_smooth_faces);
  pcl::console::parse_argument (argc, argv, "-specify_histograms", specify_histograms);
  pcl::console::parse_argument (argc, argv, "-upper_bound", UPPER_BOUND_);
  pcl::console::parse_argument (argc, argv, "-vis_cues_", vis_cues_);
  pcl::console::parse_argument (argc, argv, "-LS_short_circuit", LS_short_circuit_);
  pcl::console::parse_argument (argc, argv, "-models_dir", path);
  pcl::console::parse_argument (argc, argv, "-pcd_file", pcd_file);
  pcl::console::parse_argument (argc, argv, "-detect_clutter", detect_clutter);
  pcl::console::parse_argument (argc, argv, "-hv_method", hv_method);
  pcl::console::parse_argument (argc, argv, "-model_scale", model_scale);
  pcl::console::parse_argument (argc, argv, "-go_opt_type", go_opt_type);
  pcl::console::parse_argument (argc, argv, "-go_iterations", go_iterations);
  pcl::console::parse_argument (argc, argv, "-go_use_replace_moves", go_use_replace_moves);
  pcl::console::parse_argument (argc, argv, "-go_inlier_thres", go_inlier_thres);
  pcl::console::parse_argument (argc, argv, "-go_resolution", go_resolution);
  pcl::console::parse_argument (argc, argv, "-go_initial_temp", init_temp);
  pcl::console::parse_argument (argc, argv, "-go_require_normals", go_require_normals);
  pcl::console::parse_argument (argc, argv, "-go_log", go_log);
  pcl::console::parse_argument (argc, argv, "-go_init", go_init);
  pcl::console::parse_argument (argc, argv, "-PLAY", PLAY_);
  pcl::console::parse_argument (argc, argv, "-go_log_file", go_log_file_);
  pcl::console::parse_argument (argc, argv, "-Z_DIST", Z_DIST_);
  pcl::console::parse_argument (argc, argv, "-show_gt", SHOW_GT_);
  pcl::console::parse_argument (argc, argv, "-go_use_model_normals", use_model_normals);
  pcl::console::parse_argument (argc, argv, "-go_hyp_penalty", go_hyp_penalty);
  pcl::console::parse_argument (argc, argv, "-go_min_contribution", go_min_contribution);
  pcl::console::parse_argument (argc, argv, "-go_outlier_regularizer", go_outlier_regularizer);
  pcl::console::parse_argument (argc, argv, "-go_use_planes", go_use_planes);
  pcl::console::parse_argument (argc, argv, "-go_color_sigma_l", go_color_sigma_l_);
  pcl::console::parse_argument (argc, argv, "-go_color_sigma_ab", go_color_sigma_ab_);
  pcl::console::parse_argument (argc, argv, "-stat_file", STATISTIC_OUTPUT_FILE_);
  pcl::console::parse_argument (argc, argv, "-pose_stats_file", POSE_STATISTICS_OUTPUT_FILE_);
  pcl::console::parse_argument (argc, argv, "-pose_stats_file_angle", POSE_STATISTICS_ANGLE_OUTPUT_FILE_);
  pcl::console::parse_argument (argc, argv, "-timing_file", TIMING_OUTPUT_FILE_);
  MODELS_DIR_FOR_VIS_ = path;

  pcl::console::parse_argument (argc, argv, "-models_dir_vis", MODELS_DIR_FOR_VIS_);
  pcl::console::parse_argument (argc, argv, "-GT_DIR", GT_DIR_);
  pcl::console::parse_argument (argc, argv, "-hypotheses_dir", HYPOTHESES_DIR_);
  pcl::console::parse_argument (argc, argv, "-max_occlusion", MAX_OCCLUSION_);
  pcl::console::parse_argument (argc, argv, "-use_hv", use_hv);
  pcl::console::parse_argument (argc, argv, "-hypotheses_dir_out", HYPOTHESES_DIR_OUT_);
  bool use_clutter_exp = false;
  pcl::console::parse_argument (argc, argv, "-use_clutter_exp", use_clutter_exp);

  MODELS_DIR_ = path;

  if (pcd_file.compare ("") == 0)
  {
    PCL_ERROR("Set the directory containing mians scenes using the -mians_scenes_dir [dir] option\n");
    return -1;
  }

  if (path.compare ("") == 0)
  {
    PCL_ERROR("Set the directory containing the models of mian dataset using the -models_dir [dir] option\n");
    return -1;
  }

  int source_models = 0; //0-ply models, 1-pcd models
  bool parameter_selection_ = false;

  pcl::console::parse_argument (argc, argv, "-source_models", source_models);
  pcl::console::parse_argument (argc, argv, "-parameter_selection", parameter_selection_);

  parameters_for_go.use_clutter_exp_ = use_clutter_exp;
  parameters_for_go.radius_normals_go_ = radius_normals_go_;
  parameters_for_go.radius_clutter = go_radius_clutter;
  parameters_for_go.clutter_regularizer = go_clutter_regularizer;
  parameters_for_go.regularizer = go_outlier_regularizer;
  parameters_for_go.init_temp = init_temp;
  parameters_for_go.go_init = go_init;
  parameters_for_go.go_inlier_thres = go_inlier_thres;
  parameters_for_go.go_iterations = go_iterations;
  parameters_for_go.go_opt_type = go_opt_type;
  parameters_for_go.require_normals = go_require_normals;
  parameters_for_go.go_resolution = go_resolution;
  parameters_for_go.go_use_replace_moves = go_use_replace_moves;
  parameters_for_go.detect_clutter = static_cast<bool> (detect_clutter);
  parameters_for_go.LS_short_circuit_ = static_cast<bool> (LS_short_circuit_);
  parameters_for_go.visualize_cues_ = vis_cues_;
  parameters_for_go.color_sigma_ab_ = go_color_sigma_ab_;
  parameters_for_go.color_sigma_l_ = go_color_sigma_l_;
  parameters_for_go.use_model_normals_ = use_model_normals;
  parameters_for_go.min_contribution_ = go_min_contribution;
  parameters_for_go.hyp_penalty_ = go_hyp_penalty;
  parameters_for_go.use_plane_hypotheses_ = go_use_planes;
  parameters_for_go.use_histogram_specification_ = specify_histograms;
  parameters_for_go.use_smooth_faces_ = use_smooth_faces;
  parameters_for_go.use_planes_on_different_sides_ = use_planes_on_different_sides;
  parameters_for_go.color_space_ = color_space;
  parameters_for_go.visualize_accepted_ = visualize_accepted;
  parameters_for_go.duplicity_cm_weight_ = dup_cm_weight;
  parameters_for_go.ignore_color_ = go_ignore_color;
  parameters_for_go.go_use_supervoxels_ = go_use_supervoxels;
  parameters_for_go.best_color_weight_ = go_bcw;
  parameters_for_go.duplicity_weight_test_ = go_duplicity_weight_test;
  parameters_for_go.duplicity_curvature_max_ = go_duplicity_max_curvature;
  parameters_for_go.use_normals_from_visible_ = use_normals_from_visible;
  parameters_for_go.weight_for_bad_normals_ = weight_for_bad_normals;

  if(parameter_selection_)
  {
      std::vector<results_and_parameters> results_and_params;

      float bcw = 0.8f; //start value
      float bcw_step = 0.05f; //step
      float bcw_end = 0.8f;
      while(bcw <= bcw_end)
      {
          parameters_for_go.best_color_weight_ = bcw;
          std::cout << "color sigma L:" << parameters_for_go.color_sigma_l_ << std::endl;
          float color_sigma_l = 0.17f; //start value
          float color_sigma_l_step = 0.01f; //step
          float color_sigma_l_end = 0.22f;
          while(color_sigma_l <= color_sigma_l_end)
          {
              parameters_for_go.color_sigma_l_ = color_sigma_l;
              std::cout << "color sigma L:" << parameters_for_go.color_sigma_l_ << std::endl;

              float color_sigma_ab = 0.2f; //start value
              float color_sigma_ab_step = 0.01f; //step
              float color_sigma_ab_end = 0.24f;
              while(color_sigma_ab <= color_sigma_ab_end)
              {
                  faat_pcl::rec_3d_framework::or_evaluator::RecognitionStatisticsResults rsr;
                  parameters_for_go.color_sigma_ab_ = color_sigma_ab;
                  std::cout << "color sigma AB:" << parameters_for_go.color_sigma_ab_ << std::endl;

                  if(source_models == 0)
                  {
                      boost::shared_ptr < faat_pcl::rec_3d_framework::ModelOnlySource<pcl::PointXYZ, pcl::PointXYZ>
                              > source (new faat_pcl::rec_3d_framework::ModelOnlySource<pcl::PointXYZ, pcl::PointXYZ>);
                      source->setPath (MODELS_DIR_);
                      source->setLoadViews (false);
                      source->setModelScale(model_scale);
                      source->setLoadIntoMemory(false);
                      std::string test = "irrelevant";
                      std::cout << "calling generate" << std::endl;
                      source->setExtension("ply");
                      source->generate (test);
                      rsr = recognizeAndVisualize<pcl::PointXYZ, pcl::PointXYZ> (source, pcd_file);
                  }
                  else if(source_models == 1)
                  {
                      boost::shared_ptr < faat_pcl::rec_3d_framework::ModelOnlySource<pcl::PointXYZRGBNormal, pcl::PointXYZRGB>
                              > source (new faat_pcl::rec_3d_framework::ModelOnlySource<pcl::PointXYZRGBNormal, pcl::PointXYZRGB>);
                      source->setPath (MODELS_DIR_);
                      source->setLoadViews (false);
                      source->setModelScale(model_scale);
                      source->setLoadIntoMemory(false);
                      std::string test = "irrelevant";
                      std::cout << "calling generate" << std::endl;
                      source->setExtension("pcd");
                      source->generate (test);
                      rsr = recognizeAndVisualize<pcl::PointXYZRGBNormal, pcl::PointXYZRGB> (source, pcd_file, "pcd");
                  }

                  results_and_parameters r_and_p;
                  r_and_p.results_ = rsr;
                  r_and_p.sigma_l_ = color_sigma_l;
                  r_and_p.sigma_ab_ = color_sigma_ab;
                  results_and_params.push_back(r_and_p);
                  color_sigma_ab += color_sigma_ab_step;

              }

              color_sigma_l += color_sigma_l_step;
          }

          bcw += bcw_step;
      }

      float best_fscore = 0;
      int best_idx = 0;
      for(size_t i=0; i < results_and_params.size(); i++)
      {
          std::cout << " precision, recall, fscore:" << results_and_params[i].results_.precision_ << ", " << results_and_params[i].results_.recall_ << ", " << results_and_params[i].results_.fscore_ << std::endl;
          std::cout << " sigma L:" << results_and_params[i].sigma_l_ << std::endl;
          std::cout << " sigma AB:" << results_and_params[i].sigma_ab_ << std::endl;

          if(results_and_params[i].results_.fscore_ > best_fscore)
          {
              best_fscore = results_and_params[i].results_.fscore_;
              best_idx = i;
          }
      }

      std::cout << "BEST:" << std::endl;
      std::cout << " precision, recall, fscore:" << results_and_params[best_idx].results_.precision_ << ", " << results_and_params[best_idx].results_.recall_ << ", " << results_and_params[best_idx].results_.fscore_ << std::endl;
      std::cout << " sigma L:" << results_and_params[best_idx].sigma_l_ << std::endl;
      std::cout << " sigma AB:" << results_and_params[best_idx].sigma_ab_ << std::endl;
  }
  else
  {
      if(source_models == 0)
      {
          boost::shared_ptr < faat_pcl::rec_3d_framework::ModelOnlySource<pcl::PointXYZ, pcl::PointXYZ>
                  > source (new faat_pcl::rec_3d_framework::ModelOnlySource<pcl::PointXYZ, pcl::PointXYZ>);
          source->setPath (MODELS_DIR_);
          source->setLoadViews (false);
          source->setModelScale(model_scale);
          source->setLoadIntoMemory(false);

          if(parameters_for_go.use_model_normals_)
          {
              source->setRadiusNormals(parameters_for_go.radius_normals_go_);
          }

          std::string test = "irrelevant";
          std::cout << "calling generate" << std::endl;
          source->setExtension("ply");
          source->generate (test);
          recognizeAndVisualize<pcl::PointXYZ, pcl::PointXYZ> (source, pcd_file);
      }
      else if(source_models == 1)
      {
          boost::shared_ptr < faat_pcl::rec_3d_framework::ModelOnlySource<pcl::PointXYZRGBNormal, pcl::PointXYZRGB>
                  > source (new faat_pcl::rec_3d_framework::ModelOnlySource<pcl::PointXYZRGBNormal, pcl::PointXYZRGB>);
          source->setPath (MODELS_DIR_);
          source->setLoadViews (false);
          source->setModelScale(model_scale);
          source->setLoadIntoMemory(false);
          std::string test = "irrelevant";
          std::cout << "calling generate" << std::endl;
          source->setExtension("pcd");
          source->generate (test);
          recognizeAndVisualize<pcl::PointXYZRGBNormal, pcl::PointXYZRGB> (source, pcd_file, "pcd");
      }
  }
}
