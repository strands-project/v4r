/*
 * local_recognition_mian_dataset.cpp
 *
 *  Created on: Mar 24, 2012
 *      Author: aitor
 */

#include <pcl/console/parse.h>
#include <faat_pcl/3d_rec_framework/pc_source/mesh_source.h>
#include <faat_pcl/3d_rec_framework/pipeline/local_recognizer.h>
#include <faat_pcl/3d_rec_framework/pipeline/hough_grouping_local_recognizer.h>
#include <faat_pcl/3d_rec_framework/pipeline/recognizer.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <faat_pcl/3d_rec_framework/feature_wrapper/local/shot_local_estimator.h>
#include <faat_pcl/3d_rec_framework/feature_wrapper/local/shot_local_estimator_omp.h>
#include <faat_pcl/3d_rec_framework/feature_wrapper/local/fpfh_local_estimator.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/recognition/cg/correspondence_grouping.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <faat_pcl/recognition/cg/graph_geometric_consistency.h>
#include <pcl/recognition/cg/hough_3d.h>
//#include <pcl/recognition/hv/hv_papazov.h>
#include <faat_pcl/recognition/hv/hv_go_1.h>
//#include <pcl/recognition/hv/greedy_verification.h>
#include <pcl/filters/passthrough.h>
#include <faat_pcl/3d_rec_framework/tools/or_evaluator.h>
#include <faat_pcl/3d_rec_framework/segmentation/multiplane_segmentation.h>

float VX_SIZE_ICP_ = 0.005f;
bool PLAY_ = false;
std::string go_log_file_ = "test.txt";
float Z_DIST_ = 1.5f;
std::string GT_DIR_;
std::string MODELS_DIR_;
std::string MODELS_DIR_FOR_VIS_;
float model_scale = 1.f;
bool SHOW_GT_ = true;
int use_hv = 1;
std::string RESULTS_OUTPUT_DIR_ = "";

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

inline bool
sortFiles (const std::string & file1, const std::string & file2)
{
  std::vector<std::string> strs1;
  boost::split (strs1, file1, boost::is_any_of ("/"));

  std::vector<std::string> strs2;
  boost::split (strs2, file2, boost::is_any_of ("/"));

  std::string id_1 = strs1[strs1.size () - 1];
  std::string id_2 = strs2[strs2.size () - 1];

  size_t pos1 = id_1.find (".ply.pcd");
  size_t pos2 = id_2.find (".ply.pcd");

  id_1 = id_1.substr (0, pos1);
  id_2 = id_2.substr (0, pos2);

  id_1 = id_1.substr (2);
  id_2 = id_2.substr (2);

  return atoi (id_1.c_str ()) < atoi (id_2.c_str ());
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

go_params parameters_for_go;

template<typename PointT>
  void
  recognizeAndVisualize (typename boost::shared_ptr<faat_pcl::rec_3d_framework::Recognizer<PointT> > & local,
                            std::string & scene_file)
  {

    faat_pcl::rec_3d_framework::or_evaluator::OREvaluator<PointT> or_eval;
    or_eval.setGTDir(GT_DIR_);
    or_eval.setModelsDir(MODELS_DIR_);
    or_eval.setCheckPose(true);
    or_eval.setScenesDir(scene_file);

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
    go->setHypPenalty(0.2f);
    go->setMinContribution(100.f);

    boost::shared_ptr<faat_pcl::HypothesisVerification<PointT, PointT> > cast_hv_alg;
    cast_hv_alg = boost::static_pointer_cast<faat_pcl::HypothesisVerification<PointT, PointT> > (go);
    //local->setHVAlgorithm (cast_hv_alg);

    typename boost::shared_ptr<faat_pcl::rec_3d_framework::Source<PointT> > model_source_ = local->getDataSource ();
    typedef typename pcl::PointCloud<PointT>::ConstPtr ConstPointInTPtr;
    typedef faat_pcl::rec_3d_framework::Model<PointT> ModelT;
    typedef boost::shared_ptr<ModelT> ModelTPtr;

    local->setVoxelSizeICP (VX_SIZE_ICP_);

    pcl::visualization::PCLVisualizer vis ("Recognition results");
    int v1, v2, v3, v4, v5;

    if(SHOW_GT_)
    {
      /*vis.createViewPort (0.0, 0.0, 0.5, 0.5, v1);
      vis.createViewPort (0.5, 0, 1, 0.5, v4);
      vis.createViewPort (0, 0.5, 1, 1.0, v2);
      vis.createViewPort (0.5, 0.5, 1.0, 1.0, v3);*/

      vis.createViewPort (0.0, 0.5, 0.33, 1.0, v1);
      vis.createViewPort (0.33, 0.5, 0.66, 1.0, v2);
      vis.createViewPort (0.0, 0.0, 0.5, 0.5, v3);
      vis.createViewPort (0.5, 0.0, 1.0, 0.5, v4);
      vis.createViewPort (0.66, 0.5, 1.0, 1.0, v5);

      vis.addText ("Ground truth", 1, 30, 18, 1, 0, 0, "gt_text", v4);
      vis.addText ("Recognition hypotheses", 1, 30, 18, 1, 0, 0, "recog_hyp_text", v2);
      vis.addText ("Recognition results", 1, 30, 18, 1, 0, 0, "recog_res_text", v3);
    }
    else
    {
      vis.createViewPort (0.0, 0.0, 0.5, 1.0, v1);
      vis.createViewPort (0.5, 0, 1.0, 1.0, v2);
      vis.addText ("Recognition results", 1, 30, 18, 1, 0, 0, "recog_res_text", v2);
    }

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
        pcl::PointCloud<pcl::PointXYZ>::Ptr scene_cloud (new pcl::PointCloud<pcl::PointXYZ>);
        std::cout << files[i] << std::endl;
        std::stringstream filestr;
        filestr << scene_file << "/" << files[i];
        std::string file = filestr.str ();
        files_to_recognize.push_back (file);
      }

      std::sort (files_to_recognize.begin (), files_to_recognize.end (), sortFiles);
      or_eval.setScenesDir(scene_file);
      or_eval.setDataSource(local->getDataSource());
      or_eval.loadGTData();
    }
    else
    {
      files_to_recognize.push_back (scene_file);
    }

    for (size_t i = 0; i < files_to_recognize.size (); i++)
    {

      std::vector<std::string> strs1;
      boost::split (strs1, files_to_recognize[i], boost::is_any_of ("/"));

      std::string id_1 = strs1[strs1.size () - 1];
      size_t pos1 = id_1.find (".pcd");

      id_1 = id_1.substr (0, pos1);

      pcl::PointCloud<pcl::PointXYZ>::Ptr scene (new pcl::PointCloud<pcl::PointXYZ>);
      pcl::io::loadPCDFile (files_to_recognize[i], *scene);

      if (Z_DIST_ > 0)
      {
        pcl::PassThrough<PointT> pass_;
        pass_.setFilterLimits (0.f, Z_DIST_);
        pass_.setFilterFieldName ("z");
        pass_.setInputCloud (scene);
        pass_.setKeepOrganized (true);
        pass_.filter (*scene);
      }

      pcl::visualization::PointCloudColorHandlerGenericField<PointT> scene_handler(scene, "z");
      vis.addPointCloud<PointT> (scene, scene_handler, "scene_cloud", v1);

      //Multiplane segmentation
      faat_pcl::MultiPlaneSegmentation<PointT> mps;
      mps.setInputCloud(scene);
      mps.setMinPlaneInliers(1000);
      mps.setResolution(parameters_for_go.go_resolution);
      mps.segment();
      std::vector<faat_pcl::PlaneModel<PointT> > planes_found;
      planes_found = mps.getModels();

      for(size_t kk=0; kk < planes_found.size(); kk++)
      {
        std::stringstream pname;
        pname << "plane_" << kk;

        pcl::visualization::PointCloudColorHandlerRandom<pcl::PointXYZ> scene_handler(planes_found[kk].plane_cloud_);
        vis.addPointCloud<pcl::PointXYZ> (planes_found[kk].plane_cloud_, scene_handler, pname.str(), v2);

        pname << "chull";
        vis.addPolygonMesh (*planes_found[kk].convex_hull_, pname.str(), v2);
      }

      local->setInputCloud (scene);
      {
        pcl::ScopeTime ttt ("Recognition");
        local->recognize ();
      }

      std::vector < std::string > strs;
      boost::split (strs, files_to_recognize[i], boost::is_any_of ("/"));
      vis.addText (strs[strs.size() - 1], 1, 30, 18, 1, 0, 0, "scene_text", v1);

      boost::shared_ptr<std::vector<ModelTPtr> > models = local->getModels ();
      boost::shared_ptr<std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > > transforms = local->getTransforms ();

      if(use_hv)
      {
          //visualize results

          std::vector<typename pcl::PointCloud<PointT>::ConstPtr> aligned_models;
          aligned_models.resize (models->size ());

          for (size_t kk = 0; kk < models->size (); kk++)
          {
           ConstPointInTPtr model_cloud = models->at (kk)->getAssembled (parameters_for_go.go_resolution);
           typename pcl::PointCloud<PointT>::Ptr model_aligned (new pcl::PointCloud<PointT>);
           pcl::transformPointCloud (*model_cloud, *model_aligned, transforms->at (kk));
           aligned_models[kk] = model_aligned;
          }

          go->setSceneCloud (scene);
          //addModels
          go->addModels (aligned_models, true);
          //append planar models
          go->addPlanarModels(planes_found);
          //verify
          go->verify ();
          std::vector<bool> mask_hv;
          go->getMask (mask_hv);

          std::vector<int> coming_from;
          coming_from.resize(aligned_models.size() + planes_found.size());
          for(size_t j=0; j < aligned_models.size(); j++)
           coming_from[j] = 0;

          for(size_t j=0; j < planes_found.size(); j++)
           coming_from[aligned_models.size() + j] = 1;

          if(SHOW_GT_)
          {
            pcl::visualization::PointCloudColorHandlerCustom<PointT> scene_handler(scene, 125,125,125);
            vis.addPointCloud<PointT> (scene, scene_handler, "scene_cloud_v4", v4);
            or_eval.visualizeGroundTruth(vis, id_1, v4, false);
          }

          pcl::PointCloud<pcl::PointXYZRGBA>::Ptr smooth_cloud_ =  go->getSmoothClustersRGBCloud();
          if(smooth_cloud_)
          {
            pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> random_handler (smooth_cloud_);
            vis.addPointCloud<pcl::PointXYZRGBA> (smooth_cloud_, random_handler, "smooth_cloud", v5);
          }

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

              float r, g, b;
              std::cout << models->at (j)->id_ << std::endl;
              r = 255.0f;
              g = 0.0f;
              b = 0.0f;

              if (models->at (j)->id_.compare ("Zoe") == 0)
              {
                r = 0.0f;
                g = 255.0f;
                b = 0.0f;
              }
              else if (models->at (j)->id_.compare ("Kid") == 0)
              {
                r = 0.0f;
                g = 255.0f;
                b = 255.0f;
              }
              else if (models->at (j)->id_.compare ("BigBird") == 0)
              {
                r = 255.0f;
                g = 255.0f;
                b = 0.f;
              }
              else if (models->at (j)->id_.compare ("Angel") == 0)
              {
                r = 0.0f;
                g = 122.5f;
                b = 122.5f;
              }
              else if (models->at (j)->id_.compare ("chef") == 0)
              {
                r = 0.0f;
                g = 255.0f;
                b = 0.0f;
              }
              else if (models->at (j)->id_.compare ("chicken") == 0)
              {
                r = 0.0f;
                g = 255.0f;
                b = 255.0f;
              }
              else if (models->at (j)->id_.compare ("para") == 0)
              {
                r = 255.0f;
                g = 255.0f;
                b = 0.f;
              }
              else if (models->at (j)->id_.compare ("rhino") == 0)
              {
                r = 255.0f;
                g = 105.0f;
                b = 180.f;
              }

              /*pcl::visualization::PointCloudColorHandlerCustom<PointT> random_handler (model_aligned, r, g, b);
              vis.addPointCloud<PointT> (model_aligned, random_handler, name.str (), v3);*/

              std::stringstream pathPly;
              pathPly << MODELS_DIR_FOR_VIS_ << "/" << models->at (j)->id_ << ".ply";
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
            else
            {
              std::stringstream pname;
              pname << "plane_" << j;

              /*pcl::visualization::PointCloudColorHandlerRandom<pcl::PointXYZ> scene_handler(planes_found[j - models->size()].plane_cloud_);
              vis.addPointCloud<pcl::PointXYZ> (planes_found[j - models->size()].plane_cloud_, scene_handler, pname.str(), v3);*/

              pname << "chull";
              vis.addPolygonMesh (*planes_found[j - models->size()].convex_hull_, pname.str(), v3);
            }
          }

          or_eval.addRecognitionResults(id_1, verified_models, verified_transforms);

          //log results into file
          logfile_stream << files_to_recognize[i] << "\t";
          go->writeToLog (logfile_stream, false);

      }
      else
      {
          or_eval.addRecognitionResults(id_1, models, transforms);
      }


      for (size_t j = 0; j < models->size (); j++)
      {
        std::stringstream name;
        name << "cloud_before_hv_" << j;

        ConstPointInTPtr model_cloud = models->at (j)->getAssembled (0.003f);
        typename pcl::PointCloud<PointT>::Ptr model_aligned (new pcl::PointCloud<PointT>);
        pcl::transformPointCloud (*model_cloud, *model_aligned, transforms->at (j));

        float r, g, b;
        r = 255.0f;
        g = 0.0f;
        b = 0.0f;

        if (models->at (j)->id_.compare ("Zoe") == 0)
        {
          r = 0.0f;
          g = 255.0f;
          b = 0.0f;
        }
        else if (models->at (j)->id_.compare ("Kid") == 0)
        {
          r = 0.0f;
          g = 255.0f;
          b = 255.0f;
        }
        else if (models->at (j)->id_.compare ("BigBird") == 0)
        {
          r = 255.0f;
          g = 255.0f;
          b = 0.f;
        }
        else if (models->at (j)->id_.compare ("Angel") == 0)
        {
          r = 0.0f;
          g = 122.5f;
          b = 122.5f;
        }
        else if (models->at (j)->id_.compare ("chef") == 0)
        {
          r = 0.0f;
          g = 255.0f;
          b = 0.0f;
        }
        else if (models->at (j)->id_.compare ("chicken") == 0)
        {
          r = 0.0f;
          g = 255.0f;
          b = 255.0f;
        }
        else if (models->at (j)->id_.compare ("para") == 0)
        {
          r = 255.0f;
          g = 255.0f;
          b = 0.f;
        }
        else if (models->at (j)->id_.compare ("rhino") == 0)
        {
          r = 255.0f;
          g = 105.0f;
          b = 180.f;
        }

        pcl::visualization::PointCloudColorHandlerCustom<PointT> random_handler (model_aligned, r, g, b);
        vis.addPointCloud<PointT> (model_aligned, random_handler, name.str (), v2);
      }

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
    }

    or_eval.computeStatistics();

    if(RESULTS_OUTPUT_DIR_.compare("") != 0)
        or_eval.saveRecognitionResults(RESULTS_OUTPUT_DIR_);

    logfile_stream.close ();
  }

typedef pcl::ReferenceFrame RFType;

int CG_SIZE_ = 3;
float CG_THRESHOLD_ = 0.005f;

/*
Mian
-----
./bin/local_recognition_ply -models_dir /home/aitor/data/Mians_dataset/models_with_rhino/ -pcd_file /home/aitor/data/Mians_dataset/scenes/hard_scenes/ -training_dir /home/aitor/data/Mians_dataset/Mians_trained_models_voxelsize_0.003/ -gc_size 5 -icp_type 1 -use_hv 1 -icp_iterations 50 -use_cache 1 -splits 512 -hv_method 0 -model_scale 0.001 -go_opt_type 0 -thres_hyp 0.2 -go_iterations 10000 -go_resolution 0.003 -go_inlier_thres 0.008 -go_initial_temp 500 -idx_flann_fn mian_flann_with_rhino_new.idx -PLAY 0 -vx_size_icp 0.005 -go_require_normals 0 -go_init 0 -go_log_file test.txt -use_hough 0 -gc_threshold 0.01 -gc_ransac_threshold 0.01 -test_sampling_density 0.005 -force_retrain 0 -use_gc_graph 1 -desc_radius 0.04 -GT_DIR /home/aitor/data/Mians_dataset/gt_or_format_rhino_with_occ -models_dir_vis /home/aitor/data/Mians_dataset/models_with_rhino_vis/ -gc_min_dist_cf 0.5 -use_board 0 -rf_radius_hough 0.04 -gc_dot_threshold 0.25 -visualize_graph 0 -LS_short_circuit 1 -go_use_replace_moves 0
./bin/local_recognition_ply -models_dir /home/aitor/data/Mians_dataset/models/ -pcd_file /home/aitor/data/Mians_dataset/scenes/hard_scenes/ -training_dir /home/aitor/data/Mians_dataset/Mians_trained_models_voxelsize_0.003/ -gc_size 3 -icp_type 1 -use_hv 1 -icp_iterations 50 -use_cache 1 -splits 512 -hv_method 0 -model_scale 0.001 -go_opt_type 0 -thres_hyp 0.2 -go_iterations 10000 -go_resolution 0.0025 -go_inlier_thres 0.005 -go_initial_temp 500 -idx_flann_fn mian_flann_without_rhino.idx -PLAY 0 -vx_size_icp 0.005 -go_require_normals 0 -go_init 0 -go_log_file test.txt -use_hough 0 -gc_threshold 0.01 -gc_ransac_threshold 0.01 -test_sampling_density 0.005 -force_retrain 0 -use_gc_graph 1 -desc_radius 0.04 -GT_DIR /home/aitor/data/Mians_dataset/gt_or_format_rhino_with_occ -models_dir_vis /home/aitor/data/Mians_dataset/models_with_rhino_vis/ -gc_min_dist_cf 0.5 -use_board 0 -rf_radius_hough 0.04 -gc_dot_threshold 0.25 -visualize_graph 0 -LS_short_circuit 0 -go_use_replace_moves 1 (works on the whole dataset)
./bin/local_recognition_ply -models_dir /home/aitor/data/Mians_dataset/models_with_rhino/ -pcd_file /home/aitor/data/Mians_dataset/scenes/hard_scenes/ -training_dir /home/aitor/data/Mians_dataset/Mians_trained_models_voxelsize_0.003/ -gc_size 3 -icp_type 1 -use_hv 1 -icp_iterations 50 -use_cache 1 -splits 512 -hv_method 0 -model_scale 0.001 -go_opt_type 0 -thres_hyp 0.2 -go_iterations 10000 -go_resolution 0.0025 -go_inlier_thres 0.005 -go_initial_temp 500 -idx_flann_fn mian_flann_with_rhino_new.idx -PLAY 0 -vx_size_icp 0.005 -go_require_normals 0 -go_init 0 -go_log_file test.txt -use_hough 0 -gc_threshold 0.01 -gc_ransac_threshold 0.01 -test_sampling_density 0.005 -force_retrain 0 -use_gc_graph 1 -desc_radius 0.04 -GT_DIR /home/aitor/data/Mians_dataset/gt_or_format_rhino_with_occ -models_dir_vis /home/aitor/data/Mians_dataset/models_with_rhino_vis/ -gc_min_dist_cf 0.5 -use_board 0 -rf_radius_hough 0.04 -gc_dot_threshold 0.25 -visualize_graph 0 -LS_short_circuit 0 -go_use_replace_moves 1

Queens
-----
./bin/local_recognition_ply -models_dir /home/aitor/data/queens_dataset/pcd_models/ -pcd_file /home/aitor/data/queens_dataset/hard_scenes -training_dir /home/aitor/data/queens_dataset/trained_models -gc_size 3 -icp_type 1 -use_hv 1 -icp_iterations 10 -use_cache 1 -splits 32 -hv_method 0 -model_scale 1 -go_opt_type 0 -thres_hyp 0.2 -go_iterations 5000 -go_use_replace_moves 1 -go_resolution 0.005 -go_inlier_thres 0.01 -go_initial_temp 500 -idx_flann_fn queens_flann.idx -PLAY 0 -vx_size_icp 0.005 -go_require_normals 0 -go_init 0 -go_log_file test.txt -use_hough 0 -gc_threshold 0.01 -gc_ransac_threshold 0.01 -test_sampling_density 0.005 -force_retrain 0 -use_gc_graph 1 -desc_radius 0.04 -GT_DIR /home/aitor/data/queens_dataset/gt_or_format_all_with_occ/ -models_dir_vis /home/aitor/data/queens_dataset/models_for_visualization/ -gc_min_dist_cf 0 -gc_dot_threshold 0.5 -vis_cues_ 1

*/
int
main (int argc, char ** argv)
{
  std::string path = "";
  std::string desc_name = "shot_omp";
  std::string training_dir = "trained_models/";
  std::string pcd_file = "";
  int force_retrain = 0;
  int icp_iterations = 20;
  int use_cache = 1;
  int splits = 512;
  int scene = -1;
  int detect_clutter = 1;
  int hv_method = 0;
  float thres_hyp_ = 0.2f;
  float desc_radius = 0.04f;
  int icp_type = 0;
  int go_opt_type = 2;
  int go_iterations = 7000;
  bool go_use_replace_moves = true;
  float go_inlier_thres = 0.01f;
  float go_resolution = 0.005f;
  float init_temp = 1000.f;
  std::string idx_flann_fn;
  float radius_normals_go_ = 0.015f;
  bool go_require_normals = false;
  bool go_log = true;
  bool go_init = false;
  float test_sampling_density = 0.005f;
  int tes_level_ = 1;
  bool use_hough_ = false;
  float gc_ransac_threshold;
  bool use_gc_graph = true;
  float min_dist_cf_ = 1.f;
  float gc_dot_threshold_ = 1.f;
  bool use_board = false;
  float rf_radius_hough = 0.04f;
  bool visualize_graph = false;
  bool LS_short_circuit_ = false;
  int vis_cues_ = 0;

  pcl::console::parse_argument (argc, argv, "-vis_cues_", vis_cues_);
  pcl::console::parse_argument (argc, argv, "-LS_short_circuit", LS_short_circuit_);
  pcl::console::parse_argument (argc, argv, "-visualize_graph", visualize_graph);
  pcl::console::parse_argument (argc, argv, "-use_gc_graph", use_gc_graph);
  pcl::console::parse_argument (argc, argv, "-rf_radius_hough", rf_radius_hough);
  pcl::console::parse_argument (argc, argv, "-use_board", use_board);
  pcl::console::parse_argument (argc, argv, "-models_dir", path);
  pcl::console::parse_argument (argc, argv, "-training_dir", training_dir);
  pcl::console::parse_argument (argc, argv, "-descriptor_name", desc_name);
  pcl::console::parse_argument (argc, argv, "-pcd_file", pcd_file);
  pcl::console::parse_argument (argc, argv, "-force_retrain", force_retrain);
  pcl::console::parse_argument (argc, argv, "-icp_iterations", icp_iterations);
  pcl::console::parse_argument (argc, argv, "-use_cache", use_cache);
  pcl::console::parse_argument (argc, argv, "-splits", splits);
  pcl::console::parse_argument (argc, argv, "-gc_size", CG_SIZE_);
  pcl::console::parse_argument (argc, argv, "-gc_threshold", CG_THRESHOLD_);
  pcl::console::parse_argument (argc, argv, "-scene", scene);
  pcl::console::parse_argument (argc, argv, "-detect_clutter", detect_clutter);
  pcl::console::parse_argument (argc, argv, "-hv_method", hv_method);
  pcl::console::parse_argument (argc, argv, "-use_hv", use_hv);
  pcl::console::parse_argument (argc, argv, "-thres_hyp", thres_hyp_);
  pcl::console::parse_argument (argc, argv, "-icp_type", icp_type);
  pcl::console::parse_argument (argc, argv, "-vx_size_icp", VX_SIZE_ICP_);
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
  pcl::console::parse_argument (argc, argv, "-idx_flann_fn", idx_flann_fn);
  pcl::console::parse_argument (argc, argv, "-PLAY", PLAY_);
  pcl::console::parse_argument (argc, argv, "-go_log_file", go_log_file_);
  pcl::console::parse_argument (argc, argv, "-test_sampling_density", test_sampling_density);
  pcl::console::parse_argument (argc, argv, "-tes_level", tes_level_);
  pcl::console::parse_argument (argc, argv, "-Z_DIST", Z_DIST_);
  pcl::console::parse_argument (argc, argv, "-use_hough", use_hough_);
  pcl::console::parse_argument (argc, argv, "-gc_ransac_threshold", gc_ransac_threshold);
  pcl::console::parse_argument (argc, argv, "-desc_radius", desc_radius);
  pcl::console::parse_argument (argc, argv, "-show_gt", SHOW_GT_);
  pcl::console::parse_argument (argc, argv, "-gc_min_dist_cf", min_dist_cf_);
  pcl::console::parse_argument (argc, argv, "-gc_dot_threshold", gc_dot_threshold_);

  MODELS_DIR_FOR_VIS_ = path;

  pcl::console::parse_argument (argc, argv, "-models_dir_vis", MODELS_DIR_FOR_VIS_);
  pcl::console::parse_argument (argc, argv, "-GT_DIR", GT_DIR_);
  pcl::console::parse_argument (argc, argv, "-output_dir_before_hv", RESULTS_OUTPUT_DIR_);
  MODELS_DIR_ = path;

  std::cout << "VX_SIZE_ICP_" << VX_SIZE_ICP_ << std::endl;
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

  bf::path models_dir_path = path;
  if (!bf::exists (models_dir_path))
  {
    PCL_ERROR("Models dir path %s does not exist, use -models_dir [dir] option\n", path.c_str());
    return -1;
  }
  else
  {
    std::vector<std::string> files;
    std::string start = "";
    std::string ext = std::string ("ply");
    bf::path dir = models_dir_path;
    getModelsInDirectory (dir, start, files, ext);
    std::cout << "Number of models in directory is:" << files.size () << std::endl;
  }

  parameters_for_go.radius_normals_go_ = radius_normals_go_;
  parameters_for_go.radius_clutter = 0.03f;
  parameters_for_go.clutter_regularizer = 10.f;
  parameters_for_go.regularizer = 5.f;
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
  //configure mesh source
  boost::shared_ptr<faat_pcl::rec_3d_framework::MeshSource<pcl::PointXYZ> > mesh_source (new faat_pcl::rec_3d_framework::MeshSource<pcl::PointXYZ>);
  mesh_source->setPath (path);
  mesh_source->setResolution (250);
  mesh_source->setTesselationLevel (tes_level_);
  mesh_source->setViewAngle (57.f);
  mesh_source->setRadiusSphere (1.5f);
  mesh_source->setModelScale (model_scale);
  //mesh_source->setRadiusNormals (radius_normals_go_);
  mesh_source->generate (training_dir);

  boost::shared_ptr<faat_pcl::rec_3d_framework::Source<pcl::PointXYZ> > cast_source;
  cast_source = boost::static_pointer_cast<faat_pcl::rec_3d_framework::MeshSource<pcl::PointXYZ> > (mesh_source);

  //configure normal estimator
  boost::shared_ptr<faat_pcl::rec_3d_framework::PreProcessorAndNormalEstimator<pcl::PointXYZ, pcl::Normal> > normal_estimator;
  normal_estimator.reset (new faat_pcl::rec_3d_framework::PreProcessorAndNormalEstimator<pcl::PointXYZ, pcl::Normal>);
  normal_estimator->setCMR (false);
  normal_estimator->setDoVoxelGrid (true);
  normal_estimator->setRemoveOutliers (true);
  normal_estimator->setValuesForCMRFalse (0.003f, 0.012f);

  //configure keypoint extractor
  boost::shared_ptr<faat_pcl::rec_3d_framework::UniformSamplingExtractor<pcl::PointXYZ> >
                                                                                          uniform_keypoint_extractor (
                                                                                                                      new faat_pcl::rec_3d_framework::UniformSamplingExtractor<
                                                                                                                          pcl::PointXYZ>);

  uniform_keypoint_extractor->setSamplingDensity (0.01f);
  //uniform_keypoint_extractor->setSamplingDensity (0.005f);
  uniform_keypoint_extractor->setFilterPlanar (true);

  boost::shared_ptr<faat_pcl::rec_3d_framework::KeypointExtractor<pcl::PointXYZ> > keypoint_extractor;
  keypoint_extractor = boost::static_pointer_cast<faat_pcl::rec_3d_framework::KeypointExtractor<pcl::PointXYZ> > (uniform_keypoint_extractor);

  //configure cg algorithm (geometric consistency grouping)
  boost::shared_ptr<pcl::CorrespondenceGrouping<pcl::PointXYZ, pcl::PointXYZ> > cast_cg_alg;

  {
    boost::shared_ptr<pcl::GeometricConsistencyGrouping<pcl::PointXYZ, pcl::PointXYZ> > gcg_alg (
                                                                                                 new pcl::GeometricConsistencyGrouping<pcl::PointXYZ,
                                                                                                     pcl::PointXYZ>);
    gcg_alg->setGCThreshold (CG_SIZE_);
    gcg_alg->setGCSize (CG_THRESHOLD_);
    //gcg_alg->setRansacThreshold (gc_ransac_threshold);

    cast_cg_alg = boost::static_pointer_cast<pcl::CorrespondenceGrouping<pcl::PointXYZ, pcl::PointXYZ> > (gcg_alg);
  }

  {
    boost::shared_ptr<faat_pcl::GraphGeometricConsistencyGrouping<pcl::PointXYZ, pcl::PointXYZ> > gcg_alg (
                                                                                                 new faat_pcl::GraphGeometricConsistencyGrouping<pcl::PointXYZ,
                                                                                                     pcl::PointXYZ>);
    gcg_alg->setGCThreshold (CG_SIZE_);
    gcg_alg->setGCSize (CG_THRESHOLD_);
    gcg_alg->setRansacThreshold (gc_ransac_threshold);
    gcg_alg->setUseGraph(use_gc_graph);
    gcg_alg->setDistForClusterFactor(min_dist_cf_);
    gcg_alg->setPrune(false);
    gcg_alg->setVisualizeGraph(visualize_graph);
    gcg_alg->setDotDistance(gc_dot_threshold_);
    gcg_alg->setMaxTimeForCliquesComputation(100);
    cast_cg_alg = boost::static_pointer_cast<pcl::CorrespondenceGrouping<pcl::PointXYZ, pcl::PointXYZ> > (gcg_alg);
  }

  boost::shared_ptr<pcl::Hough3DGrouping<pcl::PointXYZ, pcl::PointXYZ, pcl::ReferenceFrame, pcl::ReferenceFrame> >
                                                                                                                   hough_3d_voting_cg_alg (
                                                                                                                                           new pcl::Hough3DGrouping<
                                                                                                                                               pcl::PointXYZ,
                                                                                                                                               pcl::PointXYZ,
                                                                                                                                               pcl::ReferenceFrame,
                                                                                                                                               pcl::ReferenceFrame>);

  hough_3d_voting_cg_alg->setHoughBinSize (CG_THRESHOLD_);
  hough_3d_voting_cg_alg->setHoughThreshold (CG_SIZE_);
  hough_3d_voting_cg_alg->setUseInterpolation (true);
  hough_3d_voting_cg_alg->setUseDistanceWeight (false);
  //hough_3d_voting_cg_alg->setRansacThreshold (gc_ransac_threshold);

  //configure hypothesis verificator
  /*boost::shared_ptr<pcl::PapazovHV<pcl::PointXYZ, pcl::PointXYZ> > papazov (new pcl::PapazovHV<pcl::PointXYZ, pcl::PointXYZ>);
   papazov->setResolution (0.005f);
   papazov->setInlierThreshold (0.005f);
   papazov->setSupportThreshold (0.08f);
   papazov->setPenaltyThreshold (0.05f);
   papazov->setConflictThreshold (0.02f);
   papazov->setOcclusionThreshold (0.01f);*/

  /*boost::shared_ptr<pcl::GreedyVerification<pcl::PointXYZ, pcl::PointXYZ> > greedy (new pcl::GreedyVerification<pcl::PointXYZ, pcl::PointXYZ> (2.f));
   greedy->setResolution (0.005f);
   greedy->setInlierThreshold (0.01f);
   greedy->setOcclusionThreshold (0.01f);

   boost::shared_ptr<pcl::HypothesisVerification<pcl::PointXYZ, pcl::PointXYZ> > cast_hv_alg;

   switch (hv_method)
   {
   case 1:
   cast_hv_alg = boost::static_pointer_cast<pcl::HypothesisVerification<pcl::PointXYZ, pcl::PointXYZ> > (greedy);
   break;
   case 2:
   cast_hv_alg = boost::static_pointer_cast<pcl::HypothesisVerification<pcl::PointXYZ, pcl::PointXYZ> > (papazov);
   break;
   default:
   cast_hv_alg = boost::static_pointer_cast<pcl::HypothesisVerification<pcl::PointXYZ, pcl::PointXYZ> > (go);
   }*/

#ifdef _MSC_VER
  _CrtCheckMemory();
#endif

  boost::shared_ptr<faat_pcl::rec_3d_framework::Recognizer<pcl::PointXYZ> > cast_recog;

  if (desc_name.compare ("shot_omp") == 0)
  {
    desc_name = std::string ("shot");
    boost::shared_ptr<faat_pcl::rec_3d_framework::SHOTLocalEstimationOMP<pcl::PointXYZ, pcl::Histogram<352> > > estimator;
    estimator.reset (new faat_pcl::rec_3d_framework::SHOTLocalEstimationOMP<pcl::PointXYZ, pcl::Histogram<352> >);
    estimator->setNormalEstimator (normal_estimator);
    estimator->addKeypointExtractor (keypoint_extractor);
    estimator->setSupportRadius (desc_radius);

    boost::shared_ptr<faat_pcl::rec_3d_framework::LocalEstimator<pcl::PointXYZ, pcl::Histogram<352> > > cast_estimator;
    cast_estimator = boost::dynamic_pointer_cast<faat_pcl::rec_3d_framework::LocalEstimator<pcl::PointXYZ, pcl::Histogram<352> > > (estimator);

#ifdef _MSC_VER
    _CrtCheckMemory();
#endif

    if (use_hough_)
    {
      cast_cg_alg = boost::static_pointer_cast<pcl::Hough3DGrouping<pcl::PointXYZ, pcl::PointXYZ> > (hough_3d_voting_cg_alg);
      boost::shared_ptr<faat_pcl::rec_3d_framework::LocalRecognitionHoughGroupingPipeline<flann::L1, pcl::PointXYZ, pcl::Histogram<352> > > local;
      local.reset (new faat_pcl::rec_3d_framework::LocalRecognitionHoughGroupingPipeline<flann::L1, pcl::PointXYZ, pcl::Histogram<352> > (idx_flann_fn));
      local->setDataSource (cast_source);
      local->setTrainingDir (training_dir);
      local->setDescriptorName (desc_name);
      local->setFeatureEstimator (cast_estimator);
      local->setCGAlgorithm (cast_cg_alg);
      local->setUseBoard(use_board);
      local->setRFRadius(rf_radius_hough);
      //if (use_hv)
      //local.setHVAlgorithm (cast_hv_alg);

      local->setUseCache (static_cast<bool> (use_cache));
      local->setVoxelSizeICP (VX_SIZE_ICP_);

      local->initialize (static_cast<bool> (force_retrain));
      local->setThresholdAcceptHyp (thres_hyp_);

      uniform_keypoint_extractor->setSamplingDensity (test_sampling_density);
      local->setICPIterations (icp_iterations);
      local->setKdtreeSplits (splits);
      local->setICPType (icp_type);

      cast_recog
          = boost::static_pointer_cast<faat_pcl::rec_3d_framework::LocalRecognitionHoughGroupingPipeline<flann::L1, pcl::PointXYZ, pcl::Histogram<352> > > (local);
    }
    else
    {
      boost::shared_ptr<faat_pcl::rec_3d_framework::LocalRecognitionPipeline<flann::L1, pcl::PointXYZ, pcl::Histogram<352> > > local;
      local.reset (new faat_pcl::rec_3d_framework::LocalRecognitionPipeline<flann::L1, pcl::PointXYZ, pcl::Histogram<352> > (idx_flann_fn));
      local->setDataSource (cast_source);
      local->setTrainingDir (training_dir);
      local->setDescriptorName (desc_name);
      local->setFeatureEstimator (cast_estimator);
      local->setCGAlgorithm (cast_cg_alg);

      //if (use_hv)
      //local.setHVAlgorithm (cast_hv_alg);

      local->setUseCache (static_cast<bool> (use_cache));
      local->setVoxelSizeICP (VX_SIZE_ICP_);

      local->initialize (static_cast<bool> (force_retrain));
      local->setThresholdAcceptHyp (thres_hyp_);

      uniform_keypoint_extractor->setSamplingDensity (test_sampling_density);
      local->setICPIterations (icp_iterations);
      local->setKdtreeSplits (splits);
      local->setICPType (icp_type);

      cast_recog
          = boost::static_pointer_cast<faat_pcl::rec_3d_framework::LocalRecognitionPipeline<flann::L1, pcl::PointXYZ, pcl::Histogram<352> > > (local);
    }

#ifdef _MSC_VER
    _CrtCheckMemory();
#endif
  }

  recognizeAndVisualize<pcl::PointXYZ> (cast_recog, pcd_file);
}
