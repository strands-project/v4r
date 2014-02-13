/*
 * local_recognition_mian_dataset.cpp
 *
 *  Created on: Mar 24, 2012
 *      Author: aitor
 */
#include <pcl/console/parse.h>
#include <faat_pcl/3d_rec_framework/pc_source/mesh_source.h>
#include <faat_pcl/3d_rec_framework/pipeline/hough_grouping_local_recognizer.h>
#include <faat_pcl/3d_rec_framework/pipeline/local_recognizer.h>
#include <faat_pcl/3d_rec_framework/pipeline/global_nn_recognizer_cvfh.h>
#include <faat_pcl/3d_rec_framework/pipeline/recognizer.h>
#include <faat_pcl/3d_rec_framework/pipeline/multi_pipeline_recognizer.h>
#include <faat_pcl/3d_rec_framework/utils/metrics.h>
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
//#include <faat_pcl/recognition/hv/hv_go.h>
#include <faat_pcl/recognition/hv/hv_go_1.h>
//#include <pcl/recognition/hv/greedy_verification.h>
#include <pcl/filters/passthrough.h>

#include <pcl/segmentation/organized_multi_plane_segmentation.h>
#include <pcl/segmentation/planar_polygon_fusion.h>
#include <pcl/segmentation/plane_coefficient_comparator.h>
#include <pcl/segmentation/euclidean_plane_coefficient_comparator.h>
#include <pcl/segmentation/rgb_plane_coefficient_comparator.h>
#include <pcl/segmentation/edge_aware_plane_comparator.h>
#include <pcl/segmentation/euclidean_cluster_comparator.h>
#include <pcl/segmentation/organized_connected_component_segmentation.h>
#include <faat_pcl/3d_rec_framework/tools/or_evaluator.h>
#include <pcl/filters/fast_bilateral.h>
#include <faat_pcl/3d_rec_framework/segmentation/multiplane_segmentation.h>
#include <pcl/apps/dominant_plane_segmentation.h>

float VX_SIZE_ICP_ = 0.005f;
bool PLAY_ = false;
std::string go_log_file_ = "test.txt";
float Z_DIST_ = 1.5f;
std::string GT_DIR_;
std::string MODELS_DIR_;
std::string MODELS_DIR_FOR_VIS_;
float model_scale = 1.f;
bool use_HV = true;

template<typename PointT>
void
doSegmentation (typename pcl::PointCloud<PointT>::Ptr & xyz_points,
                  std::vector<pcl::PointIndices> & indices,
                  Eigen::Vector4f & table_plane,
                  int seg = 0)
{
  std::cout << "Start segmentation..." << std::endl;

  typename pcl::PointCloud<PointT>::Ptr xyz_points_andy (new pcl::PointCloud<PointT>);
  pcl::PassThrough<PointT> pass_;
  pass_.setFilterLimits (0.f, Z_DIST_);
  pass_.setFilterFieldName ("z");
  pass_.setInputCloud (xyz_points);
  pass_.setKeepOrganized (true);
  pass_.filter (*xyz_points_andy);
  int min_cluster_size_ = 500;

  if(seg == 0)
  {
    pcl::IntegralImageNormalEstimation<PointT, pcl::Normal> ne;
    ne.setNormalEstimationMethod (ne.COVARIANCE_MATRIX);
    ne.setMaxDepthChangeFactor (0.02f);
    ne.setNormalSmoothingSize (20.0f);
    ne.setBorderPolicy (pcl::IntegralImageNormalEstimation<PointT, pcl::Normal>::BORDER_POLICY_IGNORE);
    ne.setInputCloud (xyz_points);
    pcl::PointCloud<pcl::Normal>::Ptr normal_cloud (new pcl::PointCloud<pcl::Normal>);
    ne.compute (*normal_cloud);

    int num_plane_inliers = 2500;

    pcl::OrganizedMultiPlaneSegmentation<PointT, pcl::Normal, pcl::Label> mps;
    mps.setMinInliers (num_plane_inliers);
    mps.setAngularThreshold (0.017453 * 5.f); // 2 degrees
    mps.setDistanceThreshold (0.01); // 1cm
    mps.setInputNormals (normal_cloud);
    mps.setInputCloud (xyz_points_andy);

    std::vector<pcl::PlanarRegion<PointT>, Eigen::aligned_allocator<pcl::PlanarRegion<PointT> > > regions;
    std::vector<pcl::ModelCoefficients> model_coefficients;
    std::vector<pcl::PointIndices> inlier_indices;
    pcl::PointCloud<pcl::Label>::Ptr labels (new pcl::PointCloud<pcl::Label>);
    std::vector<pcl::PointIndices> label_indices;
    std::vector<pcl::PointIndices> boundary_indices;

    typename pcl::PlaneRefinementComparator<PointT, pcl::Normal, pcl::Label>::Ptr ref_comp (
                                                                                             new pcl::PlaneRefinementComparator<PointT,
                                                                                                 pcl::Normal, pcl::Label> ());
    ref_comp->setDistanceThreshold (0.01f, true);
    ref_comp->setAngularThreshold (0.017453 * 10);
    mps.setRefinementComparator (ref_comp);
    mps.segmentAndRefine (regions, model_coefficients, inlier_indices, labels, label_indices, boundary_indices);

    std::cout << "Number of planes found:" << model_coefficients.size () << std::endl;

    int table_plane_selected = 0;
    int max_inliers_found = -1;
    std::vector<int> plane_inliers_counts;
    plane_inliers_counts.resize (model_coefficients.size ());

    for (size_t i = 0; i < model_coefficients.size (); i++)
    {
      Eigen::Vector4f table_plane = Eigen::Vector4f (model_coefficients[i].values[0], model_coefficients[i].values[1],
                                                     model_coefficients[i].values[2], model_coefficients[i].values[3]);

      std::cout << "Number of inliers for this plane:" << inlier_indices[i].indices.size () << std::endl;
      int remaining_points = 0;
      typename pcl::PointCloud<PointT>::Ptr plane_points (new pcl::PointCloud<PointT> (*xyz_points_andy));
      for (int j = 0; j < plane_points->points.size (); j++)
      {
        Eigen::Vector3f xyz_p = plane_points->points[j].getVector3fMap ();

        if (!pcl_isfinite (xyz_p[0]) || !pcl_isfinite (xyz_p[1]) || !pcl_isfinite (xyz_p[2]))
          continue;

        float val = xyz_p[0] * table_plane[0] + xyz_p[1] * table_plane[1] + xyz_p[2] * table_plane[2] + table_plane[3];

        if (std::abs (val) > 0.01)
        {
          plane_points->points[j].x = std::numeric_limits<float>::quiet_NaN ();
          plane_points->points[j].y = std::numeric_limits<float>::quiet_NaN ();
          plane_points->points[j].z = std::numeric_limits<float>::quiet_NaN ();
        }
        else
          remaining_points++;
      }

      plane_inliers_counts[i] = remaining_points;

      if (remaining_points > max_inliers_found)
      {
        table_plane_selected = i;
        max_inliers_found = remaining_points;
      }
    }

    size_t itt = static_cast<size_t> (table_plane_selected);
    table_plane = Eigen::Vector4f (model_coefficients[itt].values[0], model_coefficients[itt].values[1],
                                                   model_coefficients[itt].values[2], model_coefficients[itt].values[3]);

    Eigen::Vector3f normal_table = Eigen::Vector3f (model_coefficients[itt].values[0], model_coefficients[itt].values[1],
                                                    model_coefficients[itt].values[2]);

    int inliers_count_best = plane_inliers_counts[itt];

    //check that the other planes with similar normal are not higher than the table_plane_selected
    for (size_t i = 0; i < model_coefficients.size (); i++)
    {
      Eigen::Vector4f model = Eigen::Vector4f (model_coefficients[i].values[0], model_coefficients[i].values[1], model_coefficients[i].values[2],
                                               model_coefficients[i].values[3]);

      Eigen::Vector3f normal = Eigen::Vector3f (model_coefficients[i].values[0], model_coefficients[i].values[1], model_coefficients[i].values[2]);

      int inliers_count = plane_inliers_counts[i];

      std::cout << "Dot product is:" << normal.dot (normal_table) << std::endl;
      if ((normal.dot (normal_table) > 0.95) && (inliers_count_best * 0.5 <= inliers_count))
      {
        //check if this plane is higher, projecting a point on the normal direction
        std::cout << "Check if plane is higher, then change table plane" << std::endl;
        std::cout << model[3] << " " << table_plane[3] << std::endl;
        if (model[3] < table_plane[3])
        {
          PCL_WARN ("Changing table plane...");
          table_plane_selected = i;
          table_plane = model;
          normal_table = normal;
          inliers_count_best = inliers_count;
        }
      }
    }

    table_plane = Eigen::Vector4f (model_coefficients[table_plane_selected].values[0], model_coefficients[table_plane_selected].values[1],
                                   model_coefficients[table_plane_selected].values[2], model_coefficients[table_plane_selected].values[3]);

    //cluster..
    typename pcl::EuclideanClusterComparator<PointT, pcl::Normal, pcl::Label>::Ptr
                                                                                             euclidean_cluster_comparator_ (
                                                                                                                            new pcl::EuclideanClusterComparator<
                                                                                                                                PointT,
                                                                                                                                pcl::Normal,
                                                                                                                                pcl::Label> ());

    //create two labels, 1 one for points belonging to or under the plane, 1 for points above the plane
    label_indices.resize (2);

    for (int j = 0; j < xyz_points_andy->points.size (); j++)
    {
      Eigen::Vector3f xyz_p = xyz_points_andy->points[j].getVector3fMap ();

      if (!pcl_isfinite (xyz_p[0]) || !pcl_isfinite (xyz_p[1]) || !pcl_isfinite (xyz_p[2]))
        continue;

      float val = xyz_p[0] * table_plane[0] + xyz_p[1] * table_plane[1] + xyz_p[2] * table_plane[2] + table_plane[3];

      if (val >= 0.005f)
      {
        /*plane_points->points[j].x = std::numeric_limits<float>::quiet_NaN ();
         plane_points->points[j].y = std::numeric_limits<float>::quiet_NaN ();
         plane_points->points[j].z = std::numeric_limits<float>::quiet_NaN ();*/
        labels->points[j].label = 1;
        label_indices[0].indices.push_back (j);
      }
      else
      {
        labels->points[j].label = 0;
        label_indices[1].indices.push_back (j);
      }
    }

    std::vector<bool> plane_labels;
    plane_labels.resize (label_indices.size (), false);
    plane_labels[0] = true;

    euclidean_cluster_comparator_->setInputCloud (xyz_points_andy);
    euclidean_cluster_comparator_->setLabels (labels);
    euclidean_cluster_comparator_->setExcludeLabels (plane_labels);
    euclidean_cluster_comparator_->setDistanceThreshold (0.035f, true);

    pcl::PointCloud<pcl::Label> euclidean_labels;
    std::vector<pcl::PointIndices> euclidean_label_indices;
    pcl::OrganizedConnectedComponentSegmentation<PointT, pcl::Label> euclidean_segmentation (euclidean_cluster_comparator_);
    euclidean_segmentation.setInputCloud (xyz_points_andy);
    euclidean_segmentation.segment (euclidean_labels, euclidean_label_indices);

    for (size_t i = 0; i < euclidean_label_indices.size (); i++)
    {
      if (euclidean_label_indices[i].indices.size () >= min_cluster_size_)
      {
        indices.push_back (euclidean_label_indices[i]);
      }
    }
  }
  else
  {
    pcl::apps::DominantPlaneSegmentation<PointT> dps;
    dps.setInputCloud (xyz_points);
    dps.setMaxZBounds (Z_DIST_);
    dps.setObjectMinHeight (0.01);
    dps.setMinClusterSize (min_cluster_size_);
    dps.setWSize (9);
    dps.setDistanceBetweenClusters (0.05f);
    std::vector<typename pcl::PointCloud<PointT>::Ptr> clusters;
    dps.setDownsamplingSize (0.01f);
    dps.compute_fast (clusters);
    dps.getIndicesClusters (indices);
    dps.getTableCoefficients (table_plane);
  }
}

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
      std::string so_far = rel_path_so_far + (itr->path ().filename ()).string() + "/";
#else
      std::string so_far = rel_path_so_far + (itr->path ()).filename () + "/";
#endif
      bf::path curr_path = itr->path ();
      getScenesInDirectory (curr_path, so_far, relative_paths);
    }
    else
    {
      //check that it is a ply file and then add, otherwise ignore..
      std::vector < std::string > strs;
#if BOOST_FILESYSTEM_VERSION == 3
      std::string file = (itr->path ().filename ()).string();
#else
      std::string file = (itr->path ().filename ());
#endif

      boost::split (strs, file, boost::is_any_of ("."));
      std::string extension = strs[strs.size () - 1];

      if (extension == "pcd")
      {

#if BOOST_FILESYSTEM_VERSION == 3
        std::string path = rel_path_so_far + (itr->path ().filename ()).string();
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
      std::string so_far = rel_path_so_far + (itr->path ().filename ()).string() + "/";
#else
      std::string so_far = rel_path_so_far + (itr->path ()).filename () + "/";
#endif

      bf::path curr_path = itr->path ();
      getModelsInDirectory (curr_path, so_far, relative_paths, ext);
    }
    else
    {
      //check that it is a ply file and then add, otherwise ignore..
      std::vector < std::string > strs;
#if BOOST_FILESYSTEM_VERSION == 3
      std::string file = (itr->path ().filename ()).string();
#else
      std::string file = (itr->path ()).filename ();
#endif

      boost::split (strs, file, boost::is_any_of ("."));
      std::string extension = strs[strs.size () - 1];

      if (extension.compare (ext) == 0)
      {
#if BOOST_FILESYSTEM_VERSION == 3
        std::string path = rel_path_so_far + (itr->path ().filename ()).string();
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
  std::vector < std::string > strs1;
  boost::split (strs1, file1, boost::is_any_of ("/"));

  std::vector < std::string > strs2;
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

class go_params {
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

  void writeParamsToFile(std::ofstream & of) {
    of << "Params: \t" << go_opt_type << "\t";
    of << static_cast<int>(go_init) << "\t";
    of << radius_clutter << "\t" << clutter_regularizer << "\t";
    of << radius_normals_go_ << "\t" << static_cast<int>(go_use_replace_moves);
    of << std::endl;
  }
};

go_params parameters_for_go;

template<typename PointT>
void
recognizeAndVisualize (typename boost::shared_ptr<faat_pcl::rec_3d_framework::MultiRecognitionPipeline<PointT> > & local,
                          std::string & scene_file, int seg, bool add_planes=true)
{


  faat_pcl::rec_3d_framework::or_evaluator::OREvaluator<PointT> or_eval;
  or_eval.setGTDir(GT_DIR_);
  or_eval.setModelsDir(MODELS_DIR_);

  std::ofstream logfile_stream;
  logfile_stream.open (go_log_file_.c_str());
  parameters_for_go.writeParamsToFile(logfile_stream);

  boost::shared_ptr<faat_pcl::GlobalHypothesesVerification_1<PointT, PointT> > go (
                                                                                          new faat_pcl::GlobalHypothesesVerification_1<PointT,
                                                                                          PointT>);
  go->setSmoothSegParameters(0.1, 0.0125, 0.01);
  //go->setRadiusNormals(0.03f);
  go->setResolution (parameters_for_go.go_resolution);
  go->setMaxIterations (parameters_for_go.go_iterations);
  go->setInlierThreshold (parameters_for_go.go_inlier_thres);
  go->setRadiusClutter (parameters_for_go.radius_clutter);
  go->setRegularizer (parameters_for_go.regularizer);
  go->setClutterRegularizer (parameters_for_go.clutter_regularizer);
  go->setDetectClutter (parameters_for_go.detect_clutter);
  go->setOcclusionThreshold (0.01f);
  go->setOptimizerType(parameters_for_go.go_opt_type);
  go->setUseReplaceMoves(parameters_for_go.go_use_replace_moves);
  go->setInitialTemp(parameters_for_go.init_temp);
  go->setRadiusNormals(parameters_for_go.radius_normals_go_);
  go->setRequiresNormals(parameters_for_go.require_normals);
  go->setInitialStatus(parameters_for_go.go_init);

  boost::shared_ptr<faat_pcl::HypothesisVerification<PointT, PointT> > cast_hv_alg;
  cast_hv_alg = boost::static_pointer_cast<faat_pcl::HypothesisVerification<PointT, PointT> > (go);

  /*if(use_HV)
    local->setHVAlgorithm(cast_hv_alg);*/

  typename boost::shared_ptr<faat_pcl::rec_3d_framework::Source<PointT> > model_source_ = local->getDataSource ();
  typedef typename pcl::PointCloud<PointT>::ConstPtr ConstPointInTPtr;
  typedef faat_pcl::rec_3d_framework::Model<PointT> ModelT;
  typedef boost::shared_ptr<ModelT> ModelTPtr;

  local->setVoxelSizeICP (VX_SIZE_ICP_);

  pcl::visualization::PCLVisualizer vis ("Recognition results");
  int v1, v2, v3, v4, v5, v6;
  vis.createViewPort (0.0, 0.0, 0.33, 0.5, v1);
  vis.createViewPort (0.33, 0, 0.66, 0.5, v2);
  vis.createViewPort (0.0, 0.5, 0.33, 1, v3);
  vis.createViewPort (0.33, 0.5, 0.66, 1, v4);
  vis.createViewPort (0.66, 0, 1, 0.5, v5);
  vis.createViewPort (0.66, 0.5, 1, 1, v6);

  vis.addText ("go segmentation", 1, 30, 18, 1, 0, 0, "go_smooth", v3);
  vis.addText ("Ground truth", 1, 30, 18, 1, 0, 0, "gt_text", v4);
  vis.addText ("Scene", 1, 30, 18, 1, 0, 0, "scene_texttt", v5);
  vis.addText ("Hypotheses", 1, 30, 18, 1, 0, 0, "hypotheses_text", v6);
  vis.addText ("Final Results", 1, 30, 18, 1, 0, 0, "final_res_text", v2);

  bf::path input = scene_file;
  std::vector<std::string> files_to_recognize;

  if (bf::is_directory (input))
  {
    std::vector < std::string > files;
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

    std::sort(files_to_recognize.begin(),files_to_recognize.end());
    or_eval.setScenesDir(scene_file);
    or_eval.setDataSource(local->getDataSource());
    or_eval.loadGTData();
  }
  else
  {
    files_to_recognize.push_back (scene_file);
  }

  std::cout << "is segmentation required:" << local->isSegmentationRequired() << std::endl;

  for(size_t i=0; i < files_to_recognize.size(); i++) {

    std::cout << "recognizing " << files_to_recognize[i] << std::endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr scene (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile (files_to_recognize[i], *scene);

    {
      pcl::visualization::PointCloudColorHandlerGenericField<PointT> scene_handler (scene, "z");
      vis.addPointCloud<PointT> (scene, scene_handler, "scene_cloud_z_coloured", v5);
    }

    std::vector<std::string> strs1;
    boost::split (strs1, files_to_recognize[i], boost::is_any_of ("/"));

    std::string id_1 = strs1[strs1.size () - 1];
    size_t pos1 = id_1.find (".pcd");

    id_1 = id_1.substr (0, pos1);

    if(Z_DIST_ > 0)
    {
      pcl::PassThrough<PointT> pass_;
      pass_.setFilterLimits (0.f, Z_DIST_);
      pass_.setFilterFieldName ("z");
      pass_.setInputCloud (scene);
      pass_.setKeepOrganized (true);
      pass_.filter (*scene);
    }

    pcl::visualization::PointCloudColorHandlerCustom<PointT> scene_handler (scene, 125, 125, 125);
    vis.addPointCloud<PointT> (scene, scene_handler, "scene_cloud", v1);
    vis.addText (files_to_recognize[i], 1, 30, 14, 1, 0, 0, "scene_text", v1);

    //Multiplane segmentation
    faat_pcl::MultiPlaneSegmentation<PointT> mps;
    mps.setInputCloud(scene);
    mps.setMinPlaneInliers(1000);
    mps.setResolution(parameters_for_go.go_resolution);

    std::vector<faat_pcl::PlaneModel<PointT> > planes_found;
    mps.segment();
    planes_found = mps.getModels();

    if(planes_found.size() == 0 && scene->isOrganized())
    {
      mps.segment(true);
      planes_found = mps.getModels();
    }

    for(size_t kk=0; kk < planes_found.size(); kk++)
    {
      std::stringstream pname;
      pname << "plane_" << kk;

      pcl::visualization::PointCloudColorHandlerRandom<pcl::PointXYZ> scene_handler(planes_found[kk].plane_cloud_);
      vis.addPointCloud<pcl::PointXYZ> (planes_found[kk].plane_cloud_, scene_handler, pname.str(), v1);

      pname << "chull";
      vis.addPolygonMesh (*planes_found[kk].convex_hull_, pname.str(), v1);
    }

    vis.spinOnce();

    std::vector<pcl::PointIndices> indices;
    Eigen::Vector4f table_plane;
    doSegmentation<PointT>(scene, indices, table_plane, seg);

    if(local->isSegmentationRequired()) {
      //visualize segmentation
      for (size_t c = 0; c < indices.size (); c++)
      {
        /*if (indices[c].indices.size () < 500)
          continue;*/

        std::stringstream name;
        name << "cluster_" << c;

        typename pcl::PointCloud<PointT>::Ptr cluster (new pcl::PointCloud<PointT>);
        pcl::copyPointCloud (*scene, indices[c].indices, *cluster);

        pcl::visualization::PointCloudColorHandlerRandom<PointT> handler_rgb (cluster);
        vis.addPointCloud<PointT> (cluster, handler_rgb, name.str (), v1);
      }
    }

    //use table plane to define indices for the local pipeline as well...
    std::vector<int> indices_above_plane;

    {
      for (int k = 0; k < scene->points.size (); k++)
      {
        Eigen::Vector3f xyz_p = scene->points[k].getVector3fMap ();

        if (!pcl_isfinite (xyz_p[0]) || !pcl_isfinite (xyz_p[1]) || !pcl_isfinite (xyz_p[2]))
          continue;

        float val = xyz_p[0] * table_plane[0] + xyz_p[1] * table_plane[1] + xyz_p[2] * table_plane[2] + table_plane[3];

        if (val >= 0.01)
        {
          indices_above_plane.push_back (static_cast<int> (k));
        }
      }
    }

    local->setSegmentation(indices);
    local->setIndices(indices_above_plane);
    local->setInputCloud (scene);
    {
      pcl::ScopeTime ttt ("Recognition");
      local->recognize ();
    }

    //HV
    //transforms models
    boost::shared_ptr < std::vector<ModelTPtr> > models = local->getModels ();
    boost::shared_ptr < std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > > transforms = local->getTransforms ();
    std::vector<typename pcl::PointCloud<PointT>::ConstPtr> aligned_models;
    aligned_models.resize (models->size ());
    std::vector<std::string> model_ids;
    for (size_t kk = 0; kk < models->size (); kk++)
    {
      ConstPointInTPtr model_cloud = models->at (kk)->getAssembled (parameters_for_go.go_resolution);
      typename pcl::PointCloud<PointT>::Ptr model_aligned (new pcl::PointCloud<PointT>);
      pcl::transformPointCloud (*model_cloud, *model_aligned, transforms->at (kk));
      aligned_models[kk] = model_aligned;
      model_ids.push_back(models->at (kk)->id_);
    }

    go->setSceneCloud (scene);
    //addModels
    go->addModels (aligned_models, true);
    //append planar models
    if(add_planes)
    {
      go->addPlanarModels(planes_found);
      for(size_t kk=0; kk < planes_found.size(); kk++)
      {
        std::stringstream plane_id;
        plane_id << "plane_" << kk;
        model_ids.push_back(plane_id.str());
      }
    }

    go->setObjectIds(model_ids);
    //verify
    go->verify ();
    std::vector<bool> mask_hv;
    go->getMask (mask_hv);

    if(use_HV)
    {
      pcl::PointCloud<pcl::PointXYZRGBA>::Ptr smooth_cloud_ =  go->getSmoothClustersRGBCloud();
      if(smooth_cloud_)
      {
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> random_handler (smooth_cloud_);
        vis.addPointCloud<pcl::PointXYZRGBA> (smooth_cloud_, random_handler, "smooth_cloud", v3);
      }
    }

    std::vector<int> coming_from;
    coming_from.resize(aligned_models.size() + planes_found.size());
    for(size_t j=0; j < aligned_models.size(); j++)
      coming_from[j] = 0;

    for(size_t j=0; j < planes_found.size(); j++)
      coming_from[aligned_models.size() + j] = 1;

    or_eval.visualizeGroundTruth(vis, id_1, v4);

    boost::shared_ptr<std::vector<ModelTPtr> > verified_models(new std::vector<ModelTPtr>);
    boost::shared_ptr<std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > > verified_transforms;
    verified_transforms.reset(new std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >);

    if(models)
    {
      vtkSmartPointer < vtkTransform > scale_models = vtkSmartPointer<vtkTransform>::New ();
      scale_models->Scale(model_scale, model_scale, model_scale);

      for (size_t j = 0; j < mask_hv.size (); j++)
      {
        std::stringstream name;
        name << "cloud_" << j;

        if(!mask_hv[j])
        {
          if(coming_from[j] == 0)
          {
            ConstPointInTPtr model_cloud = models->at (j)->getAssembled (VX_SIZE_ICP_);
            typename pcl::PointCloud<PointT>::Ptr model_aligned (new pcl::PointCloud<PointT>);
            pcl::transformPointCloud (*model_cloud, *model_aligned, transforms->at (j));

            pcl::visualization::PointCloudColorHandlerRandom<PointT> random_handler (model_aligned);
            vis.addPointCloud<PointT> (model_aligned, random_handler, name.str (), v6);

            /*std::stringstream pathPly;
            pathPly << MODELS_DIR_FOR_VIS_ << "/" << models->at (j)->id_ << ".ply";
            vtkSmartPointer < vtkTransform > poseTransform = vtkSmartPointer<vtkTransform>::New ();
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
            vis.addModelFromPLYFile (pathPly.str (), poseTransform, cluster_name.str (), v3);*/
          }
          continue;
        }

        if(coming_from[j] == 0)
        {
          verified_models->push_back(models->at(j));
          verified_transforms->push_back(transforms->at(j));

          ConstPointInTPtr model_cloud = models->at (j)->getAssembled (VX_SIZE_ICP_);
          typename pcl::PointCloud<PointT>::Ptr model_aligned (new pcl::PointCloud<PointT>);
          pcl::transformPointCloud (*model_cloud, *model_aligned, transforms->at (j));

          std::cout << models->at (j)->id_ << std::endl;

          pcl::visualization::PointCloudColorHandlerRandom<PointT> random_handler (model_aligned);
          vis.addPointCloud<PointT> (model_aligned, random_handler, name.str (), v2);

          std::stringstream pathPly;
          pathPly << MODELS_DIR_FOR_VIS_ << "/" << models->at (j)->id_ << ".ply";
          vtkSmartPointer < vtkTransform > poseTransform = vtkSmartPointer<vtkTransform>::New ();
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
          vis.addModelFromPLYFile (pathPly.str (), poseTransform, cluster_name.str (), v2);
        }
        else
        {
          std::stringstream pname;
          pname << "plane_" << j;

          pcl::visualization::PointCloudColorHandlerRandom<pcl::PointXYZ> scene_handler(planes_found[j - models->size()].plane_cloud_);
          vis.addPointCloud<pcl::PointXYZ> (planes_found[j - models->size()].plane_cloud_, scene_handler, pname.str(), v2);

          pname << "chull";
          vis.addPolygonMesh (*planes_found[j - models->size()].convex_hull_, pname.str(), v2);
        }
      }
    }

    or_eval.addRecognitionResults(id_1, verified_models, verified_transforms);

    vis.setBackgroundColor(0.0,0.0,0.0);
    if(PLAY_) {
      vis.spinOnce (100.f, true);
    } else {
      vis.spin ();
    }

    vis.removePointCloud ("scene_cloud");
    vis.removeShape ("scene_text");
    vis.removeAllShapes(v2);
    vis.removeAllPointClouds();

    //log results into file
    if(use_HV)
      logfile_stream << files_to_recognize[i] << "\t";
    //go->writeToLog(logfile_stream, false);
  }

  or_eval.computeStatistics();
  logfile_stream.close();
}

typedef pcl::ReferenceFrame RFType;

int CG_SIZE_ = 3;
float CG_THRESHOLD_ = 0.005f;

/*
 * ./bin/mp_training_recognition -pcd_file /home/aitor/data/ECCV_dataset/pcd_files_reduced/ -models_dir /home/aitor/data/ECCV_dataset/cad_models_2/ -training_dir /home/aitor/data/eccv_trained/ -idx_flann_fn eccv_flann_new.idx -go_require_normals 0 -GT_DIR /home/aitor/data/ECCV_dataset/gt_or_format/ -tes_level 0 -model_scale 0.001 -icp_iterations 10 -pipelines_to_use shot_omp,our_cvfh -go_opt_type 0 -gc_size 5 -gc_threshold 0.01 -splits 32 -test_sampling_density 0.01 -icp_type 1
 * ./bin/mp_training_recognition -pcd_file /home/aitor/data/ECCV_dataset/pcd_files_reduced/ -models_dir /home/aitor/data/ECCV_dataset/cad_models_reduced/ -training_dir /home/aitor/data/eccv_trained_level_1/ -idx_flann_fn eccv_flann_shot_cad_models_reduced_tes_level1.idx -go_require_normals 0 -GT_DIR /home/aitor/data/ECCV_dataset/gt_or_format/ -tes_level 1 -model_scale 0.001 -pipelines_to_use shot_omp -go_opt_type 0 -gc_size 5 -gc_threshold 0.015 -splits 64 -test_sampling_density 0.005 -icp_type 1 -training_dir_shot /home/aitor/data/eccv_trained_level_1/ -icp_iterations 5 -max_our_cvfh_hyp_ 20 -load_views 0 -normalize_ourcvfh_bins 0 -thres_hyp 0
 */

/*
 * ./bin/mp_training_recognition -pcd_file /home/aitor/data/ECCV_dataset/pcd_files/ -models_dir /home/aitor/data/ECCV_dataset/cad_models/ -training_dir /home/aitor/data/eccv_trained_level_1/ -idx_flann_fn eccv_flann_new.idx -go_require_normals 0 -GT_DIR /home/aitor/data/ECCV_dataset/gt_or_format/ -tes_level 0 -model_scale 0.001 -icp_iterations 10 -pipelines_to_use shot_omp,our_cvfh -go_opt_type 0 -gc_size 5 -gc_threshold 0.01 -splits 32 -test_sampling_density 0.005 -icp_type 1 -training_dir_shot /home/aitor/data/eccv_trained_level_0/ -Z_DIST 1.7 -normalize_ourcvfh_bins 0 -detect_clutter 1 -go_resolution 0.005 -go_regularizer 1 -go_inlier_thres 0.005 -PLAY 0 -max_our_cvfh_hyp_ 20 -seg_type 1 -add_planes 1 -use_codebook 1
 */

int
main (int argc, char ** argv)
{
  std::string path = "";
  std::string desc_name = "shot_omp";
  std::string training_dir = "trained_models/";
  std::string training_dir_shot = "";
  std::string pcd_file = "";
  int force_retrain = 0;
  int icp_iterations = 20;
  int use_cache = 1;
  int splits = 512;
  int scene = -1;
  int detect_clutter = 1;
  float thres_hyp_ = 0.2f;
  float desc_radius = 0.04f;
  int icp_type = 0;
  int go_opt_type = 2;
  int go_iterations = 7000;
  bool go_use_replace_moves = true;
  float go_inlier_thres = 0.01f;
  float go_resolution = 0.005f;
  float go_regularizer = 2.f;
  float go_clutter_regularizer = 5.f;
  float go_radius_clutter = 0.05f;
  float init_temp = 1000.f;
  std::string idx_flann_fn;
  float radius_normals_go_ = 0.02f;
  bool go_require_normals = false;
  bool go_log = true;
  bool go_init = false;
  float test_sampling_density = 0.005f;
  int tes_level_ = 1;
  int tes_level_our_cvfh_ = 1;
  std::string pipelines_to_use_ = "shot_omp,our_cvfh";
  bool normalize_ourcvfh_bins = false;
  int max_our_cvfh_hyp_ = 30;
  bool use_hough = false;
  float ransac_threshold_cg_ = CG_THRESHOLD_;
  bool load_views = true;
  int seg_type = 0;
  bool add_planes = true;
  bool cg_prune_hyp = false;
  bool use_codebook = false;

  pcl::console::parse_argument (argc, argv, "-use_codebook", use_codebook);
  pcl::console::parse_argument (argc, argv, "-cg_prune_hyp", cg_prune_hyp);
  pcl::console::parse_argument (argc, argv, "-seg_type", seg_type);
  pcl::console::parse_argument (argc, argv, "-add_planes", add_planes);
  pcl::console::parse_argument (argc, argv, "-use_hv", use_HV);
  pcl::console::parse_argument (argc, argv, "-max_our_cvfh_hyp_", max_our_cvfh_hyp_);
  pcl::console::parse_argument (argc, argv, "-normalize_ourcvfh_bins", normalize_ourcvfh_bins);
  pcl::console::parse_argument (argc, argv, "-models_dir", path);
  pcl::console::parse_argument (argc, argv, "-training_dir", training_dir);
  pcl::console::parse_argument (argc, argv, "-training_dir_shot", training_dir_shot);
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
  pcl::console::parse_argument (argc, argv, "-go_regularizer", go_regularizer);
  pcl::console::parse_argument (argc, argv, "-go_clutter_regularizer", go_clutter_regularizer);
  pcl::console::parse_argument (argc, argv, "-go_radius_clutter", go_radius_clutter);
  pcl::console::parse_argument (argc, argv, "-go_log", go_log);
  pcl::console::parse_argument (argc, argv, "-go_init", go_init);
  pcl::console::parse_argument (argc, argv, "-idx_flann_fn", idx_flann_fn);
  pcl::console::parse_argument (argc, argv, "-PLAY", PLAY_);
  pcl::console::parse_argument (argc, argv, "-go_log_file", go_log_file_);
  pcl::console::parse_argument (argc, argv, "-test_sampling_density", test_sampling_density);
  pcl::console::parse_argument (argc, argv, "-tes_level", tes_level_);
  pcl::console::parse_argument (argc, argv, "-Z_DIST", Z_DIST_);
  pcl::console::parse_argument (argc, argv, "-pipelines_to_use", pipelines_to_use_);
  pcl::console::parse_argument (argc, argv, "-use_hough", use_hough);
  pcl::console::parse_argument (argc, argv, "-ransac_threshold_cg_", ransac_threshold_cg_);
  pcl::console::parse_argument (argc, argv, "-load_views", load_views);

  MODELS_DIR_FOR_VIS_ = path;
  pcl::console::parse_argument (argc, argv, "-models_dir_vis", MODELS_DIR_FOR_VIS_);
  pcl::console::parse_argument (argc, argv, "-GT_DIR", GT_DIR_);
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
    std::vector < std::string > files;
    std::string start = "";
    std::string ext = std::string ("ply");
    bf::path dir = models_dir_path;
    getModelsInDirectory (dir, start, files, ext);
    std::cout << "Number of models in directory is:" << files.size() << std::endl;
  }

  parameters_for_go.radius_normals_go_ = radius_normals_go_;
  parameters_for_go.radius_clutter = go_radius_clutter;
  parameters_for_go.clutter_regularizer = go_clutter_regularizer;
  parameters_for_go.regularizer = go_regularizer;
  parameters_for_go.init_temp = init_temp;
  parameters_for_go.go_init = go_init;
  parameters_for_go.go_inlier_thres = go_inlier_thres;
  parameters_for_go.go_iterations = go_iterations;
  parameters_for_go.go_opt_type = go_opt_type;
  parameters_for_go.require_normals = go_require_normals;
  parameters_for_go.go_resolution = go_resolution;
  parameters_for_go.go_use_replace_moves = go_use_replace_moves;
  parameters_for_go.detect_clutter = static_cast<bool>(detect_clutter);

  //configure normal estimator
  boost::shared_ptr<faat_pcl::rec_3d_framework::PreProcessorAndNormalEstimator<pcl::PointXYZ, pcl::Normal> > normal_estimator;
  normal_estimator.reset (new faat_pcl::rec_3d_framework::PreProcessorAndNormalEstimator<pcl::PointXYZ, pcl::Normal>);
  normal_estimator->setCMR (false);
  normal_estimator->setDoVoxelGrid (true);
  normal_estimator->setRemoveOutliers (true);
  normal_estimator->setValuesForCMRFalse (0.003f, 0.02f);

  //configure keypoint extractor
  boost::shared_ptr<faat_pcl::rec_3d_framework::UniformSamplingExtractor<pcl::PointXYZ> > uniform_keypoint_extractor ( new faat_pcl::rec_3d_framework::UniformSamplingExtractor<pcl::PointXYZ>);
  
  uniform_keypoint_extractor->setSamplingDensity (0.01f);
  //uniform_keypoint_extractor->setSamplingDensity (0.005f);
  uniform_keypoint_extractor->setFilterPlanar (true);

  boost::shared_ptr<faat_pcl::rec_3d_framework::KeypointExtractor<pcl::PointXYZ> > keypoint_extractor;
  keypoint_extractor = boost::static_pointer_cast<faat_pcl::rec_3d_framework::KeypointExtractor<pcl::PointXYZ> > (uniform_keypoint_extractor);

  //configure cg algorithm (geometric consistency grouping)
  boost::shared_ptr<pcl::CorrespondenceGrouping<pcl::PointXYZ, pcl::PointXYZ> > cast_cg_alg;

  boost::shared_ptr<pcl::Hough3DGrouping<pcl::PointXYZ, pcl::PointXYZ, pcl::ReferenceFrame, pcl::ReferenceFrame> >
    hough_3d_voting_cg_alg(new pcl::Hough3DGrouping<pcl::PointXYZ, pcl::PointXYZ, pcl::ReferenceFrame, pcl::ReferenceFrame>);

  hough_3d_voting_cg_alg->setHoughBinSize (CG_THRESHOLD_);
  hough_3d_voting_cg_alg->setHoughThreshold (CG_SIZE_);
  hough_3d_voting_cg_alg->setUseInterpolation (false);
  hough_3d_voting_cg_alg->setUseDistanceWeight (false);
  //hough_3d_voting_cg_alg->setRansacThreshold(ransac_threshold_cg_);

  boost::shared_ptr<pcl::GeometricConsistencyGrouping<pcl::PointXYZ, pcl::PointXYZ> > gcg_alg (
                                                                                               new pcl::GeometricConsistencyGrouping<pcl::PointXYZ,
                                                                                                   pcl::PointXYZ>);


  gcg_alg->setGCThreshold (CG_SIZE_);
  gcg_alg->setGCSize (CG_THRESHOLD_);
  //gcg_alg->setRansacThreshold(ransac_threshold_cg_);

  cast_cg_alg = boost::static_pointer_cast<pcl::CorrespondenceGrouping<pcl::PointXYZ, pcl::PointXYZ> > (gcg_alg);

  {
    boost::shared_ptr<faat_pcl::GraphGeometricConsistencyGrouping<pcl::PointXYZ, pcl::PointXYZ> > gcg_alg (
                                                                                                 new faat_pcl::GraphGeometricConsistencyGrouping<pcl::PointXYZ,
                                                                                                     pcl::PointXYZ>);
    gcg_alg->setGCThreshold (CG_SIZE_);
    gcg_alg->setGCSize (CG_THRESHOLD_);
    gcg_alg->setRansacThreshold (ransac_threshold_cg_);
    gcg_alg->setUseGraph(true);
    gcg_alg->setPrune(cg_prune_hyp);
    gcg_alg->setDotDistance(1.f);
    gcg_alg->setDistForClusterFactor(0.5f);
    cast_cg_alg = boost::static_pointer_cast<pcl::CorrespondenceGrouping<pcl::PointXYZ, pcl::PointXYZ> > (gcg_alg);
  }


#ifdef _MSC_VER
  _CrtCheckMemory();
#endif

  std::vector<std::string> strs;
  boost::split (strs, pipelines_to_use_, boost::is_any_of (","));

  boost::shared_ptr<faat_pcl::rec_3d_framework::MultiRecognitionPipeline<pcl::PointXYZ> > multi_recog;
  multi_recog.reset(new faat_pcl::rec_3d_framework::MultiRecognitionPipeline<pcl::PointXYZ>);

  for(size_t i=0; i < strs.size(); i++)
  {

    boost::shared_ptr<faat_pcl::rec_3d_framework::Recognizer<pcl::PointXYZ> > cast_recog;

    if (strs[i].compare ("shot_omp") == 0)
    {
      desc_name = std::string ("shot");
      boost::shared_ptr<faat_pcl::rec_3d_framework::SHOTLocalEstimationOMP<pcl::PointXYZ, pcl::Histogram<352> > > estimator;
      estimator.reset (new faat_pcl::rec_3d_framework::SHOTLocalEstimationOMP<pcl::PointXYZ, pcl::Histogram<352> >);
      estimator->setNormalEstimator (normal_estimator);
      estimator->addKeypointExtractor (keypoint_extractor);
      estimator->setSupportRadius (desc_radius);
      estimator->setAdaptativeMLS (true);

      boost::shared_ptr<faat_pcl::rec_3d_framework::LocalEstimator<pcl::PointXYZ, pcl::Histogram<352> > > cast_estimator;
      cast_estimator = boost::dynamic_pointer_cast<faat_pcl::rec_3d_framework::LocalEstimator<pcl::PointXYZ, pcl::Histogram<352> > > (estimator);

      //configure mesh source for shot
      boost::shared_ptr<faat_pcl::rec_3d_framework::MeshSource<pcl::PointXYZ> > mesh_source (new faat_pcl::rec_3d_framework::MeshSource<pcl::PointXYZ>);
      mesh_source->setPath (path);
      mesh_source->setResolution (250);
      mesh_source->setTesselationLevel (tes_level_);
      mesh_source->setViewAngle (57.f);
      mesh_source->setRadiusSphere (1.5f);
      mesh_source->setModelScale (model_scale);
      mesh_source->setLoadViews(load_views);
      mesh_source->setLoadIntoMemory(false);
      //mesh_source->setRadiusNormals(radius_normals_go_);
      mesh_source->generate (training_dir_shot);

      boost::shared_ptr<faat_pcl::rec_3d_framework::Source<pcl::PointXYZ> > cast_source;
      cast_source = boost::static_pointer_cast<faat_pcl::rec_3d_framework::MeshSource<pcl::PointXYZ> > (mesh_source);

      if(use_hough)
      {

        cast_cg_alg = boost::static_pointer_cast<pcl::Hough3DGrouping<pcl::PointXYZ, pcl::PointXYZ> > (hough_3d_voting_cg_alg);

        boost::shared_ptr<faat_pcl::rec_3d_framework::LocalRecognitionHoughGroupingPipeline<flann::L1, pcl::PointXYZ, pcl::Histogram<352> > > local;
        local.reset(new faat_pcl::rec_3d_framework::LocalRecognitionHoughGroupingPipeline<flann::L1, pcl::PointXYZ, pcl::Histogram<352> > (idx_flann_fn));
        local->setDataSource (cast_source);
        local->setTrainingDir (training_dir_shot);
        local->setDescriptorName (desc_name);
        local->setFeatureEstimator (cast_estimator);
        local->setCGAlgorithm (cast_cg_alg);

        local->setUseCache (static_cast<bool> (use_cache));
        local->setVoxelSizeICP (VX_SIZE_ICP_);
        local->setThresholdAcceptHyp (thres_hyp_);
        uniform_keypoint_extractor->setSamplingDensity (test_sampling_density);
        local->setICPIterations (0);
        local->setKdtreeSplits (splits);
        local->setICPType(icp_type);

        local->initialize (static_cast<bool> (force_retrain));

        cast_recog = boost::static_pointer_cast<faat_pcl::rec_3d_framework::LocalRecognitionHoughGroupingPipeline<flann::L1, pcl::PointXYZ, pcl::Histogram<352> > > (local);
        multi_recog->addRecognizer(cast_recog);
      }
      else
      {
        boost::shared_ptr<faat_pcl::rec_3d_framework::LocalRecognitionPipeline<flann::L1, pcl::PointXYZ, pcl::Histogram<352> > > local;
        local.reset(new faat_pcl::rec_3d_framework::LocalRecognitionPipeline<flann::L1, pcl::PointXYZ, pcl::Histogram<352> > (idx_flann_fn));
        local->setDataSource (cast_source);
        local->setTrainingDir (training_dir_shot);
        local->setDescriptorName (desc_name);
        local->setFeatureEstimator (cast_estimator);
        local->setCGAlgorithm (cast_cg_alg);

        local->setUseCache (static_cast<bool> (use_cache));
        local->setVoxelSizeICP (VX_SIZE_ICP_);
        local->setThresholdAcceptHyp (thres_hyp_);
        uniform_keypoint_extractor->setSamplingDensity (test_sampling_density);
        local->setICPIterations (0);
        local->setKdtreeSplits (splits);
        local->setICPType(icp_type);
        local->setUseCodebook(use_codebook);
        local->initialize (static_cast<bool> (force_retrain));

        cast_recog = boost::static_pointer_cast<faat_pcl::rec_3d_framework::LocalRecognitionPipeline<flann::L1, pcl::PointXYZ, pcl::Histogram<352> > > (local);
        multi_recog->addRecognizer(cast_recog);
      }
    }

    if (strs[i].compare ("our_cvfh") == 0)
    {
      bool normalize_bins_ = normalize_ourcvfh_bins;
      int nn_ = 45;
      desc_name = std::string ("our_cvfh");

      //configure mesh source for our_cvfh
      boost::shared_ptr<faat_pcl::rec_3d_framework::MeshSource<pcl::PointXYZ> > mesh_source (new faat_pcl::rec_3d_framework::MeshSource<pcl::PointXYZ>);
      mesh_source->setPath (path);
      mesh_source->setResolution (250);
      mesh_source->setTesselationLevel (tes_level_our_cvfh_);
      mesh_source->setViewAngle (57.f);
      mesh_source->setRadiusSphere (1.5f);
      mesh_source->setModelScale (model_scale);
      mesh_source->setLoadViews(load_views);
      mesh_source->setLoadIntoMemory(false);
      //mesh_source->setRadiusNormals(radius_normals_go_);
      mesh_source->generate (training_dir);

      boost::shared_ptr<faat_pcl::rec_3d_framework::Source<pcl::PointXYZ> > cast_source;
      cast_source = boost::static_pointer_cast<faat_pcl::rec_3d_framework::MeshSource<pcl::PointXYZ> > (mesh_source);

      boost::shared_ptr<faat_pcl::rec_3d_framework::GlobalNNCVFHRecognizer<faat_pcl::Metrics::HistIntersectionUnionDistance, pcl::PointXYZ, pcl::VFHSignature308> > ourcvfh_global_;

      boost::shared_ptr<faat_pcl::rec_3d_framework::OURCVFHEstimator<pcl::PointXYZ, pcl::VFHSignature308> > vfh_estimator;
      vfh_estimator.reset (new faat_pcl::rec_3d_framework::OURCVFHEstimator<pcl::PointXYZ, pcl::VFHSignature308>);
      vfh_estimator->setNormalEstimator (normal_estimator);
      vfh_estimator->setNormalizeBins (normalize_bins_);
      vfh_estimator->setCVFHParams (0.13f, 0.0125f, 2.f);
      vfh_estimator->setRefineClustersParam (2.5f);
      vfh_estimator->setAdaptativeMLS (false);

      vfh_estimator->setAxisRatio (1.f);
      vfh_estimator->setMinAxisValue (1.f);

      //vfh_estimator->setCVFHParams (0.15f, 0.015f, 2.5f);

      std::string desc_name = "our_cvfh";
      if (normalize_bins_)
      {
        desc_name = "our_cvfh_normalized";
      }

      std::cout << "Descriptor name:" << desc_name << std::endl;
      boost::shared_ptr<faat_pcl::rec_3d_framework::OURCVFHEstimator<pcl::PointXYZ, pcl::VFHSignature308> > cast_estimator;
      cast_estimator = boost::dynamic_pointer_cast<faat_pcl::rec_3d_framework::OURCVFHEstimator<pcl::PointXYZ, pcl::VFHSignature308> > (vfh_estimator);

      ourcvfh_global_.reset(new faat_pcl::rec_3d_framework::GlobalNNCVFHRecognizer<faat_pcl::Metrics::HistIntersectionUnionDistance, pcl::PointXYZ, pcl::VFHSignature308>);
      ourcvfh_global_->setDataSource (cast_source);
      ourcvfh_global_->setTrainingDir (training_dir);
      ourcvfh_global_->setDescriptorName (desc_name);
      ourcvfh_global_->setFeatureEstimator (cast_estimator);
      ourcvfh_global_->setNN (nn_);
      ourcvfh_global_->setICPIterations (0);
      ourcvfh_global_->setNoise (0.0f);
      ourcvfh_global_->setUseCache (false);
      ourcvfh_global_->setMaxHyp(max_our_cvfh_hyp_);
      ourcvfh_global_->initialize (static_cast<bool> (force_retrain));

      {
        //segmentation parameters for recognition
        //vfh_estimator->setCVFHParams (0.15f, 0.015f, 2.5f);

        std::vector<float> eps_thresholds, cur_thresholds, clus_thresholds;
        eps_thresholds.push_back (0.13);
        eps_thresholds.push_back (0.150f);
        eps_thresholds.push_back (0.1750f);
        //cur_thresholds.push_back (0.015f);
        //cur_thresholds.push_back (0.0175f);
        //cur_thresholds.push_back (0.02f);
        //cur_thresholds.push_back (0.01f);
        cur_thresholds.push_back (0.0125f);
        cur_thresholds.push_back (0.015f);
        //cur_thresholds.push_back (0.02f);
        //cur_thresholds.push_back (0.035f);
        clus_thresholds.push_back (2.5f);

        vfh_estimator->setClusterToleranceVector (clus_thresholds);
        vfh_estimator->setEpsAngleThresholdVector (eps_thresholds);
        vfh_estimator->setCurvatureThresholdVector (cur_thresholds);

        vfh_estimator->setAxisRatio (0.8f);
        vfh_estimator->setMinAxisValue (0.8f);

        vfh_estimator->setAdaptativeMLS (true);
      }

      cast_recog = boost::static_pointer_cast<faat_pcl::rec_3d_framework::GlobalNNCVFHRecognizer<faat_pcl::Metrics::HistIntersectionUnionDistance, pcl::PointXYZ, pcl::VFHSignature308> > (ourcvfh_global_);
      multi_recog->addRecognizer(cast_recog);
    }
  }

  multi_recog->setVoxelSizeICP(VX_SIZE_ICP_);
  multi_recog->setICPType(icp_type);
  multi_recog->setICPIterations(icp_iterations);
  multi_recog->initialize();
  recognizeAndVisualize<pcl::PointXYZ> (multi_recog, pcd_file, seg_type, add_planes);
}
