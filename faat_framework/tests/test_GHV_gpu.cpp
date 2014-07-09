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
#include <faat_pcl/recognition/hv/ghv_cuda_wrapper.h>

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
bool go_use_planes = true;
std::string timine_file_ = "gpu_timing.txt";

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
std::string TIMING_OUTPUT_FILE_ = "gpu_timing.txt";

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

    faat_pcl::rec_3d_framework::or_evaluator::OREvaluator<PointT> or_hypotheses;
    or_hypotheses.setGTDir(HYPOTHESES_DIR_);
    or_hypotheses.setModelsDir(MODELS_DIR_);
    or_hypotheses.setModelFileExtension(model_ext);
    or_hypotheses.setReplaceModelExtension(false);
    or_hypotheses.setDataSource(source);

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
            or_hypotheses.setIgnoreList(ignore_list);
        }

        or_hypotheses.setScenesDir(scene_file);
        or_hypotheses.loadGTData();
    }
    else
    {
        files_to_recognize.push_back (scene_file);
    }

    typename faat_pcl::recognition::GHVCudaWrapper<PointT> ghv;
    pcl::visualization::PCLVisualizer vis("GHV - GPU visualizer");
    int v1,v2,v3;
    vis.createViewPort(0,0,0.33,1,v1);
    vis.createViewPort(0.33,0,0.66,1,v2);
    vis.createViewPort(0.66,0,1,1,v3);

    TimingStructure timing;

    for (size_t i = 0; i < files_to_recognize.size (); i++)
    {

        pcl::ScopeTime tttt("The whole processing");
        vis.removeAllPointClouds();
        vis.removeAllShapes();

        std::cout << parameters_for_go.color_sigma_ab_ << " " << parameters_for_go.color_sigma_l_ << std::endl;

        std::string file_to_recognize(files_to_recognize[i]);
        boost::replace_all (file_to_recognize, scene_file, "");

        boost::replace_all (file_to_recognize, ".pcd", "");

        std::string id_1 = file_to_recognize;

        typename pcl::PointCloud<PointT>::Ptr scene (new pcl::PointCloud<PointT>);
        pcl::io::loadPCDFile (files_to_recognize[i], *scene);

        vis.addPointCloud(scene, "scene", v1);

        pcl::PointCloud<pcl::Normal>::Ptr normal_cloud_scene (new pcl::PointCloud<pcl::Normal>);

        typename pcl::PointCloud<PointT>::Ptr occlusion_cloud (new pcl::PointCloud<PointT>(*scene));

        float res = 0.004f;
        std::vector<faat_pcl::PlaneModel<PointT> > planes_found;

        if (Z_DIST_ > 0)
        {
            pcl::ScopeTime t("finding planes...");
            //compute planes
            typename pcl::PointCloud<PointT>::Ptr voxelized (new pcl::PointCloud<PointT>());

            pcl::PassThrough<PointT> pass_;
            pass_.setFilterLimits (0.f, Z_DIST_);
            pass_.setFilterFieldName ("z");
            pass_.setInputCloud (scene);
            pass_.setKeepOrganized (true);
            pass_.filter (*voxelized);

            faat_pcl::MultiPlaneSegmentation<PointT> mps;
            mps.setInputCloud(voxelized);
            mps.setMinPlaneInliers(1000);
            mps.setResolution(res);
            mps.setMergePlanes(true);
            mps.segment(false);
            planes_found = mps.getModels();
            std::cout << "Number of planes found in the scene:" << planes_found.size() << std::endl;
        }

        if (Z_DIST_ > 0)
        {

            typename pcl::PointCloud<PointT>::Ptr voxelized (new pcl::PointCloud<PointT>());

            pcl::PassThrough<PointT> pass_;
            pass_.setFilterLimits (0.f, Z_DIST_);
            pass_.setFilterFieldName ("z");
            pass_.setInputCloud (scene);
            pass_.setKeepOrganized (false);
            pass_.filter (*voxelized);

            float VOXEL_SIZE_ICP_ = res;
            pcl::VoxelGrid<PointT> voxel_grid_icp;
            voxel_grid_icp.setInputCloud (voxelized);
            voxel_grid_icp.setLeafSize (VOXEL_SIZE_ICP_, VOXEL_SIZE_ICP_, VOXEL_SIZE_ICP_);
            voxel_grid_icp.filter (*scene);

            std::cout << "cloud is organized:" << scene->isOrganized() << std::endl;
            std::cout << scene->width << " " << scene->height << " " << FORCE_UNORGANIZED_ << " " << Z_DIST_ << std::endl;
        }

        pcl::NormalEstimationOMP<PointT, pcl::Normal> ne;
        ne.setRadiusSearch(0.02f);
        ne.setInputCloud (scene);
        ne.compute (*normal_cloud_scene);

        boost::shared_ptr<std::vector<ModelTPtr> > models;
        boost::shared_ptr<std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > > transforms;
        models.reset(new std::vector<ModelTPtr>);
        transforms.reset(new std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >);

        or_hypotheses.getGroundTruthModelsAndPoses(id_1, models, transforms);

        int kept = 0;
        float fsv_threshold = 0.2f;

        for (size_t kk = 0; kk < models->size (); kk++, kept++)
        {
             ConstPointInTPtr model_cloud = models->at (kk)->getAssembled (res);
             typename pcl::PointCloud<PointT>::Ptr model_aligned (new pcl::PointCloud<PointT>);
             pcl::transformPointCloud (*model_cloud, *model_aligned, transforms->at (kk));

             if(PRE_FILTER_)
             {
                 if(occlusion_cloud->isOrganized())
                 {
                     //compute FSV for the model and occlusion_cloud
                     faat_pcl::registration::VisibilityReasoning<PointT> vr (525.f, 640, 480);
                     vr.setThresholdTSS (0.01f);

                     float fsv_ij = 0;

                     pcl::PointCloud<pcl::Normal>::ConstPtr normal_cloud = models->at (kk)->getNormalsAssembled (res);
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

                     if(fsv_ij > fsv_threshold)
                     {
                         kept--;
                         continue;
                     }
                 }
             }

             models->at(kept) = models->at(kk);
             transforms->at(kept) = transforms->at(kk);
        }

        models->resize(kept);
        transforms->resize(kept);

        std::vector<typename pcl::PointCloud<PointT>::ConstPtr> aligned_models;
        std::vector<pcl::PointCloud<pcl::Normal>::ConstPtr> aligned_normals;
        std::vector<pcl::PointCloud<pcl::PointXYZL>::Ptr> aligned_smooth_faces;

        aligned_models.resize (models->size ());
        aligned_smooth_faces.resize (models->size ());
        aligned_normals.resize (models->size ());

        std::map<std::string, int> id_to_model_clouds;
        std::map<std::string, int>::iterator it;
        std::vector<Eigen::Matrix4f> transformations;
        std::vector<int> transforms_to_models;
        transforms_to_models.resize(models->size());
        transformations.resize(models->size());

        int individual_models = 0;

        for (size_t kk = 0; kk < models->size (); kk++)
        {

            int pos = 0;
            it = id_to_model_clouds.find(models->at(kk)->id_);
            if(it == id_to_model_clouds.end())
            {
                //not included yet
                ConstPointInTPtr model_cloud = models->at (kk)->getAssembled (res);
                pcl::PointCloud<pcl::Normal>::ConstPtr normal_cloud = models->at (kk)->getNormalsAssembled (res);
                aligned_models[individual_models] = model_cloud;
                aligned_normals[individual_models] = normal_cloud;
                pos = individual_models;

                id_to_model_clouds.insert(std::make_pair(models->at(kk)->id_, individual_models));

                individual_models++;
            }
            else
            {
                pos = it->second;
            }

            transformations[kk] = transforms->at(kk);
            transforms_to_models[kk] = pos;

            //visualize all hypotheses
            std::stringstream name;
            name << "hypotheses_" << kk;
            ConstPointInTPtr model_cloud = models->at (kk)->getAssembled (0.005f);
            typename pcl::PointCloud<PointT>::Ptr model_aligned (new pcl::PointCloud<PointT>);
            pcl::transformPointCloud (*model_cloud, *model_aligned, transforms->at (kk));

            pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> scene_handler(model_aligned);
            vis.addPointCloud<pcl::PointXYZRGB> (model_aligned, scene_handler, name.str (), v2);
        }

        aligned_models.resize(individual_models);
        aligned_normals.resize(individual_models);
        std::cout << "aligned models size:" << aligned_models.size() << " " << models->size() << std::endl;

        ghv.setSceneCloud(scene);
        ghv.setSceneNormals(normal_cloud_scene);
        ghv.setOcclusionCloud(occlusion_cloud);
        ghv.addModelNormals(aligned_normals);
        ghv.addModels(aligned_models, transformations, transforms_to_models);

        if(go_use_planes)
            ghv.addPlanarModels(planes_found);

        ghv.verify();

        float t_cues = ghv.getCuesComputationTime();
        float t_opt = ghv.getOptimizationTime();
        int num_p = ghv.getNumberOfVisiblePoints();

        timing.addResult(static_cast<int>(models->size() + planes_found.size()), t_cues, t_opt, num_p);

        std::vector<bool> sol = ghv.getSolution();
        for(size_t i=0; i < sol.size(); i++)
        {
            if(!sol[i])
                continue;

            if(i < models->size())
            {
                //object hypotheses verified
                std::stringstream name;
                name << "hypotheses_verified" << i;
                ConstPointInTPtr model_cloud = models->at (i)->getAssembled (0.001f);
                typename pcl::PointCloud<PointT>::Ptr model_aligned (new pcl::PointCloud<PointT>);
                pcl::transformPointCloud (*model_cloud, *model_aligned, transforms->at (i));

                pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> scene_handler(model_aligned);
                vis.addPointCloud<pcl::PointXYZRGB> (model_aligned, scene_handler, name.str (), v3);
            }
            else
            {
                //plane
                std::stringstream pname;
                pname << "plane_" << i;

                pcl::visualization::PointCloudColorHandlerCustom<PointT> scene_handler(planes_found[i - models->size()].plane_cloud_, 0, 255, 0);
                vis.addPointCloud (planes_found[i - models->size()].plane_cloud_, scene_handler, pname.str(), v3);

                /*pname << "chull";
                vis.addPolygonMesh (*planes_found[i - models->size()].convex_hull_, pname.str(), v3);*/
            }
        }

        if (PLAY_)
        {
          vis.spinOnce (500.f, true);
        }
        else
        {
          vis.spin ();
        }
    }

    timing.writeToFile(TIMING_OUTPUT_FILE_);
}

typedef pcl::ReferenceFrame RFType;

int CG_SIZE_ = 3;
float CG_THRESHOLD_ = 0.005f;

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

    pcl::console::parse_argument (argc, argv, "-source_models", source_models);


    if(source_models == 0)
    {
        /*boost::shared_ptr < faat_pcl::rec_3d_framework::ModelOnlySource<pcl::PointXYZ, pcl::PointXYZ>
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
        recognizeAndVisualize<pcl::PointXYZ, pcl::PointXYZ> (source, pcd_file);*/
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
