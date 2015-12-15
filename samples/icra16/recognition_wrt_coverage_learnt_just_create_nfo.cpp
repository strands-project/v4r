// --in_turntable /home/thomas/Documents/icra16/turntable_models --in_learnt /home/thomas/Documents/icra16/learnt_models/controlled -t /home/thomas/Documents/icra16/keyframes/controlled_ba -i /home/thomas/Documents/icra16/keyframes/controlled_run.nfo --do_shot 1 --do_sift 1 --knn_sift 3 -z 2 --knn_shot 1 --cg_size_thresh 5 -r 0.1 -o /home/thomas/icra_rec_partial_controlled --step_size_start_training_views_ 5 --step_size_test_views_ 4

#define BOOST_NO_SCOPED_ENUMS
#define BOOST_NO_CXX11_SCOPED_ENUMS

#include <v4r_config.h>
#include <v4r/common/faat_3d_rec_framework_defines.h>
#include <v4r/common/noise_model_based_cloud_integration.h>
#include <v4r/common/noise_models.h>
#include <v4r/common/miscellaneous.h>
#include <v4r/features/sift_local_estimator.h>
#include <v4r/features/shot_local_estimator_omp.h>
#include <v4r/io/eigen.h>
#include <v4r/io/filesystem.h>
#include <v4r/recognition/ghv.h>
#include <v4r/recognition/local_recognizer.h>
#include <v4r/recognition/multi_pipeline_recognizer.h>
#include <v4r/recognition/registered_views_source.h>
#include <v4r/recognition/recognizer.h>

#include <boost/any.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <pcl/filters/passthrough.h>
#include <pcl/point_cloud.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <time.h>
#include <stdlib.h>
#include <glog/logging.h>

#ifndef HAVE_SIFTGPU
#include <v4r/features/opencv_sift_local_estimator.h>
#endif

namespace po = boost::program_options;
namespace bf = boost::filesystem;

// some helper functions
bool hasEnding (std::string const &fullString, std::string const &ending) {
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

struct view {
    std::string name;
    int id;
};

struct by_id {
    bool operator()(view const &a, view const &b) {
        return a.id < b.id;
    }
};

class EvalPartialModelRecognizer
{

private:
    typedef pcl::PointXYZRGB PointT;
    typedef v4r::Model<PointT> ModelT;
    typedef boost::shared_ptr<ModelT> ModelTPtr;
    typedef pcl::Histogram<128> FeatureT;


    std::string in_learnt_;
    std::string in_turntable_;
    std::string test_dir_;
    std::string info_file_; /// @brief which describes which test folder belongs to which object (format: "test_folder_id patrol_run_id object_name\n")
    std::string out_tmp_;
    std::string out_results_;
    std::vector<std::string> model_list;
    bool visualize_;
    size_t step_size_start_training_views_, step_size_test_views_;

    // recognition stuff
    v4r::GHV<PointT, PointT>::Parameter paramGHV;
    v4r::GraphGeometricConsistencyGrouping<PointT, PointT>::Parameter paramGgcg;
    v4r::LocalRecognitionPipeline<flann::L1, PointT, FeatureT >::Parameter paramLocalRecSift;
    v4r::LocalRecognitionPipeline<flann::L1, PointT, pcl::Histogram<352> >::Parameter paramLocalRecShot;
    v4r::MultiRecognitionPipeline<PointT>::Parameter paramMultiPipeRec;
    v4r::SHOTLocalEstimationOMP<PointT, pcl::Histogram<352> >::Parameter paramLocalEstimator;
    boost::shared_ptr<v4r::MultiRecognitionPipeline<PointT> > rr_;
    boost::shared_ptr < v4r::GraphGeometricConsistencyGrouping<PointT, PointT> > gcg_alg_;

private:


    /**
     * @brief sort views because if sort was alpabetically, the order would be cloud_0.pcd, cloud_1.pcd, cloud_10.pcd, cloud_11.pcd,... (the camera tracker should now add leading zeros to avoid this but the old dataset does not)
     * @param views to be sorted
     */
    static std::vector<std::string>
    sortViews(const std::vector<std::string> &views)
    {
        std::vector<view> training_views_sorted_tmp (views.size());

        for (size_t i=0; i<views.size(); i++)
        {
            training_views_sorted_tmp[i].name = views[i];
            std::vector<std::string> string_parts;
            boost::split (string_parts, views[i], boost::is_any_of ("_"));
            std::istringstream buffer(string_parts[1]);
            buffer >> training_views_sorted_tmp[i].id;
        }

        std::sort(training_views_sorted_tmp.begin(), training_views_sorted_tmp.end(), by_id());

        std::vector<std::string> training_views_sorted (views.size());
        for (size_t i=0; i<views.size(); i++)
        {
            training_views_sorted[i] = training_views_sorted_tmp[i].name;
        }
        std::cout << "sorted";
        return training_views_sorted;
    }


public:
    class Parameter
    {
    public:
        int normal_method_;
        float vox_res_;
        double chop_z_;
        bool do_sift_;
        bool do_shot_;
        bool do_ourcvfh_;

        Parameter
        ( int normal_method = 2,
          float vox_res = 0.03f,
          float chop_z = std::numeric_limits<float>::max(),
          bool do_sift = true,
          bool do_shot = true,
          bool do_ourcvfh = false,
          float resolution = 0.005f)
            : normal_method_ (normal_method),
              vox_res_ (vox_res),
              chop_z_ (chop_z),
              do_sift_ (do_sift),
              do_shot_ (do_shot),
              do_ourcvfh_ (do_ourcvfh)
        {}
    }param_;

    EvalPartialModelRecognizer(const Parameter &p = Parameter())
    {
        param_ = p;
        out_tmp_ = "/tmp/delete_me/";
        out_results_ = "/tmp/icra16_rec_results";

        paramGgcg.max_time_allowed_cliques_comptutation_ = 100;
        paramLocalRecSift.use_cache_ = paramLocalRecShot.use_cache_ = true;
        paramLocalRecSift.save_hypotheses_ = paramLocalRecShot.save_hypotheses_ = true;
        paramLocalRecShot.kdtree_splits_ = 128;
        visualize_ = false;
        step_size_start_training_views_ = 1;
        step_size_test_views_ = 1;
    }


    void
    initialize(int argc, char ** argv)
    {
        po::options_description desc("Evaluation of recognition of partial models\n**Allowed options");
        desc.add_options()
                ("help,h", "produce help message")
                ("in_turntable", po::value<std::string>(&in_turntable_)->required(), "input model directory")
                ("test_dir,t", po::value<std::string>(&test_dir_)->required(), "test directory")
                ("info_file,i", po::value<std::string>(&info_file_)->required(), "which describes which test folder belongs to which object (format: \"test_folder_id patrol_run_id object_name\\n\")")
                ("out_dir,o", po::value<std::string>(&out_results_)->default_value(out_results_), "output directory")
                ("step_size_start_training_views_", po::value<size_t>(&step_size_start_training_views_)->default_value(step_size_start_training_views_), "step size taking from one bunch of partial views to the next")
                ("step_size_test_views_", po::value<size_t>(&step_size_test_views_)->default_value(step_size_test_views_), "step size taking from one bunch of partial views to the next")
                ("in_learnt", po::value<std::string>(&in_learnt_)->required(), "input model directory procuded by dynamic object learning")
                ("tmp_dir", po::value<std::string>(&out_tmp_)->default_value(out_tmp_), "directory where temporary training database is stored")

                ("do_sift", po::value<bool>(&param_.do_sift_)->default_value(param_.do_sift_), "if true, generates hypotheses using SIFT (visual texture information)")
                ("do_shot", po::value<bool>(&param_.do_shot_)->default_value(param_.do_shot_), "if true, generates hypotheses using SHOT (local geometrical properties)")
                ("do_ourcvfh", po::value<bool>(&param_.do_ourcvfh_)->default_value(param_.do_ourcvfh_), "if true, generates hypotheses using OurCVFH (global geometrical properties, requires segmentation!)")
                ("knn_sift", po::value<int>(&paramLocalRecSift.knn_)->default_value(paramLocalRecSift.knn_), "sets the number k of matches for each extracted SIFT feature to its k nearest neighbors")
                ("knn_shot", po::value<int>(&paramLocalRecShot.knn_)->default_value(paramLocalRecShot.knn_), "sets the number k of matches for each extracted SHOT feature to its k nearest neighbors")
                ("icp_iterations", po::value<int>(&paramMultiPipeRec.icp_iterations_)->default_value(paramMultiPipeRec.icp_iterations_), "number of icp iterations. If 0, no pose refinement will be done")
                ("icp_type", po::value<int>(&paramMultiPipeRec.icp_type_)->default_value(paramMultiPipeRec.icp_type_), "defines the icp method being used for pose refinement (0... regular ICP with CorrespondenceRejectorSampleConsensus, 1... crops point cloud of the scene to the bounding box of the model that is going to be refined)")
                ("max_corr_distance", po::value<double>(&paramMultiPipeRec.max_corr_distance_)->default_value(paramMultiPipeRec.max_corr_distance_,  boost::str(boost::format("%.2e") % paramMultiPipeRec.max_corr_distance_)), "defines the margin for the bounding box used when doing pose refinement with ICP of the cropped scene to the model")
                ("merge_close_hypotheses", po::value<bool>(&paramMultiPipeRec.merge_close_hypotheses_)->default_value(paramMultiPipeRec.merge_close_hypotheses_), "if true, close correspondence clusters (object hypotheses) of the same object model are merged together and this big cluster is refined")
                ("merge_close_hypotheses_dist", po::value<double>(&paramMultiPipeRec.merge_close_hypotheses_dist_)->default_value(paramMultiPipeRec.merge_close_hypotheses_dist_, boost::str(boost::format("%.2e") % paramMultiPipeRec.merge_close_hypotheses_dist_)), "defines the maximum distance of the centroids in meter for clusters to be merged together")
                ("merge_close_hypotheses_angle", po::value<double>(&paramMultiPipeRec.merge_close_hypotheses_angle_)->default_value(paramMultiPipeRec.merge_close_hypotheses_angle_, boost::str(boost::format("%.2e") % paramMultiPipeRec.merge_close_hypotheses_angle_) ), "defines the maximum angle in degrees for clusters to be merged together")
                ("chop_z,z", po::value<double>(&param_.chop_z_)->default_value(param_.chop_z_, boost::str(boost::format("%.2e") % param_.chop_z_) ), "points with z-component higher than chop_z_ will be ignored (low chop_z reduces computation time and false positives (noise increase with z)")
                ("cg_size_thresh", po::value<size_t>(&paramGgcg.gc_threshold_)->default_value(paramGgcg.gc_threshold_), "Minimum cluster size. At least 3 correspondences are needed to compute the 6DOF pose ")
                ("cg_size,c", po::value<double>(&paramGgcg.gc_size_)->default_value(paramGgcg.gc_size_, boost::str(boost::format("%.2e") % paramGgcg.gc_size_) ), "Resolution of the consensus set used to cluster correspondences together ")
                ("cg_ransac_threshold", po::value<double>(&paramGgcg.ransac_threshold_)->default_value(paramGgcg.ransac_threshold_, boost::str(boost::format("%.2e") % paramGgcg.ransac_threshold_) ), " ")
                ("cg_dist_for_clutter_factor", po::value<double>(&paramGgcg.dist_for_cluster_factor_)->default_value(paramGgcg.dist_for_cluster_factor_, boost::str(boost::format("%.2e") % paramGgcg.dist_for_cluster_factor_) ), " ")
                ("cg_max_taken", po::value<size_t>(&paramGgcg.max_taken_correspondence_)->default_value(paramGgcg.max_taken_correspondence_), " ")
                ("cg_max_time_for_cliques_computation", po::value<double>(&paramGgcg.max_time_allowed_cliques_comptutation_)->default_value(100.0, "100.0"), " if grouping correspondences takes more processing time in milliseconds than this defined value, correspondences will be no longer computed by this graph based approach but by the simpler greedy correspondence grouping algorithm")
                ("cg_dot_distance", po::value<double>(&paramGgcg.thres_dot_distance_)->default_value(paramGgcg.thres_dot_distance_, boost::str(boost::format("%.2e") % paramGgcg.thres_dot_distance_) ) ,"")
                ("cg_use_graph", po::value<bool>(&paramGgcg.use_graph_)->default_value(paramGgcg.use_graph_), " ")
                ("hv_clutter_regularizer", po::value<double>(&paramGHV.clutter_regularizer_)->default_value(paramGHV.clutter_regularizer_, boost::str(boost::format("%.2e") % paramGHV.clutter_regularizer_) ), "The penalty multiplier used to penalize unexplained scene points within the clutter influence radius <i>radius_neighborhood_clutter_</i> of an explained scene point when they belong to the same smooth segment.")
                ("hv_color_sigma_ab", po::value<double>(&paramGHV.color_sigma_ab_)->default_value(paramGHV.color_sigma_ab_, boost::str(boost::format("%.2e") % paramGHV.color_sigma_ab_) ), "allowed chrominance (AB channel of LAB color space) variance for a point of an object hypotheses to be considered explained by a corresponding scene point (between 0 and 1, the higher the fewer objects get rejected)")
                ("hv_color_sigma_l", po::value<double>(&paramGHV.color_sigma_l_)->default_value(paramGHV.color_sigma_l_, boost::str(boost::format("%.2e") % paramGHV.color_sigma_l_) ), "allowed illumination (L channel of LAB color space) variance for a point of an object hypotheses to be considered explained by a corresponding scene point (between 0 and 1, the higher the fewer objects get rejected)")
                ("hv_detect_clutter", po::value<bool>(&paramGHV.detect_clutter_)->default_value(paramGHV.detect_clutter_), " ")
                ("hv_duplicity_cm_weight", po::value<double>(&paramGHV.w_occupied_multiple_cm_)->default_value(paramGHV.w_occupied_multiple_cm_, boost::str(boost::format("%.2e") % paramGHV.w_occupied_multiple_cm_) ), " ")
                ("hv_histogram_specification", po::value<bool>(&paramGHV.use_histogram_specification_)->default_value(paramGHV.use_histogram_specification_), " ")
                ("hv_hyp_penalty", po::value<double>(&paramGHV.active_hyp_penalty_)->default_value(paramGHV.active_hyp_penalty_, boost::str(boost::format("%.2e") % paramGHV.active_hyp_penalty_) ), " ")
                ("hv_ignore_color", po::value<bool>(&paramGHV.ignore_color_even_if_exists_)->default_value(paramGHV.ignore_color_even_if_exists_), " ")
                ("hv_initial_status", po::value<bool>(&paramGHV.initial_status_)->default_value(paramGHV.initial_status_), "sets the initial activation status of each hypothesis to this value before starting optimization. E.g. If true, all hypotheses will be active and the cost will be optimized from that initial status.")
                ("hv_color_space", po::value<int>(&paramGHV.color_space_)->default_value(paramGHV.color_space_), "specifies the color space being used for verification (0... LAB, 1... RGB, 2... Grayscale,  3,4,5,6... ?)")
                ("hv_inlier_threshold", po::value<double>(&paramGHV.inliers_threshold_)->default_value(paramGHV.inliers_threshold_, boost::str(boost::format("%.2e") % paramGHV.inliers_threshold_) ), "Represents the maximum distance between model and scene points in order to state that a scene point is explained by a model point. Valid model points that do not have any corresponding scene point within this threshold are considered model outliers")
                ("hv_occlusion_threshold", po::value<double>(&paramGHV.occlusion_thres_)->default_value(paramGHV.occlusion_thres_, boost::str(boost::format("%.2e") % paramGHV.occlusion_thres_) ), "Threshold for a point to be considered occluded when model points are back-projected to the scene ( depends e.g. on sensor noise)")
                ("hv_optimizer_type", po::value<int>(&paramGHV.opt_type_)->default_value(paramGHV.opt_type_), "defines the optimization methdod. 0: Local search (converges quickly, but can easily get trapped in local minima), 1: Tabu Search, 4; Tabu Search + Local Search (Replace active hypotheses moves), else: Simulated Annealing")
                ("hv_radius_clutter", po::value<double>(&paramGHV.radius_neighborhood_clutter_)->default_value(paramGHV.radius_neighborhood_clutter_, boost::str(boost::format("%.2e") % paramGHV.radius_neighborhood_clutter_) ), "defines the maximum distance between two points to be checked for label consistency")
                ("hv_radius_normals", po::value<double>(&paramGHV.radius_normals_)->default_value(paramGHV.radius_normals_, boost::str(boost::format("%.2e") % paramGHV.radius_normals_) ), " ")
                ("hv_regularizer,r", po::value<double>(&paramGHV.regularizer_)->default_value(paramGHV.regularizer_, boost::str(boost::format("%.2e") % paramGHV.regularizer_) ), "represents a penalty multiplier for model outliers. In particular, each model outlier associated with an active hypothesis increases the global cost function.")
                ("hv_plane_method", po::value<int>(&paramGHV.plane_method_)->default_value(paramGHV.plane_method_), "defines which method to use for plane extraction (if add_planes_ is true). 0... Multiplane Segmentation, 1... ClusterNormalsForPlane segmentation")
                ("hv_add_planes", po::value<bool>(&paramGHV.add_planes_)->default_value(paramGHV.add_planes_), "if true, adds planes as possible hypotheses (slower but decreases false positives especially for planes detected as flat objects like books)")
                ("hv_plane_inlier_distance", po::value<double>(&paramGHV.plane_inlier_distance_)->default_value(paramGHV.plane_inlier_distance_, boost::str(boost::format("%.2e") % paramGHV.plane_inlier_distance_) ), "Maximum inlier distance for plane clustering")
                ("hv_plane_thrAngle", po::value<double>(&paramGHV.plane_thrAngle_)->default_value(paramGHV.plane_thrAngle_, boost::str(boost::format("%.2e") % paramGHV.plane_thrAngle_) ), "Threshold of normal angle in degree for plane clustering")
                ("hv_use_supervoxels", po::value<bool>(&paramGHV.use_super_voxels_)->default_value(paramGHV.use_super_voxels_), "If true, uses supervoxel clustering to detect smoothness violations")
                ("hv_min_plane_inliers", po::value<size_t>(&paramGHV.min_plane_inliers_)->default_value(paramGHV.min_plane_inliers_), "a planar cluster is only added as plane if it has at least min_plane_inliers_ points")
                ("normal_method,n", po::value<int>(&param_.normal_method_)->default_value(param_.normal_method_), "chosen normal computation method of the V4R library")
                ("visualize,v", po::bool_switch(&visualize_), "turn visualization on")
                ;
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc << std::endl;
            return;
        }

        try
        {
            po::notify(vm);
        }
        catch(std::exception& e)
        {
            std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl;
            return;
        }

        paramLocalRecSift.normal_computation_method_ = paramLocalRecShot.normal_computation_method_ =
                paramMultiPipeRec.normal_computation_method_ = paramLocalEstimator.normal_computation_method_ = param_.normal_method_;

        // writing parameters to file
        v4r::io::createDirIfNotExist( out_results_ );
        ofstream param_file( (out_results_ + "/param.nfo").c_str());
        for(const auto& it : vm)
        {
            param_file << "--" << it.first << " ";

            auto& value = it.second.value();
            if (auto v = boost::any_cast<double>(&value))
                param_file << std::setprecision(3) << *v;
            else if (auto v = boost::any_cast<std::string>(&value))
                param_file << *v;
            else if (auto v = boost::any_cast<bool>(&value))
                param_file << *v;
            else if (auto v = boost::any_cast<int>(&value))
                param_file << *v;
            else if (auto v = boost::any_cast<size_t>(&value))
                param_file << *v;
            else
                param_file << "error";

            param_file << " ";
        }
        param_file.close();
    }

    /**
     * @brief This function prepares the training directory such that one object model is the one being split into a subset of training views (i.e. to create a partial model of it).
     * The remaining object models are copied (symlink) to the output training directory.
     * For the partial model, a visibility percentage is computed by downsampling both the full and partial model by a voxel grid filter of same resolution
     * and counting the number of filtered points.
     */
    void
    eval()
    {
        v4r::NMBasedCloudIntegration<pcl::PointXYZRGB>::Parameter nm_int_param;
        nm_int_param.final_resolution_ = 0.002f;
        nm_int_param.min_points_per_voxel_ = 1;
        nm_int_param.min_weight_ = 0.5f;
        nm_int_param.octree_resolution_ = 0.005f;
        nm_int_param.threshold_ss_ = 0.01f;

        v4r::noise_models::NguyenNoiseModel<pcl::PointXYZRGB>::Parameter nm_param;

        pcl::visualization::PCLVisualizer vis;

        // iterate through all models and evaluate one after another
        v4r::io::getFilesInDirectory(in_turntable_ + "/models", model_list, "", ".*.pcd", false);
        std::sort(model_list.begin(), model_list.end());
        for (size_t replaced_m_id=0; replaced_m_id<model_list.size(); replaced_m_id++)
        {
            const std::string replaced_model = model_list [replaced_m_id];

            bf::remove_all(bf::path(out_tmp_ + "/models/"));
            v4r::io::createDirIfNotExist(out_tmp_ + "/models/");

            // COMPUTE COVERAGE OF TURNTABLE OBJECT
            std::vector<std::string> training_views;
            v4r::io::getFilesInDirectory(in_turntable_ + "/training_data/" + replaced_model, training_views, "", ".*cloud.*.pcd");
            std::sort(training_views.begin(), training_views.end());

            std::vector<pcl::PointCloud<PointT>::Ptr > training_clouds ( training_views.size() );
            std::vector<pcl::PointCloud<pcl::Normal>::Ptr > normal_clouds ( training_views.size() );
            std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > cameras ( training_views.size() );
            std::vector<std::vector<int> > obj_indices ( training_views.size() );
            std::vector<pcl::PointCloud<IndexPoint>::Ptr > obj_indices_cloud ( training_views.size() );
            std::vector<std::vector<float> > weights ( training_views.size() );
            std::vector<std::vector<float> > sigmas ( training_views.size() );
            std::vector<std::string> pose_fns = training_views;
            std::vector<std::string> obj_indices_fns = training_views;

            const size_t num_training_views = training_views.size();

            for(size_t v_id = 0; v_id < num_training_views; v_id++)
            {
                const std::string training_view = in_turntable_ + "/training_data/" + replaced_model + "/" + training_views[v_id];

                training_clouds[v_id].reset( new pcl::PointCloud<PointT>);
                normal_clouds[v_id].reset( new pcl::PointCloud<pcl::Normal>);
                pcl::io::loadPCDFile ( training_view, *training_clouds[v_id] );
                boost::replace_last (pose_fns[v_id], "cloud_", "pose_");
                boost::replace_last (pose_fns[v_id], ".pcd", ".txt");
                v4r::io::readMatrixFromFile ( in_turntable_ + "/training_data/" + replaced_model + "/" + pose_fns[v_id], cameras[v_id]);

                boost::replace_last (obj_indices_fns[v_id], "cloud_", "object_indices_");

                obj_indices_cloud[v_id].reset (new pcl::PointCloud<IndexPoint>);
                pcl::io::loadPCDFile ( in_turntable_ + "/training_data/" + replaced_model + "/" + obj_indices_fns[v_id], *obj_indices_cloud[v_id]);
                obj_indices[v_id].resize(obj_indices_cloud[v_id]->points.size());
                for(size_t kk=0; kk < obj_indices_cloud[v_id]->points.size(); kk++)
                    obj_indices[v_id][kk] = obj_indices_cloud[v_id]->points[kk].idx;

                v4r::computeNormals<PointT>( training_clouds[v_id], normal_clouds[v_id], param_.normal_method_);

                v4r::noise_models::NguyenNoiseModel<PointT> nm (nm_param);
                nm.setInputCloud ( training_clouds[v_id] );
                nm.setInputNormals ( normal_clouds[v_id] );
                nm.setLateralSigma(0.001);
                nm.setMaxAngle(60.f);
                nm.setUseDepthEdges(true);
                nm.compute();
                nm.getWeights( weights[ v_id ] );
                sigmas[ v_id ] = nm.getSigmas();
            }

            // just for computing the total number of points visible if all training view were considered
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr octree_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            v4r::NMBasedCloudIntegration<pcl::PointXYZRGB> nmIntegration (nm_int_param);
            nmIntegration.setInputClouds( training_clouds );
            nmIntegration.setWeights( weights );
            nmIntegration.setSigmas( sigmas );
            nmIntegration.setTransformations( cameras );
            nmIntegration.setInputNormals( normal_clouds );
            nmIntegration.setIndices( obj_indices );
            nmIntegration.compute( octree_cloud );
            pcl::PointCloud<PointT> cloud_filtered;
            pcl::VoxelGrid<PointT > sor;
            sor.setInputCloud (octree_cloud);
            sor.setLeafSize ( param_.vox_res_, param_.vox_res_, param_.vox_res_);
            sor.filter ( cloud_filtered );
            size_t total_points = cloud_filtered.points.size();
            size_t total_points_oc = octree_cloud->points.size();


            // now we know how much the object occupies when modelled on a turntable
            // get all test sequences that belong to the object model
            std::vector<std::string> runs_with_replaced_model;
            ifstream info(info_file_.c_str());
            std::string test_id, patrol_run_id, object_id;
            while (info >> test_id >> patrol_run_id >> object_id) {
                if (hasEnding(object_id, replaced_model)) {
                    runs_with_replaced_model.push_back( test_id );
                }
            }

            for(size_t m_run_id=0; m_run_id < runs_with_replaced_model.size(); m_run_id++)
            {
                const std::string search_path = in_learnt_ + "/training_data/" + runs_with_replaced_model[m_run_id] + ".pcd";
                training_views.clear();

                if(!v4r::io::existsFolder(search_path)) // in case that object modelling for a patrol run was not succesful skip it
                    continue;

                v4r::io::getFilesInDirectory(search_path, training_views, "", ".*cloud.*.pcd");
                training_views = sortViews(training_views);

                // read data from all training views
                training_clouds.resize ( training_views.size() );
                normal_clouds.resize ( training_views.size() );
                cameras.resize ( training_views.size() );
                obj_indices.resize ( training_views.size() );
                obj_indices_cloud.resize ( training_views.size() );
                weights.resize ( training_views.size() );
                sigmas.resize ( training_views.size() );
                pose_fns = training_views;
                obj_indices_fns = training_views;

                const size_t num_training_views = training_views.size();

                for(size_t v_id = 0; v_id < num_training_views; v_id++)
                {
                    const std::string training_view = search_path + "/" + training_views[v_id];

                    training_clouds[v_id].reset( new pcl::PointCloud<PointT>);
                    normal_clouds[v_id].reset( new pcl::PointCloud<pcl::Normal>);
                    pcl::io::loadPCDFile ( training_view, *training_clouds[v_id] );
                    boost::replace_last (pose_fns[v_id], "cloud_", "pose_");
                    boost::replace_last (pose_fns[v_id], ".pcd", ".txt");
                    v4r::io::readMatrixFromFile ( search_path + "/" + pose_fns[v_id], cameras[v_id]);

                    boost::replace_last (obj_indices_fns[v_id], "cloud_", "object_indices_");

                    obj_indices_cloud[v_id].reset (new pcl::PointCloud<IndexPoint>);
                    pcl::io::loadPCDFile ( search_path + "/" + obj_indices_fns[v_id], *obj_indices_cloud[v_id]);
                    obj_indices[v_id].resize(obj_indices_cloud[v_id]->points.size());
                    for(size_t kk=0; kk < obj_indices_cloud[v_id]->points.size(); kk++)
                        obj_indices[v_id][kk] = obj_indices_cloud[v_id]->points[kk].idx;

                    v4r::computeNormals<PointT>( training_clouds[v_id], normal_clouds[v_id], param_.normal_method_);

                    v4r::noise_models::NguyenNoiseModel<PointT> nm (nm_param);
                    nm.setInputCloud ( training_clouds[v_id] );
                    nm.setInputNormals ( normal_clouds[v_id] );
                    nm.setLateralSigma(0.001);
                    nm.setMaxAngle(60.f);
                    nm.setUseDepthEdges(true);
                    nm.compute();
                    nm.getWeights( weights[ v_id ] );
                    sigmas[ v_id ] = nm.getSigmas();
                }

                size_t eval_id = 0;

                // now create partial model from successive training views
                for (size_t num_used_v = num_training_views-1; num_used_v > 0; num_used_v--)
                {
                    std::vector<pcl::PointCloud<PointT>::Ptr > training_clouds_used ( num_used_v );
                    std::vector<pcl::PointCloud<pcl::Normal>::Ptr > normal_clouds_used ( num_used_v );
                    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > cameras_used ( num_used_v );
                    std::vector<std::vector<int> > obj_indices_used ( num_used_v );
                    std::vector<std::vector<float> > weights_used ( num_used_v );
                    std::vector<std::vector<float> > sigmas_used ( num_used_v );

                    //                for (size_t start_v = 0; start_v < num_training_views; start_v+=step_size_start_training_views_)
                    size_t start_v=0;
                    {
                        for (size_t v_id_rel=0; v_id_rel<num_used_v; v_id_rel++)
                        {
                            size_t v_id = ( start_v + v_id_rel ) % num_training_views;

                            training_clouds_used [ v_id_rel ] = training_clouds [ v_id ];
                            normal_clouds_used [ v_id_rel ] = normal_clouds [ v_id ];
                            cameras_used [ v_id_rel ] = cameras [ v_id ];
                            obj_indices_used [ v_id_rel ] = obj_indices [ v_id ];
                            weights_used [ v_id_rel ] = weights [ v_id ];
                            sigmas_used [ v_id_rel ] = sigmas [ v_id ];


                           // check for double walls
                            pcl::PointCloud<PointT>::Ptr current_cloud = training_clouds_used [v_id];

                            float f = 525.0f;
                            float cx = 319.5f;
                            float cy = 239.5f;

                            for(size_t compare_cloud_id=0; compare_cloud_id < v_id_rel; compare_cloud_id++)
                            {
                                const pcl::PointCloud<PointT>::Ptr compare_cloud (new pcl::PointCloud<PointT>);

                                Eigen::Matrix4f tf2current_cloud = cameras_used[ v_id].inverse() * cameras_used [ compare_cloud_id ];
                                pcl::transformPointCloud(*training_clouds_used[ compare_cloud_id ], *compare_cloud, tf2current_cloud);

                                vis.removeAllPointClouds();
                                vis.addPointCloud(compare_cloud, "cloud_A");
                                vis.addPointCloud(current_cloud, "cloud_B");
                                vis.spin();

                                for(size_t pt_id_b=0; pt_id_b<compare_cloud->points.size(); pt_id_b++)
                                {
                                    const PointT &pt_b = compare_cloud->points[pt_id_b];

                                    if (!pcl::isFinite(pt_b))
                                        continue;

                                    float x = pt_b.x;
                                    float y = pt_b.y;
                                    float z = pt_b.z;
                                    int u = static_cast<int> (f * x / z + cx);
                                    int v = static_cast<int> (f * y / z + cy);

                                    if(u<0 ||  u>=current_cloud->width || v<0 || v>=current_cloud->height)
                                        continue;

                                    PointT &pt_a = current_cloud->at(u,v);

                                    if( ( (pt_b.z - pt_a.z) < 0.05f) && ( (pt_b.z - pt_a.z) > 0) )
                                    {
                                        pt_a.x = std::numeric_limits<float>::quiet_NaN();
                                        pt_a.y = std::numeric_limits<float>::quiet_NaN();
                                        pt_a.z = std::numeric_limits<float>::quiet_NaN();
                                        sigmas_used[v_id][u*current_cloud->width + v] = std::numeric_limits<float>::quiet_NaN();
                                    }

                                }
                            }
                        }

                        nmIntegration.setInputClouds( training_clouds_used );
                        nmIntegration.setWeights( weights_used );
                        nmIntegration.setSigmas( sigmas_used );
                        nmIntegration.setTransformations( cameras_used );
                        nmIntegration.setInputNormals( normal_clouds_used );
                        nmIntegration.setIndices( obj_indices_used );
                        nmIntegration.compute( octree_cloud );

                        sor.setInputCloud (octree_cloud);
                        sor.setLeafSize ( param_.vox_res_, param_.vox_res_, param_.vox_res_);
                        sor.filter ( cloud_filtered );
                        size_t visible_pts = cloud_filtered.points.size();

                        pcl::io::savePCDFileBinary(out_tmp_ + "/models/" + runs_with_replaced_model[m_run_id] + ".pcd", *octree_cloud);

                        std::cout << cloud_filtered.points.size() << " / " << total_points << " visible "
                                  << static_cast<float>(visible_pts) / total_points << " "
                                  << static_cast<float>(octree_cloud->points.size()) / total_points_oc <<std::endl;

                        // get all test sequences that belong to the object model
                        std::vector<std::string> test_files;
                        ifstream info(info_file_.c_str());
                        std::string test_id, patrol_run_id, object_id;
                        while (info >> test_id >> patrol_run_id >> object_id) {
                            if (hasEnding(object_id, replaced_model)) {
                                std::cout << object_id << std::endl;
                                test_files.push_back( test_dir_ + "/" + test_id);
                            }
                        }

                        std::stringstream result_dir_tmp;
                        result_dir_tmp << out_results_ << "/" << replaced_model << "/" << runs_with_replaced_model[m_run_id] << "/" << eval_id;
                        v4r::io::createDirIfNotExist( result_dir_tmp.str() );
                        const std::string model_info_fn = result_dir_tmp.str() + "/model_info.txt";
                        ofstream f( model_info_fn.c_str());
                        f << replaced_model << " " << num_used_v << " " << num_training_views << " " <<
                             visible_pts << " " << total_points <<std::endl;
                        f.close();

                        eval_id++;
                    }
                }
            }
        }
    }
};


int
main (int argc, char ** argv)
{
    srand (time(NULL));
    google::InitGoogleLogging(argv[0]);
    EvalPartialModelRecognizer r;
    r.initialize(argc,argv);
    r.eval();

    return 0;
}
