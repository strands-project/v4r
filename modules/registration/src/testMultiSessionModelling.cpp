#include <v4r/io/filesystem_utils.h>
#include <v4r/registration/MultiSessionModelling.h>
#include <v4r/registration/FeatureBasedRegistration.h>
#include <v4r/registration/MvLMIcp.h>
#include <v4r/common/noise_model_based_cloud_integration.h>
#include <v4r/common/noise_models.h>
#include <v4r/common/pcl_opencv.h>
#include <v4r/registration/StablePlanesRegistration.h>

#include <pcl/console/parse.h>
#include <pcl/common/common.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <boost/algorithm/string.hpp>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <opencv2/opencv.hpp>
#include <pcl/filters/statistical_outlier_removal.h>

struct IndexPoint
{
    int idx;
};

POINT_CLOUD_REGISTER_POINT_STRUCT (IndexPoint,
                                   (int, idx, idx)
                                   )

typedef pcl::PointXYZRGB PointInT;

void computeImageGradients(pcl::PointCloud<PointInT>::Ptr & cloud,
                           std::vector<float> & image_gradients)
{

    cv::Mat_<cv::Vec3b> src; // = cv::Mat_<cv::Vec3b>(cloud->height, cloud->width);
    PCLOpenCV::ConvertPCLCloud2Image<PointInT>(cloud, src);

    cv::Mat src_gray;
    cv::Mat grad;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

    cv::GaussianBlur( src, src, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT );

    /// Convert it to gray
    cv::cvtColor( src, src_gray, CV_RGB2GRAY );

    /// Generate grad_x and grad_y
    cv::Mat grad_x, grad_y;
    cv::Mat abs_grad_x, abs_grad_y;

    /// Gradient X
    cv::Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );
    cv::convertScaleAbs( grad_x, abs_grad_x );

    /// Gradient Y
    cv::Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );
    cv::convertScaleAbs( grad_y, abs_grad_y );

    /// Total Gradient (approximate)
    cv::addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

    image_gradients.resize(cloud->height * cloud->width);
    for(int r=0; r < (unsigned)cloud->height; r++)
    {
        for(int c=0; c < (unsigned)cloud->width; c++)
        {
            float g = grad.at<float>(r,c) / 255.f;
            assert(g >= 0.f && g <= 1.f);
            image_gradients[r * cloud->width + c] = g;
        }
    }
}

void computePlane(pcl::PointCloud<PointInT>::Ptr & cloud,
                  Eigen::Vector3f & normal,
                  float z_dist = 1)
{


    pcl::PointCloud<PointInT>::Ptr cloud_pass (new pcl::PointCloud<PointInT> ());

    pcl::PassThrough<PointInT> filter_pass;
    filter_pass.setInputCloud(cloud);
    filter_pass.setFilterLimits(0,z_dist);
    filter_pass.setFilterFieldName("z");
    filter_pass.filter(*cloud_pass);

    filter_pass.setInputCloud(cloud_pass);
    filter_pass.setFilterLimits(-0.3,0.3);
    filter_pass.setFilterFieldName("x");
    filter_pass.filter(*cloud_pass);


    float vx_size = 0.005f;
    pcl::PointCloud<PointInT>::Ptr cloud_voxel (new pcl::PointCloud<PointInT> ());

    pcl::VoxelGrid<PointInT> filter;
    filter.setInputCloud(cloud_pass);
    filter.setDownsampleAllData(true);
    filter.setLeafSize(vx_size,vx_size,vx_size);
    filter.filter(*cloud_voxel);

    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);

    pcl::SACSegmentation<PointInT> seg;
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setDistanceThreshold (0.01);

    seg.setInputCloud (cloud_voxel);
    seg.segment (*inliers, *coefficients);

    normal = Eigen::Vector3f(coefficients->values[0],coefficients->values[1],coefficients->values[2]);
    //flip if necessary
    Eigen::Vector3f c(0,0,1);
    if(normal.dot(c) < 0)
        normal = normal * -1.f;

    /*pcl::visualization::PCLVisualizer plane_vis("plane");
    plane_vis.addPointCloud<PointInT>(cloud_voxel);

    pcl::PointCloud<PointInT>::Ptr cloud_plane (new pcl::PointCloud<PointInT> ());
    pcl::copyPointCloud(*cloud_voxel, *inliers, *cloud_plane);

    pcl::visualization::PointCloudColorHandlerCustom<PointInT> handler(cloud_plane, 255, 0, 0);
    plane_vis.addPointCloud(cloud_plane, handler, "plane");
    plane_vis.spin();*/

}

int
main (int argc, char ** argv)
{
    std::string directory = "";
    std::string pose_str = "pose";

    bool do_cg = false;
    float inlier_threshold = 0.015f;
    int gc_threshold = 9;
    float z_dist = 1;
    bool use_stable_planes = true;
    bool use_sift_features = true;
    int max_iterations = 10;
    float mv_vx_size = 0.003f;
    std::string export_to = "";
    bool use_mv_icp_normals_ = true;
    float threshold_ss = 0.003f;
    float model_resolution = 0.001f;
    float min_weight_ = 0.9f;
    float max_angle = 60.f;
    float lateral_sigma = 0.0015f;
    bool use_plane_for_nw = true;
    float mv_normal_dot_ = 0.9f;

    pcl::console::parse_argument (argc, argv, "-mv_normal_dot", mv_normal_dot_);
    pcl::console::parse_argument (argc, argv, "-use_plane_for_nw", use_plane_for_nw);
    pcl::console::parse_argument (argc, argv, "-lateral_sigma", lateral_sigma);
    pcl::console::parse_argument (argc, argv, "-max_angle", max_angle);
    pcl::console::parse_argument (argc, argv, "-min_weight", min_weight_);
    pcl::console::parse_argument (argc, argv, "-model_resolution", model_resolution);
    pcl::console::parse_argument (argc, argv, "-threshold_ss", threshold_ss);
    pcl::console::parse_argument (argc, argv, "-directory", directory);
    pcl::console::parse_argument (argc, argv, "-pose_str", pose_str);
    pcl::console::parse_argument (argc, argv, "-z_dist", z_dist);
    pcl::console::parse_argument (argc, argv, "-use_mv_icp_normals", use_mv_icp_normals_);

    pcl::console::parse_argument (argc, argv, "-do_cg", do_cg);
    pcl::console::parse_argument (argc, argv, "-inlier_threshold", inlier_threshold);
    pcl::console::parse_argument (argc, argv, "-gc_threshold", gc_threshold);

    pcl::console::parse_argument (argc, argv, "-use_stable_planes", use_stable_planes);
    pcl::console::parse_argument (argc, argv, "-use_sift_features", use_sift_features);

    pcl::console::parse_argument (argc, argv, "-mv_max_iterations", max_iterations);
    pcl::console::parse_argument (argc, argv, "-mv_vx_size", mv_vx_size);
    pcl::console::parse_argument (argc, argv, "-export_to", export_to);

    if(!use_sift_features && !use_stable_planes)
    {
        std::cout << "Activate either SIFT or stable planes for pairwise registration" << std::endl;
        return 0;
    }

    bool do_mv = false;
    if(max_iterations > 0)
    {
        do_mv = true;
    }

    std::vector<std::string> strs;
    boost::split(strs, directory, boost::is_any_of(","));

    std::vector<pcl::PointCloud<PointInT>::Ptr> clouds;
    std::vector<Eigen::Matrix4f> poses;
    std::vector<std::vector<int> > indices;
    std::vector<std::pair<int,int> > session_ranges;
    int elements = 0;
    std::vector<pcl::PointCloud<IndexPoint> > object_indices_clouds;

    std::vector< pcl::PointCloud<pcl::Normal>::Ptr > normals;

    std::vector<Eigen::Vector3f> plane_normals_for_sessions;
    plane_normals_for_sessions.resize(strs.size());

    for(size_t k=0; k < strs.size(); k++)
    {
        std::cout << strs[k] << std::endl;
        std::vector<std::string> to_process;
        std::string so_far = "";
        std::string pattern = ".*cloud.*.pcd";
        v4r::io::getFilesInDirectory(strs[k], to_process, so_far, pattern, true);

        std::sort(to_process.begin(), to_process.end());

        session_ranges.push_back(std::make_pair(elements, elements + to_process.size() - 1));

        for(size_t i=0; i < to_process.size(); i++)
        {
            std::stringstream view_file;
            view_file << strs[k] << "/" << to_process[i];
            pcl::PointCloud<PointInT>::Ptr cloud (new pcl::PointCloud<PointInT> ());
            pcl::io::loadPCDFile (view_file.str (), *cloud);

            std::cout << view_file.str() << std::endl;

            std::string file_replaced1 (view_file.str());
            boost::replace_last (file_replaced1, "cloud", pose_str);
            boost::replace_last (file_replaced1, ".pcd", ".txt");

            std::cout << file_replaced1 << std::endl;

            //read pose as well
            Eigen::Matrix4f pose;
            v4r::utils::readMatrixFromFile (file_replaced1, pose);

            Eigen::Matrix4f pose_inv = pose; //.inverse();

            if(use_stable_planes && (i==0))
            {
                computePlane(cloud, plane_normals_for_sessions[k], z_dist);

                //use the initial pose
                //plane_normals_for_sessions[k] = pose * Eigen::Vector4f(0,0,0,1);
            }

            std::string file_replaced2 (view_file.str());
            boost::replace_last (file_replaced2, "cloud", "object_indices");

            std::cout << file_replaced2 << std::endl;

            pcl::PointCloud<IndexPoint> obj_indices_cloud;

            pcl::io::loadPCDFile (file_replaced2, obj_indices_cloud);
            object_indices_clouds.push_back(obj_indices_cloud);

            std::vector<int> cloud_indices;
            cloud_indices.resize(obj_indices_cloud.points.size());
            for(size_t kk=0; kk < obj_indices_cloud.points.size(); kk++)
                cloud_indices[kk] = obj_indices_cloud.points[kk].idx;

            pcl::PointCloud<pcl::Normal>::Ptr normal_cloud (new pcl::PointCloud<pcl::Normal>);
            pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> ne;
            ne.setRadiusSearch(0.01f);
            ne.setInputCloud (cloud);
            ne.compute (*normal_cloud);

            clouds.push_back(cloud);
            poses.push_back(pose_inv);
            indices.push_back(cloud_indices);
            normals.push_back(normal_cloud);
        }

        elements += to_process.size();
    }

    std::cout << clouds.size() << std::endl;

    for(size_t k=0; k < session_ranges.size(); k++)
    {
        std::cout << session_ranges[k].first << " up to " << session_ranges[k].second << std::endl;
    }

    //call multiSessionModelling method
    //this will create a graph between partial models (nodes)
    //nodes might be connected by multiple edges (each one containing a transformation)
    //once all edges have been created, the multiSessionModelling will compute a tree in the graph that generates
    //a globally consistent model composed of partial models

    v4r::Registration::MultiSessionModelling<pcl::PointXYZRGB> msm;
    msm.setInputData(clouds, poses, indices, session_ranges);
    msm.setInputNormals(normals);

    //define registration algorithms
    if(use_sift_features)
    {
        std::cout << "do cg:" << do_cg << std::endl;
        boost::shared_ptr< v4r::Registration::FeatureBasedRegistration<pcl::PointXYZRGB> > fbr;
        fbr.reset(new v4r::Registration::FeatureBasedRegistration<pcl::PointXYZRGB>);
        fbr->setDoCG(do_cg);
        fbr->setGCThreshold(gc_threshold);
        fbr->setInlierThreshold(inlier_threshold);

        boost::shared_ptr< v4r::Registration::PartialModelRegistrationBase<pcl::PointXYZRGB > > cast_alg;
        cast_alg = boost::static_pointer_cast< v4r::Registration::PartialModelRegistrationBase<pcl::PointXYZRGB > > (fbr);

        msm.addRegAlgorithm(cast_alg);
    }

    if(use_stable_planes)
    {
        boost::shared_ptr< v4r::Registration::StablePlanesRegistration<pcl::PointXYZRGB> > fbr;
        fbr.reset(new v4r::Registration::StablePlanesRegistration<pcl::PointXYZRGB>);
        fbr->setSessionPlanes(plane_normals_for_sessions);
        boost::shared_ptr< v4r::Registration::PartialModelRegistrationBase<pcl::PointXYZRGB > > cast_alg;
        cast_alg = boost::static_pointer_cast< v4r::Registration::PartialModelRegistrationBase<pcl::PointXYZRGB > > (fbr);

        msm.addRegAlgorithm(cast_alg);
    }

    msm.compute();

    std::vector<Eigen::Matrix4f> output_poses;
    msm.getOutputPoses(output_poses);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr merged_cloud (new pcl::PointCloud<PointInT>);

    for(size_t i=0; i < output_poses.size(); i++)
    {

        /*std::cout << output_poses[i] << std::endl;*/

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr trans (new pcl::PointCloud<pcl::PointXYZRGB> ());
        pcl::copyPointCloud(*clouds[i], indices[i], *trans);
        pcl::transformPointCloud(*trans, *trans, output_poses[i]);
        *merged_cloud += *trans;
    }

    pcl::visualization::PCLVisualizer vis("merged model");
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler(merged_cloud);
    vis.addPointCloud<PointInT>(merged_cloud, handler, "merged_cloud");
    vis.spin();

    std::vector<std::vector<float> > weights;
    weights.resize(clouds.size());

    for(size_t i=0; i < clouds.size(); i++)
    {
        v4r::utils::noise_models::NguyenNoiseModel<pcl::PointXYZRGB> nm;
        nm.setInputCloud(clouds[i]);
        nm.setInputNormals(normals[i]);
        if(use_plane_for_nw)
            nm.setPoseToPlaneRF(poses[i]);
        nm.setLateralSigma(lateral_sigma);
        nm.setMaxAngle(max_angle);
        nm.setUseDepthEdges(true);
        nm.compute();
        nm.getWeights(weights[i]);
    }

    std::vector<Eigen::Matrix4f> final_poses;

    //MV
    if(do_mv)
    {
        typedef pcl::PointXYZRGB PointInT;

        double max_dist = 0.01f;
        int diff_type = 2;

        std::vector<pcl::PointCloud<PointInT>::Ptr> clouds_voxelized;
        std::vector<pcl::PointCloud<pcl::Normal>::Ptr> normals_voxelized;
        std::vector<std::vector<float> > weights_mv;

        weights_mv.resize(clouds.size());
        clouds_voxelized.resize(clouds.size());
        normals_voxelized.resize(clouds.size());

        for(size_t i=0; i < clouds.size(); i++)
        {

            std::vector<float> gradients;
            computeImageGradients(clouds[i], gradients);

            pcl::IndicesPtr ind;
            ind.reset(new std::vector<int>(indices[i]));

            pcl::PointCloud<int> out;
            pcl::UniformSampling<PointInT> us;
            us.setRadiusSearch(mv_vx_size);
            us.setInputCloud(clouds[i]);
            us.setIndices(ind);
            us.compute(out);

            //sample cloud and weights with out
            clouds_voxelized[i].reset(new pcl::PointCloud<PointInT>);
            normals_voxelized[i].reset(new pcl::PointCloud<pcl::Normal>);

            for(size_t k=0; k < out.points.size(); k++)
            {
                clouds_voxelized[i]->points.push_back(clouds[i]->at(out.points[k]));
                normals_voxelized[i]->points.push_back(normals[i]->at(out.points[k]));

                if( gradients[out.points[k]] < 0.5f)
                {
                    weights_mv[i].push_back(weights[i][out.points[k]] * 0.5f);
                }
                else
                {
                    weights_mv[i].push_back(weights[i][out.points[k]]);
                }

                assert(!pcl_isnan(weights[i][out.points[k]] * gradients[out.points[k]]));
            }
        }

        v4r::Registration::MvLMIcp<PointInT> nl_icp;
        nl_icp.setInputClouds(clouds_voxelized);
        nl_icp.setPoses(output_poses);
        nl_icp.setMaxCorrespondenceDistance(max_dist);
        nl_icp.setMaxIterations(max_iterations);
        nl_icp.setDiffType(diff_type);
        nl_icp.setWeights(weights_mv);
        nl_icp.setNormalDot(mv_normal_dot_);

        if(use_mv_icp_normals_)
            nl_icp.setNormals(normals_voxelized);

        nl_icp.compute();

        final_poses = nl_icp.getFinalPoses();

        pcl::PointCloud<PointInT>::Ptr big_cloud_after(new pcl::PointCloud<PointInT>);

        for(size_t i=0; i < clouds.size(); i++)
        {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr trans (new pcl::PointCloud<pcl::PointXYZRGB> ());
            pcl::copyPointCloud(*clouds[i], indices[i], *trans);

            pcl::transformPointCloud(*trans, *trans, final_poses[i]);
            *big_cloud_after += *trans;
        }

        pcl::visualization::PCLVisualizer vis("test");
        int v1,v2;
        vis.createViewPort(0,0,0.5,1,v1);
        vis.createViewPort(0.5,0,1,1,v2);
        pcl::visualization::PointCloudColorHandlerRGBField<PointInT> handler_after(big_cloud_after);
        vis.addPointCloud<PointInT>(big_cloud_after, handler_after, "after", v2);
        vis.addPointCloud<PointInT>(merged_cloud, handler, "before", v1);
        vis.spin();
    }
    else
    {
        final_poses = output_poses;
    }

    //do NM based cloud integration
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr octree_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    v4r::utils::NMBasedCloudIntegration<pcl::PointXYZRGB> nmIntegration;
    nmIntegration.setInputClouds(clouds);
    nmIntegration.setResolution(model_resolution);
    nmIntegration.setWeights(weights);
    nmIntegration.setTransformations(final_poses);
    nmIntegration.setMinWeight(min_weight_);
    nmIntegration.setInputNormals(normals);
    nmIntegration.setMinPointsPerVoxel(1);
    nmIntegration.setFinalResolution(model_resolution);
    nmIntegration.setIndices(indices);
    nmIntegration.setThresholdSameSurface(threshold_ss);
    nmIntegration.compute(octree_cloud);

    {
        pcl::visualization::PCLVisualizer vis("merged + MV + NM vs merged");
        int v1,v2;
        vis.createViewPort(0,0,0.5,1,v1);
        vis.createViewPort(0.5,0,1,1,v2);
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler_after(octree_cloud);
        vis.addPointCloud<pcl::PointXYZRGB>(octree_cloud, handler_after, "after", v2);
        vis.addPointCloud<pcl::PointXYZRGB>(merged_cloud, handler, "before", v1);
        vis.spin();
    }

    pcl::PointCloud<pcl::Normal>::Ptr octree_normals;
    nmIntegration.getOutputNormals(octree_normals);

    pcl::io::savePCDFileBinary("merged_model.pcd", *octree_cloud);

    if(export_to.compare("") != 0)
    {

        bf::path dir = export_to;
        if(!bf::exists(dir))
        {
            bf::create_directory(dir);
        }

        //save the data with new poses
        for(size_t i=0; i < final_poses.size(); i++)
        {
            std::stringstream view_file;
            view_file << export_to << "/cloud_" << setfill('0') << setw(5) << i << ".pcd";

            pcl::io::savePCDFileBinary (view_file.str (), *(clouds[i]));
            std::cout << view_file.str() << std::endl;

            std::string file_replaced1 (view_file.str());
            boost::replace_last (file_replaced1, "cloud", pose_str);
            boost::replace_last (file_replaced1, ".pcd", ".txt");

            std::cout << file_replaced1 << std::endl;

            //read pose as well
            v4r::utils::writeMatrixToFile(file_replaced1, final_poses[i]);

            std::string file_replaced2 (view_file.str());
            boost::replace_last (file_replaced2, "cloud", "object_indices");

            std::cout << file_replaced2 << std::endl;

            pcl::io::savePCDFileBinary (file_replaced2, object_indices_clouds[i]);
        }

        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr filtered_with_normals_oriented (new pcl::PointCloud<pcl::PointXYZRGBNormal>());

        std::stringstream model_output;
        model_output << export_to << "/model.pcd";
        pcl::concatenateFields(*octree_normals, *octree_cloud, *filtered_with_normals_oriented);

        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_normals_oriented (new pcl::PointCloud<pcl::PointXYZRGBNormal>());

        pcl::StatisticalOutlierRemoval<pcl::PointXYZRGBNormal> sor;
        sor.setInputCloud (filtered_with_normals_oriented);
        sor.setMeanK (50);
        sor.setStddevMulThresh (2.0);
        sor.filter (*cloud_normals_oriented);

        pcl::io::savePCDFileBinary(model_output.str(), *cloud_normals_oriented);
    }
}
