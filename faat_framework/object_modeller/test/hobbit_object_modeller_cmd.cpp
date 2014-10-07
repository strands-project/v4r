
#include <vtkPolyDataReader.h>
#include <vtkTransform.h>
#include <pcl/common/angles.h>
#include <faat_pcl/3d_rec_framework/utils/vtk_model_sampling.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>

// object modeller
#include "module.h"
#include "result.h"
#include "inputModule.h"
#include "outputModule.h"
#include "pipeline.h"
#include "config.h"

#include "reader/memory_friendly_fileReader.h"

#include "registration/cameraTracker.h"
#include "registration/checkerboard.h"
#include "registration/globalRegistration.h"

#include "segmentation/dominantPlaneExtraction.h"
#include "segmentation/ROISegmentation.h"

#include "output/pclRenderer.h"
#include "output/pointCloudWriter.h"
#include "output/indicesWriter.h"
#include "output/posesWriter.h"
#include "output/meshRenderer.h"
#include "output/renderer.h"
#include "output/pointCloudRenderer.h"

#include "util/transform.h"
#include "util/mask.h"
#include "util/multiplyMatrix.h"
#include "util/distanceFilter.h"
#include "util/normalEstimationOmp.h"
#include "util/integralImageNormalEstimation.h"
#include "util/nguyenNoiseWeights.h"
#include "util/vectorMask.h"

#include "modelling/nmBasedCloudIntegration.h"
#include "modelling/poissonReconstruction.h"

#include <pcl/common/angles.h>
#include <pcl/common/transforms.h>
#include <pcl/common/time.h>
#include <pcl/io/pcd_io.h>
#include <faat_pcl/utils/miscellaneous.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <faat_pcl/recognition/hv/occlusion_reasoning.h>
#include <faat_pcl/utils/noise_model_based_cloud_integration.h>
#include <pcl/octree/octree.h>
#include <pcl/features/normal_3d_omp.h>
#include <faat_pcl/utils/noise_models.h>

using namespace object_modeller;

// helper methods
Config::Ptr parseCommandLineArgs(int argc, char **argv);

void buildTransformationMatrixBaseToCam(Eigen::Matrix4f & matrix);
void visCloudFromSinglePoint(Eigen::Vector4f & p,
                             std::string name,
                             pcl::visualization::PCLVisualizer & vis,
                             int r, int g, int b, int p_size);

void voxelGridCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr & input,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr & output,
                    float res = 0.003);

template<typename ModelT, typename SceneT>
void
filter (typename pcl::PointCloud<SceneT>::Ptr & organized_cloud,
        typename pcl::PointCloud<ModelT>::ConstPtr & to_be_filtered,
        float f, float threshold, std::vector<int> & indices_to_keep)
{
    float cx = (static_cast<float> (organized_cloud->width) / 2.f - 0.5f);
    float cy = (static_cast<float> (organized_cloud->height) / 2.f - 0.5f);

    //std::vector<int> indices_to_keep;
    indices_to_keep.resize (to_be_filtered->points.size ());

    pcl::PointCloud<float> filtered_points_depth;
    pcl::PointCloud<int> closest_idx_points;
    filtered_points_depth.points.resize (organized_cloud->points.size ());
    closest_idx_points.points.resize (organized_cloud->points.size ());

    filtered_points_depth.width = closest_idx_points.width = organized_cloud->width;
    filtered_points_depth.height = closest_idx_points.height = organized_cloud->height;
    for (size_t i = 0; i < filtered_points_depth.points.size (); i++)
    {
        filtered_points_depth.points[i] = std::numeric_limits<float>::quiet_NaN ();
        closest_idx_points.points[i] = -1;
    }

    int keep = 0;
    for (size_t i = 0; i < to_be_filtered->points.size (); i++)
    {
        float x = to_be_filtered->points[i].x;
        float y = to_be_filtered->points[i].y;
        float z = to_be_filtered->points[i].z;
        int u = static_cast<int> (f * x / z + cx);
        int v = static_cast<int> (f * y / z + cy);

        //Not out of bounds
        if ((u >= static_cast<int> (organized_cloud->width)) || (v >= static_cast<int> (organized_cloud->height)) || (u < 0) || (v < 0))
            continue;

        //Check for invalid depth
        if (!pcl_isfinite (organized_cloud->at (u, v).x) || !pcl_isfinite (organized_cloud->at (u, v).y)
                || !pcl_isfinite (organized_cloud->at (u, v).z))
            continue;

        float z_oc = organized_cloud->at (u, v).z;

        //Check if point depth (distance to camera) is greater than the (u,v)
        if ((z - z_oc) > threshold)
            continue;

        if (pcl_isnan(filtered_points_depth.at (u, v)) || (z < filtered_points_depth.at (u, v)))
        {
            closest_idx_points.at (u, v) = static_cast<int> (i);
            filtered_points_depth.at (u, v) = z;
        }

        //indices_to_keep[keep] = static_cast<int> (i);
        //keep++;
    }

    for (size_t i = 0; i < closest_idx_points.points.size (); i++)
    {
        if(closest_idx_points[i] != -1)
        {
            indices_to_keep[keep] = closest_idx_points[i];
            keep++;
        }
    }

    indices_to_keep.resize (keep);
}

template<typename T1, typename T2>
void occlusionReasoningFromMultipleViews(typename pcl::PointCloud<T1>::Ptr & model_cloud,
                                         std::vector<typename pcl::PointCloud<T2>::Ptr> & clouds,
                                         std::vector<Eigen::Matrix4f> & poses,
                                         std::vector<int> & visible_indices)
{
    //scene-occlusions
    for(size_t k=0; k < clouds.size(); k++)
    {
        //transform model to camera coordinate
        typename pcl::PointCloud<T1>::Ptr model_in_view_coordinates(new pcl::PointCloud<T1> ());
        Eigen::Matrix4f trans =  poses[k].inverse();
        pcl::transformPointCloud(*model_cloud, *model_in_view_coordinates, trans);
        typename pcl::PointCloud<T1>::ConstPtr const_filtered(new pcl::PointCloud<T1> (*model_in_view_coordinates));

        std::vector<int> indices_cloud_occlusion;
        filter<T1,T2> (clouds[k], const_filtered, 525.f, 0.01, indices_cloud_occlusion);

        std::vector<int> final_indices = indices_cloud_occlusion;
        final_indices.resize(indices_cloud_occlusion.size());

        visible_indices.insert(visible_indices.end(), final_indices.begin(), final_indices.end());
    }

    std::set<int> s( visible_indices.begin(), visible_indices.end() );
    visible_indices.assign( s.begin(), s.end() );

}

template<typename T1, typename T2>
void countInliersAndOutliers(typename pcl::PointCloud<T1>::Ptr & model_cloud,
                             typename pcl::PointCloud<T2>::Ptr & scene_cloud,
                             int inliers_outliers[2],
                             float inliers_threshold=0.005f)
{
    typename pcl::octree::OctreePointCloudSearch<T2> octree(0.005);
    octree.setInputCloud(scene_cloud);
    octree.addPointsFromInputCloud();

    std::vector<int> nn_indices;
    std::vector<float> nn_distances;

    std::vector<bool> scene_cloud_explained(scene_cloud->points.size(), false);
    int outliers=0;
    for(size_t i=0; i < model_cloud->points.size(); i++)
    {
        T2 p_model;
        p_model.getVector3fMap() = model_cloud->points[i].getVector3fMap();
        if (octree.radiusSearch (p_model, inliers_threshold,
                                 nn_indices, nn_distances, std::numeric_limits<int>::max ()) > 0)
        {
            for(size_t k=0; k < nn_indices.size(); k++)
            {
                scene_cloud_explained[nn_indices[k]] = true;
            }
        }
        else
        {
            outliers++;
        }
    }

    int inliers=0;
    for(size_t i=0; i < scene_cloud_explained.size(); i++)
    {
        if(scene_cloud_explained[i])
            inliers++;
    }

    inliers_outliers[0] = inliers;
    inliers_outliers[1] = outliers;
}

//#define VIS

/******************************************************************
 * MAIN
 */
int main(int argc, char *argv[] )
{
    Config::Ptr config = parseCommandLineArgs(argc, argv);

    bool step = config->getBool("pipeline", "step");

    std::cout << "pipeline step: " << step << std::endl;

    // input reader
    reader::MemoryFriendlyFileReader reader;

    util::NormalEstimationOmp normal_estimation;
    util::NguyenNoiseWeights weights_calculation;

    // registration
    registration::CameraTracker camera_tracker;

    registration::GlobalRegistration global_registration;
    float min_overlap = config->getFloat("globalRegistration", "views_overlap");
    global_registration.setMinOverlap(min_overlap);
    global_registration.setMinDot(0.9f);

    //segmentation
    segmentation::ROISegmentation ROI_segmentation;

    //modelling
    modelling::NmBasedCloudIntegration nm_based_cloud_integration;

    //output
    output::PointCloudWriter<pcl::PointXYZRGB> pointcloud_writer;
    output::IndicesWriter indices_writer;
    output::PosesWriter poses_writer;
    output::PointCloudWriter<pcl::PointXYZRGBNormal> model_writer("modelWriter");

    Eigen::Matrix4f base_to_cam, base_to_cam_inverse;
    buildTransformationMatrixBaseToCam(base_to_cam);
    base_to_cam_inverse = base_to_cam.inverse();

    Eigen::Vector4f head_center_camera_link (0.05, 0, 0, 1);
    Eigen::Vector4f head_center_base_link = base_to_cam * head_center_camera_link;

    Eigen::Vector4f tt_point_cam_link_in_base_link_coordinates(0.30264, -0.38249 , -0.52173, 0);
    Eigen::Vector4f turn_table_base_link = head_center_base_link + tt_point_cam_link_in_base_link_coordinates;

    Eigen::Vector4f tt_point_camera;
    tt_point_camera = base_to_cam_inverse * turn_table_base_link;

    Eigen::Vector4f ROI_min, ROI_max;
    float box_size = 0.15f;
    float under = 0.1f;
    float above = 0.3f;
    ROI_min = Eigen::Vector4f(-box_size, -box_size ,-under, 1);
    ROI_max = Eigen::Vector4f(box_size, box_size , above, 1);

    Eigen::Affine3f turn_table_pose_for_vis;
    Eigen::Affine3f turn_table_pose;
    turn_table_pose.setIdentity();
    turn_table_pose.linear() = base_to_cam.block<3,3>(0,0).inverse();
    turn_table_pose.translation() = tt_point_camera.head<3>();
    turn_table_pose_for_vis = turn_table_pose;
    turn_table_pose = turn_table_pose.inverse();

    Eigen::Vector3f trans_table_pose = turn_table_pose_for_vis.translation();
    Eigen::Matrix3f rot = turn_table_pose_for_vis.linear();
    Eigen::Quaternionf q_table(rot);

    Eigen::Vector3f center_depth(0, 0, (above + under) / 2.f - under);
    center_depth = rot * center_depth;
    trans_table_pose += center_depth;

    ROI_segmentation.setMinMax(ROI_min, ROI_max);
    ROI_segmentation.setTransformation(turn_table_pose);

    reader.applyConfig(config);
    camera_tracker.applyConfig(config);

    std::vector<std::string> files = reader.process();
    std::cout << files.size() << std::endl;

#ifdef VIS
    pcl::visualization::PCLVisualizer vis("tracking process");
#endif

    std::vector<int> keyframes;
    std::vector<Eigen::Matrix4f> poses;
    std::vector< std::vector<int> > indices_keyframes;
    std::vector< pcl::PointCloud<pcl::PointXYZRGB>::Ptr > keyframe_clouds;

    //std::string save_path = "/media/DATA/hobbit/data_for_hannes/mug_down";

    for(size_t i=0; i < files.size(); i++)
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr current_cloud = reader.getCloud(static_cast<int>(i));
#ifdef VIS
        vis.removeAllPointClouds();
        vis.removeAllShapes();
        vis.addCoordinateSystem(0.3);
        vis.addPointCloud(current_cloud);
        vis.addCube(trans_table_pose, q_table, box_size * 2, box_size * 2, under + above, "cube");
        vis.spinOnce(1, true);
#endif

        /*visCloudFromSinglePoint(tt_point_camera, "tt_point_camera", vis, 255, 255, 0, 48); //yellow*/

        //segment region of interest using pose approximation
        std::vector<int> kept_ind;
        ROI_segmentation.processSingle(current_cloud, kept_ind);

        //track frame knowing if keyframe or not...
        Eigen::Matrix4f pose;
        bool is_key_frame;

        camera_tracker.trackSingleFrame(current_cloud, pose, is_key_frame);

        std::cout << "pose:" << pose << std::endl;
        std::cout << "keyframe:" << is_key_frame << std::endl;

        if(is_key_frame)
        {
            keyframes.push_back(i);
            poses.push_back(pose.inverse());
            indices_keyframes.push_back(kept_ind);
            keyframe_clouds.push_back(current_cloud);
        }

        //save data
        /*std::stringstream save_to;
        save_to << save_path << "/cloud_" << setw(5) << setfill('0') << i << ".pcd";
        pcl::io::savePCDFileBinary(save_to.str(), *current_cloud);*/
    }

#ifdef VIS
    vis.removeAllShapes();
    vis.removeAllPointClouds();
    vis.addCoordinateSystem(0.3);
#endif

    std::cout << "Number of keyframes:" << keyframes.size() << std::endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr scene_icp_initial(new pcl::PointCloud<pcl::PointXYZ>);

    for(size_t i=0; i < keyframes.size(); i++)
    {
        //pcl::PointCloud<pcl::PointXYZRGB>::Ptr current_cloud = reader.getCloud(static_cast<int>(keyframes[i]));
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr current_cloud = keyframe_clouds[i];
        std::stringstream name;
        name << "cloud_" << i;

        Eigen::Matrix4f pose = poses[i]; //.inverse();
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr trans(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::copyPointCloud(*current_cloud, indices_keyframes[i], *trans);
        pcl::transformPointCloud(*trans, *trans, pose);

#ifdef VIS
        vis.addPointCloud(trans, name.str());
#endif

        pcl::PointCloud<pcl::PointXYZ>::Ptr trans_xyz(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::copyPointCloud(*trans, *trans_xyz);

        *scene_icp_initial += *trans_xyz;
    }

#ifdef VIS
    vis.spin();
#endif

    //compute keyframes normals and weights
    std::vector<pcl::PointCloud<pcl::Normal>::Ptr> normals;
    std::vector< std::vector<float> > weights;

    {
        pcl::ScopeTime t("compute normals and weights");
        //normals = normal_estimation.process(keyframe_clouds);
        //weights = weights_calculation.process(keyframe_clouds, normals);

        for(size_t i=0; i < indices_keyframes.size(); i++)
        {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene = keyframe_clouds[i];

            pcl::PointCloud<pcl::Normal>::Ptr normal_cloud (new pcl::PointCloud<pcl::Normal>);
            pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> ne;
            ne.setInputCloud (scene);
            pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
            ne.setSearchMethod (tree);
            ne.setRadiusSearch (0.02);
            ne.compute (*normal_cloud);

            normals.push_back(normal_cloud);

            faat_pcl::utils::noise_models::NguyenNoiseModel<pcl::PointXYZRGB> nm;
            nm.setInputCloud(scene);
            nm.setInputNormals(normal_cloud);
            nm.setLateralSigma(0.001f);
            nm.setMaxAngle(50);
            nm.setUseDepthEdges(true);
            nm.compute();
            std::vector<float> w;
            nm.getWeights(w);

            weights.push_back(w);
        }
    }

    if (config->getBool("pipeline", "enableMultiview"))
    {

        global_registration.applyConfig(config);

        std::vector<pcl::PointCloud<pcl::Normal>::Ptr> normals_global;
        std::vector< std::vector<float> > weights_global;
        std::vector< pcl::PointCloud<pcl::PointXYZRGB>::Ptr > clouds_global;

        //transform keyframe_clouds and normals first using initial poses into clouds_global
        //use also the indices to remove NaN values and keep only ROI
        for(size_t i=0; i < keyframe_clouds.size(); i++)
        {

            //do a uniform sampling to speed up computations
            pcl::PointIndicesPtr index(new pcl::PointIndices);
            index->indices = indices_keyframes[i];
            pcl::UniformSampling<pcl::PointXYZRGB> us;
            us.setIndices(index);
            us.setRadiusSearch(0.002f);
            us.setInputCloud(keyframe_clouds[i]);
            pcl::PointCloud<int> indices_cloud;
            us.compute(indices_cloud);

            std::vector<int> indices;
            indices.resize(indices_cloud.points.size());

            for(size_t k=0; k < indices_cloud.points.size(); k++)
            {
                indices[k] = indices_cloud.points[k];
            }

            std::cout << indices.size() << " " << indices_keyframes[i].size() << std::endl;

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            pcl::copyPointCloud(*keyframe_clouds[i], indices, *cloud);
            pcl::transformPointCloud(*cloud, *cloud, poses[i]);

            pcl::PointCloud<pcl::Normal>::Ptr normal(new pcl::PointCloud<pcl::Normal>);
            pcl::copyPointCloud(*normals[i], indices, *normal);

            pcl::PointCloud<pcl::Normal>::Ptr normal_aligned(new pcl::PointCloud<pcl::Normal>);
            faat_pcl::utils::miscellaneous::transformNormals(normal, normal_aligned, poses[i]);

            std::vector<float> w;
            w.resize(indices.size());
            for(size_t k=0; k < indices.size(); k++)
            {
                w[k] = weights[i][indices[k]];
            }

            clouds_global.push_back(cloud);
            normals_global.push_back(normal_aligned);
            weights_global.push_back(w);
        }

        std::vector<Eigen::Matrix4f> incremental_poses = global_registration.process(clouds_global,
                                                                                     normals_global,
                                                                                     weights_global);

        for(size_t i=0; i < poses.size(); i++)
        {
            poses[i] = incremental_poses[i] * poses[i];
        }

#ifdef VIS
        vis.removeAllPointClouds();
#endif

        scene_icp_initial.reset(new pcl::PointCloud<pcl::PointXYZ>);

        for(size_t i=0; i < keyframes.size(); i++)
        {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr current_cloud = keyframe_clouds[i];
            std::stringstream name;
            name << "cloud_" << i;

            Eigen::Matrix4f pose = poses[i];
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr trans(new pcl::PointCloud<pcl::PointXYZRGB>);
            pcl::copyPointCloud(*current_cloud, indices_keyframes[i], *trans);
            pcl::transformPointCloud(*trans, *trans, pose);

#ifdef VIS
            vis.addPointCloud(trans, name.str());
#endif

            pcl::PointCloud<pcl::PointXYZ>::Ptr trans_xyz(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::copyPointCloud(*trans, *trans_xyz);

            *scene_icp_initial += *trans_xyz;
        }

#ifdef VIS
        std::cout << "cloud after global alignment" << std::endl;
        vis.spin();
#endif
    }

    //segmentation of turn table on reconstructed model

    std::string model = config->getString("miscellaneous", "turn_tablemodel", "model.ply");
    float model_scale = config->getFloat("miscellaneous", "model_scale");

    pcl::PointCloud<pcl::PointXYZ>::Ptr model_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr model_cloud_trans(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr model_cloud_sampled(new pcl::PointCloud<pcl::PointXYZ>);

    //faat_pcl::rec_3d_framework::uniform_sampling (model, 100000, *model_cloud, model_scale);
    pcl::io::loadPCDFile(model, *model_cloud);

    Eigen::Matrix4f scale;
    scale.setIdentity();
    scale *= model_scale;
    scale(3,3) = 1;
    pcl::transformPointCloud(*model_cloud, *model_cloud, scale);

    float rotation = 0;
    float rotation_step = 30;
    float resolution = 0.005f;

    pcl::PointCloud<pcl::PointXYZ>::Ptr scene_icp;
    voxelGridCloud(scene_icp_initial, scene_icp, resolution);

    float best_alignment = -std::numeric_limits<float>::infinity();
    Eigen::Matrix4f best_transform;
    best_transform.setIdentity();

    while(rotation < (360 - rotation_step))
    {

        Eigen::AngleAxisf rotation_tt(pcl::deg2rad(rotation), Eigen::Vector3f::UnitZ());

        Eigen::Matrix4f turn_table_pose;
        turn_table_pose.setIdentity();
        turn_table_pose.block<3,3>(0,0) = base_to_cam.block<3,3>(0,0).inverse() * rotation_tt.toRotationMatrix();
        turn_table_pose.block<3,1>(0,3) = tt_point_camera.head<3>();

        pcl::transformPointCloud(*model_cloud, *model_cloud_sampled, turn_table_pose);

        voxelGridCloud(model_cloud_sampled, model_cloud_trans, resolution);

        /*{
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler(model_cloud_trans, 255, 0, 255);
            vis.addPointCloud(model_cloud_trans, handler, "before_icp");
            vis.spin();
            vis.removePointCloud("before_icp");
        }*/

        pcl::PointCloud<pcl::PointXYZ>::Ptr output(new pcl::PointCloud<pcl::PointXYZ>);

        pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
        icp.setInputSource(model_cloud_trans);
        icp.setInputTarget(scene_icp);
        icp.setMaxCorrespondenceDistance(0.01f);
        icp.setMaximumIterations(20);
        icp.setEuclideanFitnessEpsilon(1e-9);
        icp.align(*output);
        Eigen::Matrix4f icp_trans = icp.getFinalTransformation();

        pcl::transformPointCloud(*model_cloud_trans, *model_cloud_trans, icp_trans);

        turn_table_pose = icp_trans * turn_table_pose;

        /*vtkSmartPointer < vtkTransform > poseTransform = vtkSmartPointer<vtkTransform>::New ();
        vtkSmartPointer < vtkTransform > scale_models = vtkSmartPointer<vtkTransform>::New ();
        scale_models->Scale(model_scale, model_scale, model_scale);

        vtkSmartPointer < vtkMatrix4x4 > mat = vtkSmartPointer<vtkMatrix4x4>::New ();
        for (size_t kk = 0; kk < 4; kk++)
        {
         for (size_t k = 0; k < 4; k++)
         {
           mat->SetElement (kk, k, turn_table_pose (kk, k));
         }
        }

        poseTransform->SetMatrix (mat);
        poseTransform->Modified ();
        poseTransform->Concatenate(scale_models);

        vis.addModelFromPLYFile (model, poseTransform, "CAD model");
        vis.spin();
        vis.removeShape("CAD model");*/

        /*{
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler(model_cloud_trans, 255, 0, 255);
            vis.addPointCloud(model_cloud_trans, handler, "tt");
            vis.spin();
            vis.removePointCloud("tt");
        }*/

        std::vector<int> visible;
        occlusionReasoningFromMultipleViews<pcl::PointXYZ, pcl::PointXYZRGB>(model_cloud_trans, keyframe_clouds, poses, visible);

        pcl::PointCloud<pcl::PointXYZ>::Ptr model_cloud_visible(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::copyPointCloud(*model_cloud_trans, visible, *model_cloud_visible);

        /*pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler(model_cloud_visible, 255, 0, 0);
        vis.addPointCloud(model_cloud_visible, handler, "visible");
        vis.spin();
        vis.removePointCloud("visible");*/

        //combination of fitness score and overlap?
        //actually we should do visibility reasoning for each point and then compute outliers and inliers

        int inliers_outliers[2];
        countInliersAndOutliers<pcl::PointXYZ, pcl::PointXYZ>(model_cloud_visible, scene_icp, inliers_outliers);

        std::cout << "inliers:" << inliers_outliers[0] << " outliers:" << inliers_outliers[1] << std::endl;
        float quality = inliers_outliers[0] - inliers_outliers[1];
        if(quality > best_alignment)
        {
            best_transform = turn_table_pose;
            best_alignment = quality;
        }

        rotation += rotation_step;
    }

#ifdef VIS
    //visualize best alignment
    pcl::transformPointCloud(*model_cloud, *model_cloud_sampled, best_transform);
    voxelGridCloud(model_cloud_sampled, model_cloud_trans, resolution);

    {
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler(model_cloud_trans, 255, 0, 0);
        vis.addPointCloud(model_cloud_trans, handler, "visible");
        vis.spin();
        vis.removePointCloud("visible");
    }

    vis.removeAllPointClouds();

#endif

    //using the best alignment, segment the keyframes
    //transform with best_transform inverse and remove indices below zero
    std::vector<std::vector<int> > segmented_indices;

    for(size_t i=0; i < keyframe_clouds.size(); i++)
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr current_cloud = keyframe_clouds[i];
        Eigen::Matrix4f pose = poses[i];
        pose = best_transform.inverse() * pose;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr trans(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::copyPointCloud(*current_cloud, indices_keyframes[i], *trans);
        pcl::transformPointCloud(*trans, *trans, pose);

        std::vector<int> indices_above;
        for(size_t k=0; k < indices_keyframes[i].size(); k++)
        {
            if(trans->points[k].z > 0.01f)
            {
                indices_above.push_back(indices_keyframes[i][k]);
            }
        }

        segmented_indices.push_back(indices_above);
    }

    indices_keyframes = segmented_indices;

    //based cloud integration
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr octree_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    faat_pcl::utils::NMBasedCloudIntegration<pcl::PointXYZRGB> nmIntegration;
    nmIntegration.setInputClouds(keyframe_clouds);
    nmIntegration.setResolution(0.001f);
    nmIntegration.setWeights(weights);
    nmIntegration.setTransformations(poses);
    nmIntegration.setMinWeight(0.75f);
    nmIntegration.setInputNormals(normals);
    nmIntegration.setMinPointsPerVoxel(0);
    nmIntegration.setFinalResolution(0.001f);
    nmIntegration.setIndices(indices_keyframes);
    nmIntegration.compute(octree_cloud);

    pcl::PointCloud<pcl::Normal>::Ptr big_normals(new pcl::PointCloud<pcl::Normal>);
    nmIntegration.getOutputNormals(big_normals);

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr filtered_with_normals_oriented (new pcl::PointCloud<pcl::PointXYZRGBNormal>());
    pcl::concatenateFields(*big_normals, *octree_cloud, *filtered_with_normals_oriented);
    filtered_with_normals_oriented->is_dense = false;

#ifdef VIS
    pcl::visualization::PCLVisualizer vis_model("final_model");
    {
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal> handler(filtered_with_normals_oriented);
        vis_model.addPointCloud<pcl::PointXYZRGBNormal>(filtered_with_normals_oriented, handler, "FINAL");

        vis_model.spin();
        vis_model.removeAllPointClouds();
    }
#endif


    /*nm_based_cloud_integration.applyConfig(config);
    std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr > final_models =
            nm_based_cloud_integration.process(keyframe_clouds, poses, indices_keyframes, normals, weights);
    */

    //write stuff
    poses_writer.applyConfig(config);
    indices_writer.applyConfig(config);
    pointcloud_writer.applyConfig(config);
    model_writer.applyConfig(config);

    poses_writer.process(poses);
    indices_writer.process(indices_keyframes);
    pointcloud_writer.process(keyframe_clouds);

    std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr > final_model;
    final_model.push_back(filtered_with_normals_oriented);
    model_writer.process(final_model);
    return 0;

}

/**
 * setup command line args
 */
Config::Ptr parseCommandLineArgs(int argc, char **argv)
{
    if (argc > 1)
    {
        // ignore first arg (filename)

        // second arg is config file name
        std::string configPath(argv[1]);

        std::cout << "filename: " << configPath << std::endl;

        Config::Ptr config(new Config());
        config->loadFromFile(configPath);

        for (int i=2;i<argc;i++)
        {
            std::string arg(argv[i]);
            boost::algorithm::trim_left_if(arg, boost::algorithm::is_any_of("-"));

            std::vector<std::string> result;
            boost::algorithm::split(result, arg, boost::algorithm::is_any_of("="));

            // std::cout << result[0] << " = " << result[1] << std::endl;

            config->overrideParameter(result[0], result[1]);
        }

        config->printConfig();

        return config;
    }
    else
    {
        std::cout << "Usage: " << std::endl;
    }
}

void buildTransformationMatrixBaseToCam(Eigen::Matrix4f & matrix)
{
    Eigen::Vector3f trans_base_to_neck(-0.260, 0.000, 1.090);
    Eigen::Matrix4f base_to_neck;
    base_to_neck.setIdentity();
    base_to_neck.block<3,1>(0,3) = trans_base_to_neck; //brings point in basis CS to

    Eigen::Vector3f trans_neck_to_cam(0.012, -0.045, 0.166);
    Eigen::Quaternionf q_neck_to_cam (0.508, -0.486, 0.492, -0.514);

    Eigen::AngleAxisf rollAngle(pcl::deg2rad(-61.0), Eigen::Vector3f::UnitZ());
    Eigen::AngleAxisf yawAngle(pcl::deg2rad(40.0), Eigen::Vector3f::UnitY());

    Eigen::Matrix4f rotation_learning;
    rotation_learning.setIdentity();
    rotation_learning.block<3,3>(0,0) = rollAngle.toRotationMatrix() * yawAngle.toRotationMatrix();

    Eigen::Matrix4f neck_to_cam;
    neck_to_cam.setIdentity();
    neck_to_cam.block<3,3>(0,0) = q_neck_to_cam.toRotationMatrix();
    neck_to_cam.block<3,1>(0,3) = trans_neck_to_cam;

    matrix = base_to_neck * rotation_learning * neck_to_cam;
}

void visCloudFromSinglePoint(Eigen::Vector4f & p,
                             std::string name,
                             pcl::visualization::PCLVisualizer & vis,
                             int r, int g, int b, int p_size)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr origin_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointXYZ orig_p;
    orig_p.getVector4fMap() = p;
    origin_cloud->points.push_back(orig_p);

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler(origin_cloud, r, g, b);
    vis.addPointCloud(origin_cloud, handler, name);
    vis.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, p_size, name);
}

void voxelGridCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr & input,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr & output,
                    float res)
{
    output.reset(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> filter;
    filter.setInputCloud(input);
    filter.setLeafSize(res, res, res);
    filter.filter(*output);
}
