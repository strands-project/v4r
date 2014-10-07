#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <vtkPolyDataReader.h>
#include <vtkTransform.h>
#include <pcl/common/angles.h>
#include <faat_pcl/3d_rec_framework/utils/vtk_model_sampling.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>

// object modeller

/******************************************************************
 * MAIN
 */

/*

00063     printf("Usage: tf_echo source_frame target_frame\n\n");
00064     printf("This will echo the transform from the coordinate frame of the source_frame\n");
00065     printf("to the coordinate frame of the target_frame. \n");
00066     printf("Note: This is the transform to get data from target_frame into the source_frame.\n");

rosrun tf tf_echo base_link hobbit_neck
At time 1407327765.580
- Translation: [-0.260, 0.000, 1.090]
- Rotation: in Quaternion [0.000, 0.000, 0.000, 1.000]
            in RPY [0.000, -0.000, 0.000]

            * transform from base_link (source) to hobbit_neck (target)
            * T x_{hobbit_neck} = x_{base_link}
                    if x_{hobbit_neck} = (0,0,0), then x_{base_link} is:
                    [ 1 0 0 Tx ]   [ 0 ]   [ Tx ]   [-0.26]
                    [ 0 1 1 Ty ] X [ 0 ] = [ Ty ] = [0.000]
                    [ 0 0 1 Tz ]   [ 0 ]   [ Tz ]   [1.090]
                    [ 0 0 0 1  ]   [ 1 ]   [ 1  ]   [1    ]

              basically, the coordinates of the neck origin in the basis of base_link

rosrun tf tf_echo hobbit_neck headcam_rgb_optical_frame
At time 1407327785.213
- Translation: [0.012, -0.045, 0.166]
- Rotation: in Quaternion [-0.486, 0.492, -0.514, 0.508]
            in RPY [-1.526, 0.000, -1.583

Und die Winkel um die sich der Kopf aus dieser Position zum Turntable dreht sind:
40° nach unten
61° nach rechts.*/

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

    std::cout << neck_to_cam << std::endl;
    std::cout << "----------------------------------------------------" << std::endl;
    std::cout << rotation_learning * neck_to_cam << std::endl;
    std::cout << "----------------------------------------------------" << std::endl;

    matrix = base_to_neck * rotation_learning * neck_to_cam;
}

void voxelGridCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr & input,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr & output,
                    float res = 0.003)
{
    output.reset(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> filter;
    filter.setInputCloud(input);
    filter.setLeafSize(res, res, res);
    filter.filter(*output);
}

void RPYtoRotationMatrix(float roll, float pitch, float yaw, Eigen::Matrix3f & rotationMatrix)
{
    Eigen::AngleAxisf rollAngle(roll, Eigen::Vector3f::UnitZ());
    Eigen::AngleAxisf yawAngle(yaw, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf pitchAngle(pitch, Eigen::Vector3f::UnitX());

    Eigen::Quaternion<float> q = rollAngle * yawAngle * pitchAngle;

    rotationMatrix = q.matrix();
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

int main(int argc, char *argv[] )
{
    std::string model, model_pcd;
    std::string scene;
    float model_scale = 1.f;
    float rotation = 0;

    Eigen::Matrix4f base_to_cam, base_to_cam_inverse;
    buildTransformationMatrixBaseToCam(base_to_cam);

    std::cout << base_to_cam << std::endl;

    base_to_cam_inverse = base_to_cam.inverse();

    pcl::console::parse_argument (argc, argv, "-model", model);
    pcl::console::parse_argument (argc, argv, "-scene", scene);
    pcl::console::parse_argument (argc, argv, "-model_scale", model_scale);
    pcl::console::parse_argument (argc, argv, "-model_pcd", model_pcd);
    pcl::console::parse_argument (argc, argv, "-rotation", rotation);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::io::loadPCDFile(scene, *cloud);


    pcl::PointCloud<pcl::PointXYZ>::Ptr model_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::PointCloud<pcl::PointXYZ>::Ptr model_cloud_trans(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile(model_pcd, *model_cloud);

    pcl::visualization::PCLVisualizer vis("test");
    vis.addPointCloud(cloud, "cloud");

    //find coordinates of head_center in basis_link coordinates
    // x_{base_link} = T * x_{}

    Eigen::Vector4f head_center_camera_link (0.05, 0, 0, 1);
    Eigen::Vector4f head_center_base_link = base_to_cam * head_center_camera_link;

    Eigen::Vector4f tt_point_cam_link_in_base_link_coordinates(0.30264, -0.38249 , -0.52173, 0);
    Eigen::Vector4f turn_table_base_link = head_center_base_link + tt_point_cam_link_in_base_link_coordinates;

    Eigen::Vector4f tt_point_camera;
    tt_point_camera = base_to_cam_inverse * turn_table_base_link;

    std::cout << tt_point_camera << std::endl;

    //visCloudFromSinglePoint(tt_point_camera, "tt_point_camera", vis, 255, 255, 0, 48); //yellow

    //compute the transformation aligning the turn table model to the cloud
    Eigen::AngleAxisf rotation_tt(pcl::deg2rad(rotation), Eigen::Vector3f::UnitZ());

    Eigen::Matrix4f turn_table_pose;
    turn_table_pose.setIdentity();
    turn_table_pose.block<3,3>(0,0) = base_to_cam.block<3,3>(0,0).inverse() * rotation_tt.toRotationMatrix();
    turn_table_pose.block<3,1>(0,3) = tt_point_camera.head<3>();

    pcl::PointCloud<pcl::PointXYZ>::Ptr model_cloud_sampled(new pcl::PointCloud<pcl::PointXYZ>);
    faat_pcl::rec_3d_framework::uniform_sampling (model, 100000, *model_cloud, model_scale);
    pcl::transformPointCloud(*model_cloud, *model_cloud_sampled, turn_table_pose);

    voxelGridCloud(model_cloud_sampled, model_cloud_trans);

    pcl::PointCloud<pcl::PointXYZ>::Ptr scene_icp_initial(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr scene_icp;
    pcl::PointCloud<pcl::PointXYZ>::Ptr output(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::copyPointCloud(*cloud, *scene_icp_initial);
    voxelGridCloud(scene_icp_initial, scene_icp);

    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(model_cloud_trans);
    icp.setInputTarget(scene_icp);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaximumIterations(10);
    icp.align(*output);
    Eigen::Matrix4f icp_trans = icp.getFinalTransformation();

    pcl::transformPointCloud(*model_cloud_trans, *model_cloud_trans, icp_trans);

    turn_table_pose = icp_trans * turn_table_pose;

    /*{
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler(model_cloud_trans, 255, 0, 255);
        vis.addPointCloud(model_cloud_trans, handler, "turn_table_in_model_coordinates");
    }*/

    vtkSmartPointer < vtkTransform > poseTransform = vtkSmartPointer<vtkTransform>::New ();
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
    vis.addCoordinateSystem(0.3);
    vis.spin();

    return 0;
}
