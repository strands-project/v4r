
#include <pcl/console/parse.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <faat_pcl/utils/filesystem_utils.h>
#include <faat_pcl/3d_rec_framework/segmentation/multiplane_segmentation.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <faat_pcl/registration/registration_utils.h>
#include <faat_pcl/utils/registration_utils.h>
#include <faat_pcl/utils/noise_models.h>
#include <pcl/features/normal_3d_omp.h>
#include <faat_pcl/3d_rec_framework/defines/faat_3d_rec_framework_defines.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>

/*#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/icp.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <faat_pcl/registration/icp_with_gc.h>
#include <faat_pcl/3d_rec_framework/feature_wrapper/local/local_estimator.h>
#include <faat_pcl/utils/noise_models.h>*/

typedef pcl::PointXYZRGB PointType;

/*struct PointXYZRedGreenBlue
 {
   PCL_ADD_POINT4D;
   int red;
   int green;
   int blue;
   EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
 } EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

 POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZRedGreenBlue,           // here we assume a XYZ + "test" (as fields)
                                    (float, x, x)
                                    (float, y, y)
                                    (float, z, z)
                                    (int, red, red)
                                    (int, green, green)
                                    (int, blue, blue)
)*/

struct PointXYZRedGreenBlue
{
    PCL_ADD_POINT4D;                  // preferred way of adding a XYZ+padding
    float red;
    float green;
    float blue;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZRedGreenBlue,
                                   (float, x, x)
                                   (float, y, y)
                                   (float, z, z)
                                   (float, red, red)
                                   (float, green, green)
                                   (float, blue, blue)
                                   )

inline bool
writeMatrixToFile (std::string file, Eigen::Matrix4f & matrix)
{
    std::ofstream out (file.c_str ());
    if (!out)
    {
        std::cout << "Cannot open file.\n";
        return false;
    }

    for (size_t i = 0; i < 4; i++)
    {
        for (size_t j = 0; j < 4; j++)
        {
            out << matrix (i, j);
            if (!(i == 3 && j == 3))
                out << " ";
        }
    }
    out.close ();

    return true;
}

pcl::PointCloud<pcl::PointXYZRGB> RedGreenBlue_to_RGB(const pcl::PointCloud<PointXYZRedGreenBlue> &cloud)
{
    pcl::PointCloud<pcl::PointXYZRGB> cloud2;
    cloud2.width = cloud.width;
    cloud2.height = cloud.height;
    cloud2.is_dense = true;
    cloud2.points.resize(cloud.width * cloud.height);

    for (uint i = 0; i < cloud.points.size(); i++) {
        cloud2.points[i].x = cloud.points[i].x;
        cloud2.points[i].y = cloud.points[i].y;
        cloud2.points[i].z = cloud.points[i].z;

        int r = cloud.points[i].red;
        int g = cloud.points[i].green;
        int b = cloud.points[i].blue;
        int rgbi = b;


        rgbi += (g << 8);
        rgbi += (r << 16);
        float rgbf; // = *(float*)(&rgbi);
        //memset(&rgbf, 0, sizeof(float));
        memcpy(&rgbf, (float*)(&rgbi), 3);
        cloud2.points[i].rgb = rgbf;
    }

    std::cout << "finished loading cloud..." << std::endl;

    return cloud2;
}

int
main (int argc, char ** argv)
{
    float Z_DIST_ = 1.5f;
    std::string pcd_files_dir_;
    float x_limits = 0.4f;
    int num_plane_inliers = 500;
    int icp_iterations = 5;
    bool vis_final_icp_ = false;
    int step = 1;
    std::string pcd_files_aligned_dir_;
    std::string training_dir_out_;

    pcl::console::parse_argument (argc, argv, "-training_dir_out", training_dir_out_);
    pcl::console::parse_argument (argc, argv, "-vis_final_icp", vis_final_icp_);
    pcl::console::parse_argument (argc, argv, "-icp_iterations", icp_iterations);
    pcl::console::parse_argument (argc, argv, "-pcd_files", pcd_files_dir_);
    pcl::console::parse_argument (argc, argv, "-pcd_files_aligned", pcd_files_aligned_dir_);
    pcl::console::parse_argument (argc, argv, "-Z_DIST", Z_DIST_);
    pcl::console::parse_argument (argc, argv, "-x_limits", x_limits);
    pcl::console::parse_argument (argc, argv, "-num_plane_inliers", num_plane_inliers);
    pcl::console::parse_argument (argc, argv, "-step", step);

    std::vector < std::string > strs_2;
    boost::split (strs_2, pcd_files_aligned_dir_, boost::is_any_of ("/"));
    std::string model_name = strs_2[strs_2.size() - 1];

    std::cout << pcd_files_aligned_dir_ << std::endl;
    std::cout << model_name << std::endl;

    //Read the aligned clouds
    std::vector<Eigen::Matrix4f> pose_aligned_;
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clouds_;

    std::stringstream out_dir;
    out_dir << training_dir_out_ << "/" << model_name;
    bf::path out_dir_path = out_dir.str();
    if(!bf::exists(out_dir_path))
        bf::create_directory(out_dir_path);

    {
        std::vector<std::string> files;
        std::string pattern_scenes = ".*cloud.*.pcd";

        bf::path input_dir = pcd_files_aligned_dir_;
        faat_pcl::utils::getFilesInDirectory(input_dir, files, pattern_scenes);

        std::cout << "Number of aligned scenes is:" << files.size () << std::endl;

        std::sort(files.begin(), files.end());

        pcl::visualization::PCLVisualizer vis("TEST");

        for(size_t i=0; i < files.size(); i++)
        {
            pcl::PointCloud<PointXYZRedGreenBlue> RedGreenBlue;
            std::stringstream file_to_read;
            file_to_read << pcd_files_aligned_dir_ << "/" << files[i];
            std::cout << file_to_read.str() << std::endl;
            pcl::io::loadPCDFile(file_to_read.str(), RedGreenBlue);
            std::cout << RedGreenBlue.sensor_origin_ << std::endl;
            std::cout << RedGreenBlue.sensor_orientation_.toRotationMatrix() << std::endl;
            Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
            pose.block<3,3>(0,0) = RedGreenBlue.sensor_orientation_.toRotationMatrix();
            pose.block<3,1>(0,3) = RedGreenBlue.sensor_origin_.block<3,1>(0,0);
            pose_aligned_.push_back(pose);

            pcl::PointCloud<pcl::PointXYZRGB> cloud = RedGreenBlue_to_RGB(RedGreenBlue);
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>(cloud));

            pose = pose.inverse().eval();

            pcl::transformPointCloud(*cloud_ptr, *cloud_ptr, pose);

            clouds_.push_back(cloud_ptr);

            std::stringstream name;
            name << "cloud_" << i;
            vis.addPointCloud(cloud_ptr, name.str());
        }

        //vis.spin();

    }

    pcl::visualization::PCLVisualizer aligned("ALIGNED");

    for(size_t i=0; i < clouds_.size(); i++)
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>());

        //pcl::transformPointCloud(*clouds_[i], *cloud_ptr, pose_aligned_[i]);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered (new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> ror(true);
        ror.setMeanK(10);
        ror.setStddevMulThresh(1.5f);
        ror.setInputCloud(clouds_[i]);
        ror.setNegative(false);
        ror.filter(*filtered);

        //create image
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        cloud->width = 640;
        cloud->height = 480;
        cloud->is_dense = false;
        cloud->points.resize(cloud->width * cloud->height);
        for(size_t kk=0; kk < cloud->points.size(); kk++)
        {
            cloud->points[kk].x = cloud->points[kk].y = cloud->points[kk].z =
                    std::numeric_limits<float>::quiet_NaN();

            cloud->points[kk].r = cloud->points[kk].g = cloud->points[kk].b = 0;
        }

        float f = 525.f;
        float cx = (static_cast<float> (cloud->width) / 2.f - 0.5f);
        float cy = (static_cast<float> (cloud->height) / 2.f - 0.5f);

        pcl::PointIndices cloud_object_indices;

        int ws2 = 1;
        for (size_t kk = 0; kk < filtered->points.size (); kk++)
        {
            float x = filtered->points[kk].x;
            float y = filtered->points[kk].y;
            float z = filtered->points[kk].z;
            int u = static_cast<int> (f * x / z + cx);
            int v = static_cast<int> (f * y / z + cy);

            for(int uu = (u-ws2); uu < (u+ws2); uu++)
            {
                for(int vv = (v-ws2); vv < (v+ws2); vv++)
                {
                    //Not out of bounds
                    if ((uu >= static_cast<int> (cloud->width)) ||
                            (vv >= static_cast<int> (cloud->height)) || (uu < 0) || (vv < 0))
                        continue;

                    float z_oc = cloud->at (uu, vv).z;

                    if(pcl_isnan(z_oc))
                    {
                        cloud->at (uu, vv) = filtered->points[kk];
                        cloud_object_indices.indices.push_back(vv * cloud->width + uu);
                    }
                    else
                    {
                        if(z < z_oc)
                        {
                            cloud->at (uu, vv) = filtered->points[kk];
                        }
                    }
                }
            }
        }

        std::stringstream temp;
        temp << out_dir.str() << "/cloud_";
        temp << setw( 3 ) << setfill( '0' ) << static_cast<int>(i) << ".pcd";
        std::string scene_name;
        temp >> scene_name;
        std::cout << scene_name << std::endl;
        pcl::io::savePCDFileBinary (scene_name, *cloud);

        //write pose
        {

            std::stringstream temp;
            temp << out_dir.str() << "/pose_";
            temp << setw( 3 ) << setfill( '0' ) << static_cast<int>(i) << ".txt";
            std::string scene_name;
            temp >> scene_name;
            std::cout << scene_name << std::endl;
            writeMatrixToFile(scene_name, pose_aligned_[i]);
        }

        {
            std::stringstream temp;
            temp << out_dir.str() << "/object_indices_";
            temp << setw( 3 ) << setfill( '0' ) << static_cast<int>(i) << ".pcd";
            std::string scene_name;
            temp >> scene_name;
            std::cout << scene_name << std::endl;
            pcl::PointCloud<IndexPoint> obj_indices_cloud;
            obj_indices_cloud.width = cloud_object_indices.indices.size();
            obj_indices_cloud.height = 1;
            obj_indices_cloud.points.resize(obj_indices_cloud.width);
            for(size_t kk=0; kk < cloud_object_indices.indices.size(); kk++)
                obj_indices_cloud.points[kk].idx = cloud_object_indices.indices[kk];

            pcl::io::savePCDFileBinary(scene_name, obj_indices_cloud);
        }

        pcl::transformPointCloud(*cloud, *cloud, pose_aligned_[i]);

        std::stringstream name;
        name << "cloud_" << i;
        aligned.addPointCloud(cloud, name.str());

    }

    aligned.addCoordinateSystem(0.1f);
    //aligned.spin();
}
