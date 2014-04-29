
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
    std::string pcd_files_dir_;
    std::string pcd_files_out;

    pcl::console::parse_argument (argc, argv, "-pcd_files_out", pcd_files_out);
    pcl::console::parse_argument (argc, argv, "-pcd_files", pcd_files_dir_);

    //Read the aligned clouds
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clouds_;

    bf::path out_dir_path = pcd_files_out;
    if(!bf::exists(out_dir_path))
        bf::create_directory(out_dir_path);

    std::vector<std::string> files;

    {
        std::string pattern_scenes = ".*cloud.*.pcd";

        bf::path input_dir = pcd_files_dir_;
        std::string so_far = "";
        faat_pcl::utils::getFilesInDirectoryRecursive(input_dir, so_far, files, pattern_scenes);

        std::cout << "Number of aligned scenes is:" << files.size () << std::endl;

        std::sort(files.begin(), files.end());

        for(size_t i=0; i < files.size(); i++)
        {
            pcl::PointCloud<pcl::PointXYZRGB> RedGreenBlue;
            std::stringstream file_to_read;
            file_to_read << pcd_files_dir_ << "/" << files[i];
            std::cout << file_to_read.str() << std::endl;
            pcl::io::loadPCDFile(file_to_read.str(), RedGreenBlue);

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>(RedGreenBlue));

            clouds_.push_back(cloud_ptr);

        }
     }

    for(size_t i=0; i < clouds_.size(); i++)
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered (new pcl::PointCloud<pcl::PointXYZRGB>(*clouds_[i]));

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

        std::stringstream out_file;
        out_file << pcd_files_out << "/" << files[i];

        std::cout << out_file.str() << std::endl;
        pcl::io::savePCDFileBinary (out_file.str(), *cloud);
    }
}
