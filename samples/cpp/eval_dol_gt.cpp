#include <v4r/io/filesystem.h>
#include <v4r/common/miscellaneous.h>
#include <pcl/console/parse.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/octree.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <iostream>
#include <fstream>

//-do_erosion 1 -radius 0.005 -dot_product 0.99 -normal_method 0 -chop_z 2 -transfer_latest_only 0 -do_sift_based_camera_pose_estimation 0 -scenes_dir /media/Data/datasets/TUW/test_set -input_mask_dir /home/thomas/Desktop/test -output_dir /home/thomas/Desktop/out_test/ -visualize 1

template<typename PointT> float
computeRecall(const typename pcl::PointCloud<PointT>::ConstPtr &gt, const typename pcl::PointCloud<PointT>::ConstPtr &searchPoints, float radius=0.005f)
{
    if (!searchPoints->points.size())   // if no test points, everything is correct by definition
        return 1.0f;

    pcl::octree::OctreePointCloudSearch<PointT> octree(radius);
    octree.setInputCloud( gt );
    octree.addPointsFromInputCloud();

    size_t num_matches=0;
    for(size_t i=0; i < searchPoints->points.size(); i++)
    {
        if ( ! pcl::isFinite(searchPoints->points[i]) )
        {
            PCL_WARN ("Warning: Point is NaN.\n");    // not sure if this causes somewhere else a problem. This condition should not be fulfilled.
            continue;
        }

        std::vector<int> pointIdxRadiusSearch;
        std::vector<float> pointRadiusSquaredDistance;

        if ( octree.radiusSearch (searchPoints->points[i], radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
        {
            for( size_t nn_id = 0; nn_id < pointIdxRadiusSearch.size(); nn_id++)
            {
                if( 1 ) // check for color or something
                {
                    num_matches++;
                    break;
                }
            }
        }
    }
    return static_cast<float>( num_matches ) / searchPoints->points.size();
}

int
main (int argc, char ** argv)
{
    typedef pcl::PointXYZRGB PointT;

    std::string input_dir;
    float radius=0.005f;
    bool visualize = false;
    pcl::visualization::PCLVisualizer::Ptr vis_;

    pcl::console::parse_argument (argc, argv,  "-visualize", visualize);
    pcl::console::parse_argument (argc, argv, "-input_dir", input_dir);
    pcl::console::parse_argument (argc, argv, "-radius", radius);

    if (input_dir.compare ("") == 0)
    {
        PCL_ERROR("Set the input directory. Usage -input_dir [path_to_dir].\n");
        return -1;
    }
    if(visualize)
        vis_.reset( new pcl::visualization::PCLVisualizer("gt and dol object") );

    std::vector< std::string> sub_folder_names;
    if(!v4r::io::getFoldersInDirectory( input_dir, "", sub_folder_names) )
    {
        std::cerr << "No subfolders in directory " << input_dir << ". " << std::endl;
        sub_folder_names.push_back("");
    }

    for (size_t sub_folder_id=0; sub_folder_id < sub_folder_names.size(); sub_folder_id++)
    {
        std::vector< std::string > obj_fn;
        const std::string sub_folder = input_dir + "/" + sub_folder_names[ sub_folder_id ];
        v4r::io::getFilesInDirectory(sub_folder, obj_fn, "", ".*_dol.pcd", false);
        for (size_t o_id=0; o_id<obj_fn.size(); o_id++)
        {
            const std::string dol_fn = sub_folder + "/" + obj_fn[ o_id ];
            pcl::PointCloud<PointT>::Ptr obj_dol (new pcl::PointCloud<PointT>());
            pcl::io::loadPCDFile(dol_fn, *obj_dol);

            std::string gt_fn = dol_fn;
            boost::replace_last (gt_fn, "_dol.pcd", ".pcd");
            pcl::PointCloud<PointT>::Ptr obj_gt (new pcl::PointCloud<PointT>());
            pcl::io::loadPCDFile(gt_fn, *obj_gt);

            pcl::transformPointCloud(*obj_dol, *obj_dol, v4r::common::RotTrans2Mat4f(obj_dol->sensor_orientation_, obj_dol->sensor_origin_) );
            pcl::transformPointCloud( *obj_gt,  *obj_gt, v4r::common::RotTrans2Mat4f( obj_gt->sensor_orientation_,  obj_gt->sensor_origin_) );

            Eigen::Vector4f zero_origin;
            zero_origin[0] = zero_origin[1] = zero_origin[2] = zero_origin[3] = 0.f;
            obj_gt->sensor_origin_ = zero_origin;   // for correct visualization
            obj_gt->sensor_orientation_ = Eigen::Quaternionf::Identity();
            obj_dol->sensor_origin_ = zero_origin;   // for correct visualization
            obj_dol->sensor_orientation_ = Eigen::Quaternionf::Identity();

            std::cout << obj_fn[ o_id ] << ": " << computeRecall<PointT>(obj_gt, obj_dol, radius) << " and " << computeRecall<PointT>(obj_dol, obj_gt, radius) << std::endl;

            if(visualize)
            {
                vis_->removeAllPointClouds();
                vis_->addPointCloud(obj_gt, "ground_truth");
                vis_->addPointCloud(obj_dol, "dol");
                vis_->spin();
            }
        }
    }
    return 0;
}
