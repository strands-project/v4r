
#include <v4r/io/filesystem.h>
#include <v4r/common/miscellaneous.h>
#include <pcl/console/parse.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/octree.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <iostream>
#include <fstream>

class Result
{
public:
    float recall_;
    float precision_;
    std::string model_id_;
    std::string view_id_;
};

template<typename PointT> float
computeRecall(const typename pcl::PointCloud<PointT>::ConstPtr &gt, const typename pcl::PointCloud<PointT>::ConstPtr &searchPoints, float radius=0.005f, size_t additional_missing_points = 0)
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
    return static_cast<float>( num_matches ) / (searchPoints->points.size() + additional_missing_points);
}

int
main (int argc, char ** argv)
{
    typedef pcl::PointXYZRGB PointT;
    std::string input_dir, gt_dir;
    float radius=0.005f;
    bool visualize = false;
    std::vector<Result> result;

    pcl::visualization::PCLVisualizer::Ptr vis_;

    pcl::console::parse_argument (argc, argv, "-visualize", visualize);
    pcl::console::parse_argument (argc, argv, "-input_dir", input_dir);
    pcl::console::parse_argument (argc, argv, "-radius", radius);
    pcl::console::parse_argument (argc, argv, "-gt_dir", gt_dir);

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

    std::sort(sub_folder_names.begin(), sub_folder_names.end());
    for (size_t sub_folder_id=0; sub_folder_id < sub_folder_names.size(); sub_folder_id++)
    {
        const std::string filename = input_dir + "/" + sub_folder_names[ sub_folder_id ] + "/cloud_0.pcd";
        const std::string mask_name = input_dir + "/" + sub_folder_names[ sub_folder_id ] + "/mask.txt";

        pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>());
        pcl::PointCloud<PointT>::Ptr masked_cloud (new pcl::PointCloud<PointT>());
        pcl::PointCloud<PointT>::Ptr masked_cloud_filtered (new pcl::PointCloud<PointT>());
        pcl::io::loadPCDFile(filename, *cloud);

        std::ifstream mask_file;
        mask_file.open( mask_name.c_str() );
        int idx_tmp;
        pcl::PointIndices pind;
        while (mask_file >> idx_tmp)
        {
            pind.indices.push_back(idx_tmp);
        }
        mask_file.close();
        pcl::copyPointCloud(*cloud, pind, *masked_cloud);
//        pcl::visualization::PCLVisualizer v;
//        v.addPointCloud(masked_cloud);
//        v.spin();

        size_t kept=0;
        masked_cloud_filtered->points.resize(masked_cloud->points.size());
        for(size_t i=0; i<masked_cloud->points.size(); i++)
        {
            if( pcl::isFinite(masked_cloud->points[i] ) )
            {
                masked_cloud_filtered->points[kept] = masked_cloud->points[i];
                kept++;
            }
        }
        std::cout << "Kept: " << kept << " of " <<    masked_cloud->points.size();
        masked_cloud_filtered->points.resize(kept);

        std::vector< std::string > obj_fn;
        v4r::io::getFilesInDirectory(gt_dir + "/" + sub_folder_names[ sub_folder_id ], obj_fn, "", ".*.pcd", false);
        const std::string gt_filename = gt_dir + "/" + sub_folder_names[ sub_folder_id ] + "/" + obj_fn[0];

        pcl::PointCloud<PointT>::Ptr gt_cloud (new pcl::PointCloud<PointT>());
        pcl::io::loadPCDFile(gt_filename, *gt_cloud);

        pcl::PointCloud<PointT>::Ptr masked_cloud_src (new pcl::PointCloud<PointT>());
        pcl::PointCloud<PointT>::Ptr targ_gt (new pcl::PointCloud<PointT>());

        pcl::transformPointCloud(*masked_cloud_filtered, *masked_cloud_src, v4r::common::RotTrans2Mat4f(cloud->sensor_orientation_, cloud->sensor_origin_));
        pcl::transformPointCloud(*gt_cloud, *targ_gt, v4r::common::RotTrans2Mat4f(gt_cloud->sensor_orientation_, gt_cloud->sensor_origin_));

        if(visualize)
        {
            vis_->removeAllPointClouds();
            vis_->addPointCloud(gt_cloud, "ground_truth");
//                vis_->addPointCloud(obj_dol, "dol");
            vis_->addPointCloud(masked_cloud_src, "icp aligned cloud");
//                vis_->addPointCloud(icp_aligned_cloud2, "icp aligned cloud2");
            vis_->spin();
        }

        pcl::IterativeClosestPoint<PointT, PointT> icp;
        icp.setInputSource(masked_cloud_src);
        icp.setInputTarget(targ_gt);
        icp.setMaxCorrespondenceDistance (0.02f);
//            icp.setMaximumIterations (50);
//            icp.setRANSACIterations(100);
//            icp.setRANSACOutlierRejectionThreshold(0.003);
//            icp.setEuclideanFitnessEpsilon(1e-9);
//            icp.setTransformationEpsilon(1e-9);
        pcl::PointCloud<PointT>::Ptr icp_aligned_cloud (new pcl::PointCloud<PointT>());
        pcl::PointCloud<PointT>::Ptr icp_aligned_cloud2 (new pcl::PointCloud<PointT>());
        icp.align(*icp_aligned_cloud, Eigen::Matrix4f::Identity());
            std::cout << "has converged:" << icp.hasConverged() << " score: " <<
                         icp.getFitnessScore() << std::endl;
            std::cout << icp.getFinalTransformation() << std::endl;
        pcl::transformPointCloud(*cloud, *icp_aligned_cloud2, icp.getFinalTransformation());

        Result res;
        res.precision_ = computeRecall<PointT>(targ_gt, icp_aligned_cloud, radius);
        res.recall_ = computeRecall<PointT>(icp_aligned_cloud, targ_gt, radius,  masked_cloud->points.size() - kept);
        res.model_id_ =  obj_fn[0];
        res.view_id_ = sub_folder_names[ sub_folder_id ];
        result.push_back(res);

        std::cout << sub_folder_names[ sub_folder_id ] << ": " << res.precision_ << " and " << res.recall_ << std::endl;


    }

    const std::string filename = "/tmp/initial_mask_eval.txt";
    ofstream file;
    file.open (filename.c_str());
    for(size_t i=0; i<result.size(); i++)
    {
        file << result[i].recall_ << " " << result[i].precision_ << " " << result[i].model_id_ << " " << result[i].view_id_ << std::endl;
    }
    file.close();
    return 0;
}
