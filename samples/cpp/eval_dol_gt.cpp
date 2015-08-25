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

    std::string input_dir, result_file;
    float radius=0.005f;
    bool visualize = false;
    std::vector<Result> result;

    pcl::visualization::PCLVisualizer::Ptr vis_;

    pcl::console::parse_argument (argc, argv, "-visualize", visualize);
    pcl::console::parse_argument (argc, argv, "-input_dir", input_dir);
    pcl::console::parse_argument (argc, argv, "-radius", radius);
    pcl::console::parse_argument (argc, argv, "-result_file", result_file);

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
        std::vector< std::string > obj_fn;
        const std::string sub_folder = input_dir + "/" + sub_folder_names[ sub_folder_id ];
        v4r::io::getFilesInDirectory(sub_folder, obj_fn, "", ".*_dol.pcd", false);

        std::sort(obj_fn.begin(), obj_fn.end());
        for (size_t o_id=0; o_id<obj_fn.size(); o_id++)
        {
            const std::string dol_fn = sub_folder + "/" + obj_fn[ o_id ];
            pcl::PointCloud<PointT>::Ptr obj_dol (new pcl::PointCloud<PointT>());
            pcl::io::loadPCDFile(dol_fn, *obj_dol);

            std::string gt_fn = dol_fn;
            boost::replace_last (gt_fn, "_dol.pcd", ".pcd");
            pcl::PointCloud<PointT>::Ptr obj_gt (new pcl::PointCloud<PointT>());
            pcl::io::loadPCDFile(gt_fn, *obj_gt);


            pcl::PointCloud<PointT>::Ptr src (new pcl::PointCloud<PointT>());
            pcl::PointCloud<PointT>::Ptr trgt (new pcl::PointCloud<PointT>());

            pcl::transformPointCloud(*obj_dol, *src, v4r::common::RotTrans2Mat4f(obj_dol->sensor_orientation_, obj_dol->sensor_origin_));
            pcl::transformPointCloud(*obj_gt, *trgt, v4r::common::RotTrans2Mat4f(obj_gt->sensor_orientation_, obj_gt->sensor_origin_));

            pcl::IterativeClosestPoint<PointT, PointT> icp;
            icp.setInputSource(src);
            icp.setInputTarget(trgt);
            icp.setMaxCorrespondenceDistance (0.02f);
//            icp.setMaximumIterations (50);
//            icp.setRANSACIterations(100);
//            icp.setRANSACOutlierRejectionThreshold(0.003);
//            icp.setEuclideanFitnessEpsilon(1e-9);
//            icp.setTransformationEpsilon(1e-9);
            pcl::PointCloud<PointT>::Ptr icp_aligned_cloud (new pcl::PointCloud<PointT>());
            pcl::PointCloud<PointT>::Ptr icp_aligned_cloud2 (new pcl::PointCloud<PointT>());
            icp.align(*icp_aligned_cloud, Eigen::Matrix4f::Identity());
//            std::cout << "has converged:" << icp.hasConverged() << " score: " <<
//                         icp.getFitnessScore() << std::endl;
//            std::cout << icp.getFinalTransformation() << std::endl;
            pcl::transformPointCloud(*obj_dol, *icp_aligned_cloud2, icp.getFinalTransformation());

            Result res;
            res.precision_ = computeRecall<PointT>(trgt, icp_aligned_cloud, radius);
            res.recall_ = computeRecall<PointT>(icp_aligned_cloud, trgt, radius);
            res.model_id_ = obj_fn[ o_id ];
            res.view_id_ = sub_folder_names[ sub_folder_id ];
            result.push_back(res);

            std::cout << dol_fn << ": " << res.precision_ << " and " << res.recall_ << std::endl;

            if(visualize)
            {
                vis_->removeAllPointClouds();
                vis_->addPointCloud(obj_gt, "ground_truth");
//                vis_->addPointCloud(obj_dol, "dol");
                vis_->addPointCloud(icp_aligned_cloud, "icp aligned cloud");
//                vis_->addPointCloud(icp_aligned_cloud2, "icp aligned cloud2");
                vis_->spin();
            }
        }
    }

    ofstream file;
    file.open (result_file.c_str());
    for(size_t i=0; i<result.size(); i++)
    {
        file << result[i].recall_ << " " << result[i].precision_ << " " << result[i].model_id_ << " " << result[i].view_id_ << std::endl;
    }
    file.close();
    return 0;
}
