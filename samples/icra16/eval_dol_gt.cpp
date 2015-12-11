// deprecated
//-i /media/Data/datasets/TUW/icra_dol_evals -o /media/Data/datasets/TUW/icra_dol_evals/TUW_100.txt -r 0.02
//-i /media/Data/eval_tmp/willow_002 -o /media/Data/eval_tmp/willow_002.txt -r 0.02

// new
// -i /home/thomas/Desktop/out_TUW -g /media/Data/datasets/TUW/TUW_gt_models -o /home/thomas/TUW_icra16_results.txt -r 0.02 -s _dol.pcd

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
#include <boost/program_options.hpp>

namespace po = boost::program_options;

class Result
{
public:
    double recall_;
    double precision_;
    double error_;
    std::string model_id_;
    std::string view_id_;
};

bool hasEnding (const std::string  &fullString, const std::string &ending) {
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

template<typename PointT> double
computeRecall(const typename pcl::PointCloud<PointT>::ConstPtr &gt, const typename pcl::PointCloud<PointT>::ConstPtr &searchPoints, double &avg_error, float radius=0.005f)
{
    double error_accum_ = 0;
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
                    error_accum_ += sqrt(pointRadiusSquaredDistance[0]);
                    break;
                }
            }
        }
    }
    avg_error = error_accum_ / num_matches++;
    return static_cast<double>( num_matches ) / searchPoints->points.size();
}

int
main (int argc, char ** argv)
{
    typedef pcl::PointXYZRGB PointT;

    std::string input_dir, gt_dir, output_file, suffix = "_dol.pcd";
    float radius=0.005f;
    bool visualize = false;
    std::vector<Result> result;

    pcl::visualization::PCLVisualizer::Ptr vis_;

    po::options_description desc("Evaluation Dynamic Object Learning with Ground Truth\n======================================\n **Allowed options");
    desc.add_options()
            ("help,h", "produce help message")
            ("input_dir,i", po::value<std::string>(&input_dir)->required(), "directory containing the learnt object model as .pcd files")
            ("gt_dir,g", po::value<std::string>(&gt_dir)->required(), "directory containing the ground-truth object models as .pcd files")
            ("output_file,o", po::value<std::string>(&output_file)->required(), "output file")
            ("suffix,s", po::value<std::string>(&suffix)->default_value(suffix), "File name suffix for files containing the learnt object models")
            ("radius,r", po::value<float>(&radius)->default_value(radius), "directory containing the model .pcd files")
            ("visualize,v", po::bool_switch(&visualize), "turn visualization on")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        return false;
    }

    try
    {
        po::notify(vm);
    }
    catch(std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl;
        return false;
    }

    if(visualize)
        vis_.reset( new pcl::visualization::PCLVisualizer("gt and dol object") );

    std::vector< std::string> sub_folder_names;
    if(!v4r::io::getFoldersInDirectory( input_dir, "", sub_folder_names) )
    {
        std::cerr << "No subfolders in directory " << input_dir << ". " << std::endl;
        sub_folder_names.push_back("");
    }

    Result r_total;
    size_t total_num_obj = 0;

    std::sort(sub_folder_names.begin(), sub_folder_names.end());
    for (size_t sub_folder_id=0; sub_folder_id < sub_folder_names.size(); sub_folder_id++)
    {
        Result r;
        double prec_accum = 0;
        double rec_accum = 0;
        double error_accum = 0;
        size_t num_obj = 0;
        r.model_id_ = "";
        r.view_id_ = sub_folder_names[ sub_folder_id ];

        std::vector< std::string > obj_fn;
        const std::string sub_folder_learnt_objects = input_dir + "/" + sub_folder_names[ sub_folder_id ];
        const std::string sub_folder_gt_objects = gt_dir + "/" + sub_folder_names[ sub_folder_id ];
        v4r::io::getFilesInDirectory(sub_folder_learnt_objects, obj_fn, "", ".*" + suffix, false);


        std::sort(obj_fn.begin(), obj_fn.end());
        for (size_t o_id=0; o_id<obj_fn.size(); o_id++)
        {
            const std::string dol_fn = sub_folder_learnt_objects + "/" + obj_fn[ o_id ];

            if (hasEnding( dol_fn, "_ds_dol.pcd") ) // we don't want to process the downsampled version (used for finman et al)
                continue;

            pcl::PointCloud<PointT>::Ptr obj_dol (new pcl::PointCloud<PointT>());
            pcl::io::loadPCDFile(dol_fn, *obj_dol);

            std::string gt_fn = obj_fn[ o_id ];
            boost::replace_last (gt_fn, suffix, ".pcd");
            gt_fn = sub_folder_gt_objects + "/" + gt_fn;
            pcl::PointCloud<PointT>::Ptr obj_gt (new pcl::PointCloud<PointT>());
            pcl::io::loadPCDFile(gt_fn, *obj_gt);

            pcl::PointCloud<PointT>::Ptr src (new pcl::PointCloud<PointT>());
            pcl::PointCloud<PointT>::Ptr trgt (new pcl::PointCloud<PointT>());

            pcl::transformPointCloud(*obj_dol, *src, v4r::RotTrans2Mat4f(obj_dol->sensor_orientation_, obj_dol->sensor_origin_));
            pcl::transformPointCloud(*obj_gt, *trgt, v4r::RotTrans2Mat4f(obj_gt->sensor_orientation_, obj_gt->sensor_origin_));

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
            double avg_error_1, avg_error_2;
            res.precision_ = computeRecall<PointT>(trgt, icp_aligned_cloud, avg_error_1, radius);
            res.recall_ = computeRecall<PointT>(icp_aligned_cloud, trgt, avg_error_2, radius);
            res.model_id_ = obj_fn[ o_id ];
            res.view_id_ = sub_folder_names[ sub_folder_id ];
            res.error_ = (avg_error_1 + avg_error_2) / 2;

            error_accum += res.error_;
            rec_accum += res.recall_;
            prec_accum += res.precision_;
            num_obj++;

            std::cout << dol_fn << ": " << res.precision_ << " and " << res.recall_ << " e: " << avg_error_1 << " / " << avg_error_2 << std::endl;

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
        r.error_ = error_accum / num_obj;
        r.recall_ = rec_accum / num_obj;
        r.precision_ = prec_accum / num_obj;
        result.push_back(r);

        r_total.error_ += error_accum;
        r_total.recall_ += rec_accum;
        r_total.precision_ += prec_accum;
        total_num_obj += num_obj;
    }

    r_total.error_ /= static_cast<float>(total_num_obj);
    r_total.recall_ /= static_cast<float>(total_num_obj);
    r_total.precision_ /= static_cast<float>(total_num_obj);

    ofstream file;
    file.open (output_file.c_str());
    for(size_t i=0; i<result.size(); i++)
    {
        file << result[i].recall_ << " " << result[i].precision_ << " " << result[i].model_id_ << " " << result[i].view_id_ << " " << result[i].error_ << std::endl;
    }
    file << std::endl << r_total.recall_ << " " << r_total.precision_ << " " << r_total.model_id_ << " mean " << r_total.error_ << std::endl;
    file.close();
    return 0;
}
