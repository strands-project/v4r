/*
 * do_modelling.cpp
 *
 *  Created on: Mar 15, 2013
 *      Author: aitor
 */

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

#include <pcl/features/organized_edge_detection.h>
#include <pcl/features/integral_image_normal.h>
#include "seg_do_modelling.h"
#include <faat_pcl/registration/mv_lm_icp.h>
#include <pcl/features/normal_3d_omp.h>
#include <faat_pcl/utils/filesystem_utils.h>
//#include <faat_pcl/registration/feature_agnostic_icp_with_gc.h>
#include <faat_pcl/registration/icp_with_gc.h>
#include <faat_pcl/utils/noise_models.h>
#include <faat_pcl/registration/registration_utils.h>
#include <faat_pcl/utils/registration_utils.h>
#include <faat_pcl/utils/pcl_opencv.h>

struct IndexPoint
{
    int idx;
};

POINT_CLOUD_REGISTER_POINT_STRUCT (IndexPoint,
                                   (int, idx, idx)
                                   )

inline bool
readMatrixFromFile2 (std::string file, Eigen::Matrix4f & matrix, int ignore = 0)
{

    std::ifstream in;
    in.open (file.c_str (), std::ifstream::in);

    char linebuf[1024];
    in.getline (linebuf, 1024);
    std::string line (linebuf);
    std::vector<std::string> strs_2;
    boost::split (strs_2, line, boost::is_any_of (" "));

    int c = 0;
    for (int i = ignore; i < (ignore + 16); i++, c++)
    {
        matrix (c / 4, c % 4) = static_cast<float> (atof (strs_2[i].c_str ()));
    }

    return true;
}

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

void transformNormals(pcl::PointCloud<pcl::Normal>::Ptr & normals_cloud,
                      pcl::PointCloud<pcl::Normal>::Ptr & normals_aligned,
                      Eigen::Matrix4f & transform)
{
    normals_aligned.reset (new pcl::PointCloud<pcl::Normal>);
    normals_aligned->points.resize (normals_cloud->points.size ());
    normals_aligned->width = normals_cloud->width;
    normals_aligned->height = normals_cloud->height;
    for (size_t k = 0; k < normals_cloud->points.size (); k++)
    {
        Eigen::Vector3f nt (normals_cloud->points[k].normal_x, normals_cloud->points[k].normal_y, normals_cloud->points[k].normal_z);
        normals_aligned->points[k].normal_x = static_cast<float> (transform (0, 0) * nt[0] + transform (0, 1) * nt[1]
                                                                  + transform (0, 2) * nt[2]);
        normals_aligned->points[k].normal_y = static_cast<float> (transform (1, 0) * nt[0] + transform (1, 1) * nt[1]
                                                                  + transform (1, 2) * nt[2]);
        normals_aligned->points[k].normal_z = static_cast<float> (transform (2, 0) * nt[0] + transform (2, 1) * nt[1]
                                                                  + transform (2, 2) * nt[2]);
    }
}

template<class PointT>
inline void
computeRGBEdges (typename pcl::PointCloud<PointT>::Ptr & cloud, std::vector<int> & indices,
                 float low = 100.f, float high = 150.f, bool sobel = false)
{
    if(sobel)
    {
        cv::Mat_ < cv::Vec3b > colorImage;
        PCLOpenCV::ConvertPCLCloud2Image<pcl::PointXYZRGB> (cloud, colorImage);

        //cv::namedWindow("test");
        //cv::imshow("test", colorImage);

        cv::Mat grad_x, grad_y;
        cv::Mat abs_grad_x, abs_grad_y;
        int scale = 1;
        int delta = 0;
        int ddepth = CV_16S;
        int kernel_size = 3;

        /// Gradient X
        //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
        cv::Mat src_gray;
        cv::cvtColor( colorImage, src_gray, CV_RGB2GRAY );
        cv::Sobel( src_gray, grad_x, ddepth, 1, 0, kernel_size, scale, delta, cv::BORDER_DEFAULT );
        cv::convertScaleAbs( grad_x, abs_grad_x );

        /// Gradient Y
        //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
        cv::Sobel( src_gray, grad_y, ddepth, 0, 1, kernel_size, scale, delta, cv::BORDER_DEFAULT );
        cv::convertScaleAbs( grad_y, abs_grad_y );

        /// Total Gradient (approximate)
        cv::Mat grad;
        cv::addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
        //    cv::namedWindow("sobel");
        //    cv::imshow("sobel", grad);

        cv::Mat thresholded_gradients = src_gray;
        for ( int j=0; j<abs_grad_x.rows; j++)
        {
            for (int i=0; i<abs_grad_x.cols; i++)
            {
                thresholded_gradients.at<unsigned char>(j,i) = 0;

                /*if(abs_grad_x.at<unsigned char>(j,i) > low || abs_grad_y.at<unsigned char>(j,i) > low)
          {
            thresholded_gradients.at<unsigned char>(j,i) = 255;
          }*/

                if(grad.at<unsigned char>(j,i) > low)
                {
                    thresholded_gradients.at<unsigned char>(j,i) = 255;
                }
            }
        }

        //cv::erode(thresholded_gradients, thresholded_gradients, cv::Mat());
        //cv::dilate(thresholded_gradients, thresholded_gradients, cv::Mat());

        //    cv::namedWindow("sobel_thres");
        //    cv::imshow("sobel_thres", thresholded_gradients);
        //    cv::waitKey(0);

        for ( int j=0; j<abs_grad_x.rows; j++)
        {
            for (int i=0; i<abs_grad_x.cols; i++)
            {
                if(thresholded_gradients.at<unsigned char>(j,i) == 255)
                {
                    indices.push_back(j*cloud->width + i);
                }
            }
        }
    }
    else
    {
        pcl::OrganizedEdgeFromRGB<PointT, pcl::Label> oed;
        oed.setRGBCannyLowThreshold (low);
        oed.setRGBCannyHighThreshold (high);
        oed.setEdgeType (pcl::OrganizedEdgeBase<PointT, pcl::Label>::EDGELABEL_RGB_CANNY);
        oed.setInputCloud (cloud);

        pcl::PointCloud<pcl::Label>::Ptr labels (new pcl::PointCloud<pcl::Label>);
        std::vector<pcl::PointIndices> indices2;
        oed.compute (*labels, indices2);

        for (size_t j = 0; j < indices2.size (); j++)
        {
            if (indices2[j].indices.size () > 0)
            {
                for(size_t k=0; k < indices2[j].indices.size (); k++)
                {
                    if(pcl_isfinite(cloud->points[indices2[j].indices[k]].z))
                        indices.push_back(indices2[j].indices[k]);
                }
            }
        }
    }
}

//./bin/object_modelling_willow -pcd_files_dir /home/aitor/willow_challenge_ros_code/read_willow_data/train/object_15/ -Z_DIST 0.8 -num_plane_inliers 2000 -max_corresp_dist 0.01 -vx_size 0.003 -dt_size 0.003 -visualize 0 -fast_overlap 1 -aligned_output_dir /home/aitor/data/willow_structure/object_15.pcd -aligned_model_saved_to /home/aitor/data/willow_object_clouds/models_ml_new/object_15.pcd
// ./bin/object_modelling_willow_unknown_poses -pcd_files_dir /media/DATA/models/frucht_molke_seq2/bin/ -Z_DIST 1 -low 100 -high 150 -iterations 15 -max_corresp_dist 0.05 -vis_final 1 -step 1 -vis_pairwise_ 0 -aligned_output_dir /media/DATA/models/frucht_molke_seq2_aligned

int
main (int argc, char ** argv)
{
    float Z_DIST_ = 1.5f;
    std::string pcd_files_dir_;
    bool sort_pcd_files_ = true;
    bool use_max_cluster_ = true;
    float data_scale = 1.f;
    float x_limits = 0.4f;
    int num_plane_inliers = 500;
    bool single_object = true;
    std::string aligned_output_dir = "";
    float max_corresp_dist_ = 0.1f;
    float voxel_grid_size = 0.003f;
    float dt_size = voxel_grid_size;
    float visualize = false;
    bool fast_overlap = false;
    int ignore = 1;
    int max_vis = std::numeric_limits<int>::max();
    int iterations = 10;

    float canny_low = 100.f;
    float canny_high = 150.f;
    bool sobel = false;

    bool use_cg_ = true;
    bool vis_final_ = false;
    float views_overlap_ = 0.3f;
    bool mv_use_weights_ = true;
    bool vis_pairwise_ = false;
    bool use_shot_ = false;

    aligned_output_dir = "test_modelling";

    pcl::console::parse_argument (argc, argv, "-use_shot_", use_shot_);
    pcl::console::parse_argument (argc, argv, "-use_cg", use_cg_);
    pcl::console::parse_argument (argc, argv, "-iterations", iterations);
    pcl::console::parse_argument (argc, argv, "-low", canny_low);
    pcl::console::parse_argument (argc, argv, "-high", canny_high);
    pcl::console::parse_argument (argc, argv, "-ignore", ignore);
    pcl::console::parse_argument (argc, argv, "-pcd_files_dir", pcd_files_dir_);
    pcl::console::parse_argument (argc, argv, "-sort_pcd_files", sort_pcd_files_);
    pcl::console::parse_argument (argc, argv, "-use_max_cluster", use_max_cluster_);
    pcl::console::parse_argument (argc, argv, "-data_scale", data_scale);
    pcl::console::parse_argument (argc, argv, "-x_limits", x_limits);
    pcl::console::parse_argument (argc, argv, "-Z_DIST", Z_DIST_);
    pcl::console::parse_argument (argc, argv, "-num_plane_inliers", num_plane_inliers);
    pcl::console::parse_argument (argc, argv, "-single_object", single_object);
    pcl::console::parse_argument (argc, argv, "-vis_final", vis_final_);
    pcl::console::parse_argument (argc, argv, "-aligned_output_dir", aligned_output_dir);

    pcl::console::parse_argument (argc, argv, "-max_corresp_dist", max_corresp_dist_);
    pcl::console::parse_argument (argc, argv, "-vx_size", voxel_grid_size);
    pcl::console::parse_argument (argc, argv, "-dt_size", dt_size);
    pcl::console::parse_argument (argc, argv, "-visualize", visualize);
    pcl::console::parse_argument (argc, argv, "-fast_overlap", fast_overlap);
    pcl::console::parse_argument (argc, argv, "-max_vis", max_vis);
    pcl::console::parse_argument (argc, argv, "-sobel", sobel);
    pcl::console::parse_argument (argc, argv, "-views_overlap", views_overlap_);
    pcl::console::parse_argument (argc, argv, "-mv_use_weights", mv_use_weights_);
    pcl::console::parse_argument (argc, argv, "-vis_pairwise_", vis_pairwise_);

    bf::path aligned_output = aligned_output_dir;
    if(!bf::exists(aligned_output))
    {
        bf::create_directory(aligned_output);
    }

    float w_t = 0.75f;
    float max_angle = 60.f;
    float lateral_sigma = 0.002f;
    bool organized_normals = true;
    int seg_type = 0;
    float min_dot = 0.98f;
    bool do_multiview = true;
    bool vis_segmented = false;
    float plane_threshold = 0.01f;

    pcl::console::parse_argument (argc, argv, "-w_t", w_t);
    pcl::console::parse_argument (argc, argv, "-max_angle", max_angle);
    pcl::console::parse_argument (argc, argv, "-lateral_sigma", lateral_sigma);
    pcl::console::parse_argument (argc, argv, "-organized_normals", organized_normals);
    pcl::console::parse_argument (argc, argv, "-min_dot", min_dot);
    pcl::console::parse_argument (argc, argv, "-seg_type", seg_type);
    pcl::console::parse_argument (argc, argv, "-do_multiview", do_multiview);
    pcl::console::parse_argument (argc, argv, "-vis_segmented", vis_segmented);
    pcl::console::parse_argument (argc, argv, "-plane_threshold", plane_threshold);

    int step = 1;
    int mv_iterations = 10;

    pcl::console::parse_argument (argc, argv, "-mv_iterations", mv_iterations);
    pcl::console::parse_argument (argc, argv, "-step", step);
    std::vector<std::string> files;
    bf::path input_path = pcd_files_dir_;
    std::vector<std::string> model_files;
    std::string pattern_models = ".*cloud_.*.pcd";
    std::string relative_path = "";
    faat_pcl::utils::getFilesInDirectory(input_path, model_files, pattern_models);

    typedef pcl::PointXYZRGB PointType;
    typedef pcl::PointXYZRGBNormal PointTypeNormal;
    typedef pcl::PointXYZRGBNormal PointTInternal;

    if (sort_pcd_files_)
        std::sort (model_files.begin (), model_files.end ());

    model_files.resize(std::min((int)model_files.size(), max_vis));
    std::cout << model_files.size() << std::endl;
    if(step != 1)
    {
        int kept = 0;
        for (size_t i = 0; i < model_files.size (); i+=step)
        {
            model_files[kept] = model_files[i];
            kept++;
        }

        model_files.resize(kept);
        std::cout << "step is set" << model_files.size() << std::endl;
    }

    std::vector<pcl::PointCloud<PointType>::Ptr> clouds_;
    clouds_.resize (model_files.size ());

    std::vector<pcl::PointCloud<PointTInternal>::Ptr> xyz_normals_;
    xyz_normals_.resize (model_files.size ());

    std::vector<pcl::PointCloud<PointType>::Ptr> range_images_;
    std::vector<pcl::PointCloud<PointType>::Ptr> edges_;
    std::vector<std::vector<int> > obj_indices_;
    std::vector<Eigen::Matrix4f> poses;
    std::vector<pcl::PointCloud<pcl::Normal>::Ptr > clouds_normals_;
    std::vector<std::vector<bool> > obj_masks_;
    std::vector<pcl::PointCloud<PointType>::Ptr> filtered_clouds_;
    std::vector<pcl::PointCloud<PointType>::Ptr> filtered_clouds_threshold;
    std::vector< std::vector<float> > original_weights;
    std::vector< pcl::IndicesPtr > indices_views_;
    indices_views_.resize (model_files.size ());

    for (size_t i = 0; i < model_files.size (); i++)
    {
        pcl::PointCloud<PointType>::Ptr scene (new pcl::PointCloud<PointType>);
        pcl::PointCloud<PointType>::Ptr rimage (new pcl::PointCloud<PointType>);
        std::stringstream file_to_read;
        file_to_read << pcd_files_dir_ << "/" << model_files[i];
        pcl::io::loadPCDFile (file_to_read.str (), *scene);
        pcl::copyPointCloud(*scene, *rimage);

        range_images_.push_back(rimage);
        //segment the object of interest
        pcl::PassThrough<PointType> pass_;
        pass_.setFilterLimits (0.f, Z_DIST_);
        pass_.setFilterFieldName ("z");
        pass_.setInputCloud (scene);
        pass_.setKeepOrganized (true);
        pass_.filter (*scene);

        if (x_limits > 0)
        {
            pass_.setInputCloud (scene);
            pass_.setFilterLimits (-x_limits, x_limits);
            pass_.setFilterFieldName ("x");
            pass_.filter (*scene);
        }

        std::vector<pcl::PointIndices> indices;
        Eigen::Vector4f table_plane;
        doSegmentation<PointType> (scene, indices, table_plane, num_plane_inliers, seg_type, plane_threshold);

        std::cout << "Number of clusters found:" << indices.size () << std::endl;
        pcl::PointIndices max;
        for (size_t k = 0; k < indices.size (); k++)
        {
            if (max.indices.size () < indices[k].indices.size ())
            {
                max = indices[k];
            }
        }

        pcl::PointCloud<PointType>::Ptr obj_interest (new pcl::PointCloud<PointType>(*scene));
        for (int j = 0; j < scene->points.size (); j++)
        {
            Eigen::Vector3f xyz_p = scene->points[j].getVector3fMap ();

            if (!pcl_isfinite (xyz_p[0]) || !pcl_isfinite (xyz_p[1]) || !pcl_isfinite (xyz_p[2]))
                continue;

            float val = xyz_p[0] * table_plane[0] + xyz_p[1] * table_plane[1] + xyz_p[2] * table_plane[2] + table_plane[3];

            if (val < -0.01)
            {
                obj_interest->points[j].x = std::numeric_limits<float>::quiet_NaN ();
                obj_interest->points[j].y = std::numeric_limits<float>::quiet_NaN ();
                obj_interest->points[j].z = std::numeric_limits<float>::quiet_NaN ();
            }
        }

        /*for (size_t k = 0; k < max.indices.size (); k++)
    {
      obj_interest->points[max.indices[k]] = scene->points[max.indices[k]];
    }*/

        //pcl::copyPointCloud(*scene, max, *obj_interest);
        obj_indices_.push_back (max.indices);
        obj_masks_.push_back(registration_utils::indicesToMask(max.indices, obj_interest->points.size(), false));
        clouds_[i] = obj_interest;
    }

    {
        for(size_t i=0; i < std::min((int)clouds_.size(), max_vis); i++)
        {

            pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

            if(organized_normals)
            {
                std::cout << "Organized normals" << std::endl;
                pcl::IntegralImageNormalEstimation<PointType, pcl::Normal> ne;
                ne.setNormalEstimationMethod (ne.COVARIANCE_MATRIX);
                ne.setMaxDepthChangeFactor (0.02f);
                ne.setNormalSmoothingSize (20.0f);
                ne.setBorderPolicy (pcl::IntegralImageNormalEstimation<PointType, pcl::Normal>::BORDER_POLICY_MIRROR);
                ne.setInputCloud (clouds_[i]);
                ne.compute (*normals);
            }
            else
            {
                std::cout << "Not organized normals" << std::endl;
                pcl::NormalEstimationOMP<PointType, pcl::Normal> ne;
                ne.setInputCloud (clouds_[i]);
                pcl::search::KdTree<PointType>::Ptr tree (new pcl::search::KdTree<PointType> ());
                ne.setSearchMethod (tree);
                ne.setRadiusSearch (0.02);
                ne.compute (*normals);
            }

            clouds_normals_.push_back(normals);
            std::vector<int> cedges;
            computeRGBEdges<PointType>(clouds_[i], cedges, canny_low, canny_high, sobel);

            faat_pcl::utils::noise_models::NguyenNoiseModel<PointType> nm;
            nm.setInputCloud(clouds_[i]);
            nm.setInputNormals(normals);
            nm.setLateralSigma(lateral_sigma);
            nm.setMaxAngle(max_angle);
            nm.setUseDepthEdges(true);
            nm.compute();
            std::vector<float> weights;
            nm.getWeights(weights);
            original_weights.push_back(weights);

            {
                pcl::PointCloud<PointType>::Ptr filtered;
                nm.getFilteredCloudRemovingPoints(filtered, w_t);
                filtered_clouds_threshold.push_back(filtered);
            }

            pcl::PointCloud<PointType>::Ptr filtered;
            nm.getFilteredCloudRemovingPoints(filtered, 0.2f);
            filtered_clouds_.push_back(filtered);

            std::vector<int> filtered_cedges;
            int n_filtered = 0;
            for(size_t k=0; k < cedges.size(); k++)
            {
                if(weights[cedges[k]] > w_t)
                    filtered_cedges.push_back(cedges[k]);
                else
                    n_filtered++;
            }

            std::cout << "n_filtered:" << n_filtered << " kept:" << filtered_cedges.size() << std::endl;
            cedges = filtered_cedges;

            std::vector<float> orig_weights = original_weights[i];
            for(size_t k=0; k < original_weights[i].size(); k++)
                original_weights[i][k] *= 0.5f;

            for(size_t k=0; k < cedges.size(); k++)
            {
                if(!obj_masks_[i][cedges[k]])
                {
                    //if the edge does not belong to the object, dont change weight
                    continue;
                }

                original_weights[i][cedges[k]] = orig_weights[cedges[k]];
            }

            std::vector<bool> cedges_mask = registration_utils::indicesToMask(cedges, clouds_[i]->points.size(), false);

            pcl::PointCloud<PointType>::Ptr edges_cloud (new pcl::PointCloud<PointType>);
            pcl::PointCloud<pcl::Normal> normals_edges_cloud;

            pcl::copyPointCloud(*clouds_[i], cedges, *edges_cloud);
            pcl::copyPointCloud(*normals, cedges, normals_edges_cloud);

            pcl::UniformSampling<PointType> keypoint_extractor;
            keypoint_extractor.setRadiusSearch(0.02f);
            keypoint_extractor.setInputCloud (edges_cloud);
            pcl::PointCloud<int> keypoints_idxes_src;
            keypoint_extractor.compute (keypoints_idxes_src);

            std::vector<int> idxes;
            idxes.resize(keypoints_idxes_src.points.size());
            for(size_t kk=0; kk < idxes.size(); kk++)
              idxes[kk] = keypoints_idxes_src.points[kk];

            indices_views_[i].reset(new std::vector<int>(idxes));

            clouds_[i] = edges_cloud;
            obj_indices_[i] = registration_utils::maskToIndices(cedges_mask);
            xyz_normals_[i].reset(new pcl::PointCloud<PointTInternal>);
            pcl::copyPointCloud(*edges_cloud, *xyz_normals_[i]);
            pcl::copyPointCloud(normals_edges_cloud, *xyz_normals_[i]);
        }
    }

    if(vis_segmented)
    {
        pcl::visualization::PCLVisualizer vis ("segmented object");
        for(size_t i=1; i < clouds_.size(); i++)
        {
            std::stringstream cloud_name;
            cloud_name << "cloud_" << i << ".pcd";

            std::vector<int> obj_indices_original = registration_utils::maskToIndices(obj_masks_[i]);
            pcl::PointCloud<PointType>::Ptr masked_cloud (new pcl::PointCloud<PointType>);
            pcl::copyPointCloud(*range_images_[i], obj_indices_original, *masked_cloud);
            pcl::visualization::PointCloudColorHandlerRGBField<PointType> handler_rgb (masked_cloud);
            vis.addPointCloud<PointType> (masked_cloud, handler_rgb, cloud_name.str());
        }

        vis.spin();
    }

    {
        pcl::visualization::PCLVisualizer vis ("");
        int v1,v2, v3, v4;
        vis.createViewPort(0,0,0.5,0.5,v1);
        vis.createViewPort(0.5,0,1,0.5,v2);
        vis.createViewPort(0,0.5,0.5,1,v3);
        vis.createViewPort(0.5,0.5,1,1,v4);
        vis.setBackgroundColor(255, 255, 255);

        poses.push_back(Eigen::Matrix4f::Identity());

        for(size_t i=1; i < clouds_.size(); i++)
        {
            faat_pcl::IterativeClosestPointWithGC<PointTInternal, PointTInternal> icp;
            icp.setTransformationEpsilon (0.000001 * 0.000001);
            icp.setMinNumCorrespondences (3);
            icp.setMaxCorrespondenceDistance (max_corresp_dist_);
            icp.setUseCG (use_cg_);
            icp.setSurvivalOfTheFittest (false);
            icp.setMaximumIterations(iterations);
            icp.setOverlapPercentage(0.5);
            icp.setVisFinal(false);
            icp.setDtVxSize(dt_size);
            icp.setSourceAndTargetIndices(indices_views_[i], indices_views_[i - 1]);
            icp.setUseSHOT(use_shot_);

            icp.setRangeImages<PointType>(range_images_[i], range_images_[i-1], 525.f, 640, 480);
            pcl::registration::DefaultConvergenceCriteria<float>::Ptr convergence_criteria;
            convergence_criteria = icp.getConvergeCriteria ();
            convergence_criteria->setAbsoluteMSE (1e-12);
            convergence_criteria->setMaximumIterationsSimilarTransforms (iterations);
            convergence_criteria->setFailureAfterMaximumIterations (false);

            icp.setInputTarget (xyz_normals_[i-1]);
            icp.setInputSource (xyz_normals_[i]);

            typename pcl::PointCloud<PointTInternal>::Ptr pp_out(new pcl::PointCloud<PointTInternal>);
            icp.align (*pp_out);
            std::vector<std::pair<float, Eigen::Matrix4f> > res;
            icp.getResults(res);

            std::stringstream cloud_name;
            cloud_name << "cloud_" << i << ".pcd";

            {
                cloud_name << "v2";
                pcl::visualization::PointCloudColorHandlerRGBField<PointType> handler_rgb (filtered_clouds_[i-1]);
                vis.addPointCloud<PointType> (filtered_clouds_[i-1], handler_rgb, cloud_name.str (), v3);
            }

            {
                cloud_name << "v2";
                pcl::visualization::PointCloudColorHandlerRGBField<PointType> handler_rgb (filtered_clouds_[i-1]);
                vis.addPointCloud<PointType> (filtered_clouds_[i-1], handler_rgb, cloud_name.str (), v4);
            }

            {
                cloud_name << "v2";
                pcl::visualization::PointCloudColorHandlerRGBField<PointType> handler_rgb (filtered_clouds_[i]);
                vis.addPointCloud<PointType> (filtered_clouds_[i], handler_rgb, cloud_name.str (), v3);
            }

            {
                cloud_name << "v2";
                pcl::visualization::PointCloudColorHandlerCustom<PointType> handler_rgb (clouds_[i-1], 255, 0,0);
                vis.addPointCloud<PointType> (clouds_[i-1], handler_rgb, cloud_name.str (), v2);
            }

            {
                cloud_name << "v2";
                pcl::visualization::PointCloudColorHandlerCustom<PointType> handler_rgb (clouds_[i-1], 255, 0,0);
                vis.addPointCloud<PointType> (clouds_[i-1], handler_rgb, cloud_name.str (), v1);
            }

            pcl::PointCloud<PointType>::Ptr aligned (new pcl::PointCloud<PointType>);
            pcl::transformPointCloud(*clouds_[i], *aligned, res[0].second);

            {

                pcl::PointCloud<PointType>::Ptr aligned (new pcl::PointCloud<PointType>);
                pcl::transformPointCloud(*filtered_clouds_[i], *aligned, res[0].second);

                cloud_name << "v4";
                pcl::visualization::PointCloudColorHandlerRGBField<PointType> handler_rgb (aligned);
                vis.addPointCloud<PointType> (aligned, handler_rgb, cloud_name.str (), v4);
            }

            cloud_name << "cloud_aligned" << i << ".pcd";

            {
                //pcl::visualization::PointCloudColorHandlerRGBField<PointType> handler_rgb (aligned);
                pcl::visualization::PointCloudColorHandlerCustom<PointType> handler_rgb (aligned, 0, 0, 255);
                vis.addPointCloud<PointType> (aligned, handler_rgb, cloud_name.str (), v2);
            }

            cloud_name << "original";

            {
                //pcl::visualization::PointCloudColorHandlerRGBField<PointType> handler_rgb (aligned);
                pcl::visualization::PointCloudColorHandlerCustom<PointType> handler_rgb (clouds_[i], 0, 0, 255);
                vis.addPointCloud<PointType> (clouds_[i], handler_rgb, cloud_name.str (), v1);
            }

            {
                std::vector<int> obj_indices_original = registration_utils::maskToIndices(obj_masks_[i]);
                pcl::PointCloud<PointType>::Ptr masked_cloud (new pcl::PointCloud<PointType>);
                pcl::copyPointCloud(*range_images_[i], obj_indices_original, *masked_cloud);
                pcl::visualization::PointCloudColorHandlerCustom<PointType> handler_rgb (masked_cloud, 0, 255, 0);
                vis.addPointCloud<PointType> (masked_cloud, handler_rgb, "masked out", v3);
            }

            /*{
        pcl::visualization::PointCloudColorHandlerRGBField<PointType> handler_rgb (clouds_[i]);
        vis.addPointCloud<PointType> (clouds_[i], handler_rgb, cloud_name.str (), v1);
      }*/

            if(vis_pairwise_)
            {
                vis.spin();
            }
            else
            {
                vis.spinOnce(300, true);
            }
            vis.removeAllPointClouds();

            poses.push_back(res[0].second);

        }
    }

    pcl::visualization::PCLVisualizer vis ("");
    int v1,v2;
    vis.createViewPort(0,0,0.5,1,v1);
    vis.createViewPort(0.5,0,1,1,v2);
    vis.setBackgroundColor(255, 255, 255);

    assert(poses.size() == clouds_.size());

    pcl::PointCloud<PointType>::Ptr accumulated_cloud (new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr accumulated_cloud_aligned (new pcl::PointCloud<PointType>);
    std::vector<pcl::PointCloud<PointType>::Ptr> clouds_aligned;
    std::vector<pcl::PointCloud<pcl::Normal>::Ptr> normalclouds_aligned;
    std::vector< std::vector<float> > weights;
    weights.resize(std::min((int)clouds_.size(), max_vis));

    Eigen::Matrix4f accum = Eigen::Matrix4f::Identity();

    for(size_t i=0; i < clouds_.size(); i++)
    {
        *accumulated_cloud += *clouds_[i];
        accum = accum * poses[i];

        std::vector<int> obj_indices_original = registration_utils::maskToIndices(obj_masks_[i]);
        /*obj_indices_original.insert(obj_indices_original.end(), obj_indices_[i].begin(), obj_indices_[i].end());
    std::sort (obj_indices_original.begin (), obj_indices_original.end ());
    obj_indices_original.erase (std::unique (obj_indices_original.begin (), obj_indices_original.end ()), obj_indices_original.end ());*/

        std::vector<int> obj_indices_original_no_nans;
        for(size_t k=0; k < obj_indices_original.size(); k++)
        {
            if(!pcl_isfinite(filtered_clouds_threshold[i]->points[obj_indices_original[k]].z))
                continue;

            obj_indices_original_no_nans.push_back(obj_indices_original[k]);
        }

        pcl::PointCloud<PointType>::Ptr aligned (new pcl::PointCloud<PointType>);
        pcl::transformPointCloud(*filtered_clouds_threshold[i], *aligned, accum);

        pcl::PointCloud<PointType>::Ptr aligned_obj (new pcl::PointCloud<PointType>);
        pcl::copyPointCloud(*aligned, obj_indices_original_no_nans, *aligned_obj);

        pcl::PointCloud<pcl::Normal>::Ptr normals_aligned (new pcl::PointCloud<pcl::Normal>);
        transformNormals(clouds_normals_[i], normals_aligned, accum);

        pcl::PointCloud<pcl::Normal>::Ptr normals_aligned_obj (new pcl::PointCloud<pcl::Normal>);
        pcl::copyPointCloud(*normals_aligned, obj_indices_original_no_nans, *normals_aligned_obj);

        for(size_t k=0; k < obj_indices_original_no_nans.size(); k++)
        {
            weights[i].push_back(original_weights[i][obj_indices_original_no_nans[k]]);
        }

        /*pcl::PointCloud<PointType>::Ptr aligned (new pcl::PointCloud<PointType>);
    pcl::transformPointCloud(*range_images_[i], *aligned, accum);

    pcl::PointCloud<PointType>::Ptr aligned_obj (new pcl::PointCloud<PointType>);
    pcl::copyPointCloud(*aligned, obj_indices_[i], *aligned_obj);

    pcl::PointCloud<pcl::Normal>::Ptr normals_aligned (new pcl::PointCloud<pcl::Normal>);
    transformNormals(clouds_normals_[i], normals_aligned, accum);

    pcl::PointCloud<pcl::Normal>::Ptr normals_aligned_obj (new pcl::PointCloud<pcl::Normal>);
    pcl::copyPointCloud(*normals_aligned, obj_indices_[i], *normals_aligned_obj);

    for(size_t k=0; k < obj_indices_[i].size(); k++)
    {
      weights[i].push_back(original_weights[i][obj_indices_[i][k]]);
    }*/

        //std::cout << obj_indices_[i].size() << " " << aligned->points.size()
        *accumulated_cloud_aligned += *aligned_obj;

        clouds_aligned.push_back(aligned_obj);
        normalclouds_aligned.push_back(normals_aligned_obj);
        poses[i] = accum;
    }

    vis.removeAllPointClouds();

    {
        pcl::visualization::PointCloudColorHandlerRGBField<PointType> handler_rgb (accumulated_cloud_aligned);
        vis.addPointCloud<PointType> (accumulated_cloud_aligned, handler_rgb, "aligned_accum", v2);
    }

    {
        pcl::visualization::PointCloudColorHandlerRGBField<PointType> handler_rgb (accumulated_cloud);
        vis.addPointCloud<PointType> (accumulated_cloud, handler_rgb, "accum", v1);
    }

    if(vis_final_)
    {
        vis.spin();
    }
    else
    {
        vis.spinOnce(1000, true);
    }

    if(do_multiview)
    {
        //multiview refinement
        std::vector < std::vector<bool> > A;
        A.resize (clouds_aligned.size ());
        for (size_t i = 0; i < clouds_aligned.size (); i++)
            A[i].resize (clouds_aligned.size (), true);

        float ff=views_overlap_;
        faat_pcl::registration_utils::computeOverlapMatrix<PointType> (clouds_aligned, A, 0.01f, fast_overlap, ff);

        for (size_t i = 0; i < clouds_aligned.size (); i++)
        {
            for (size_t j = 0; j < clouds_aligned.size (); j++)
                std::cout << A[i][j] << " ";

            std::cout << std::endl;
        }

        /*std::vector<bool> valid_scan;
        valid_scan.resize(clouds_aligned.size());
        bool all_valid = true;

        for (size_t i = 0; i < clouds_aligned.size (); i++)
        {
            bool valid = false;
            for(size_t j=0; j < clouds_aligned.size (); j++)
            {
                if(A[i][j])
                    valid = true;
            }

            valid_scan[i] = valid;
            if(!valid)
                all_valid = false;

        }

        if(!all_valid)
        {
            std::cout << "Not all scans are valid..." << std::endl;
            int n_valid = 0;
            for (size_t i = 0; i < clouds_aligned.size (); i++)
            {
                std::cout << "valid scan:" << (int)valid_scan[i] << std::endl;
                if(valid_scan[i])
                {
                    clouds_aligned[n_valid] = clouds_aligned[i];
                    normalclouds_aligned[n_valid] = normalclouds_aligned[i];
                    poses[n_valid] = poses[i];
                    weights[n_valid] = weights[i];
                    range_images_[n_valid] = range_images_[i];
                    obj_masks_[n_valid] = obj_masks_[i];
                    std::cout << "copying " << i << " to " << n_valid << std::endl;
                    n_valid++;
                }
                else
                {
                }
            }

            clouds_aligned.resize(n_valid);
            normalclouds_aligned.resize(n_valid);
            poses.resize(n_valid);
            weights.resize(n_valid);
            range_images_.resize(n_valid);
            obj_masks_.resize(n_valid);

            A.resize (clouds_aligned.size ());
            for (size_t i = 0; i < clouds_aligned.size (); i++)
                A[i].resize (clouds_aligned.size (), true);

            float ff=views_overlap_;
            faat_pcl::registration_utils::computeOverlapMatrix<PointType> (clouds_aligned, A, 0.01f, fast_overlap, ff);

            for (size_t i = 0; i < clouds_aligned.size (); i++)
            {
                for (size_t j = 0; j < clouds_aligned.size (); j++)
                    std::cout << A[i][j] << " ";

                std::cout << std::endl;
            }
        }*/

        float max_corresp_dist_mv_ = 0.01f;
        float dt_size_mv_ = 0.002f;
        float inlier_threshold_mv = 0.002f;
        faat_pcl::registration::MVNonLinearICP<PointType> icp_nl (dt_size_mv_);
        icp_nl.setInlierThreshold (inlier_threshold_mv);
        icp_nl.setMaxCorrespondenceDistance (max_corresp_dist_mv_);
        icp_nl.setMaxIterations(mv_iterations);
        icp_nl.setInputNormals(normalclouds_aligned);
        icp_nl.setClouds (clouds_aligned);
        icp_nl.setVisIntermediate (false);
        icp_nl.setSparseSolver (true);
        icp_nl.setAdjacencyMatrix (A);
        icp_nl.setMinDot(min_dot);
        if(mv_use_weights_)
            icp_nl.setPointsWeight(weights);
        icp_nl.compute ();

        std::vector<Eigen::Matrix4f> transformations;
        icp_nl.getTransformation (transformations);

        {
            pcl::PointCloud<PointType>::Ptr accumulated_cloud (new pcl::PointCloud<PointType>);
            pcl::PointCloud<PointType>::Ptr accumulated_cloud_aligned (new pcl::PointCloud<PointType>);
            Eigen::Matrix4f accum = Eigen::Matrix4f::Identity();

            for(size_t i=0; i < clouds_aligned.size(); i++)
            {
                {
                    pcl::PointCloud<PointType>::Ptr aligned (new pcl::PointCloud<PointType>);
                    //pcl::transformPointCloud(*clouds_aligned[i], *aligned, poses[i]);
                    *accumulated_cloud += *clouds_aligned[i];
                }

                Eigen::Matrix4f final_trans = transformations[i];
                pcl::PointCloud<PointType>::Ptr aligned (new pcl::PointCloud<PointType>);
                pcl::transformPointCloud(*clouds_aligned[i], *aligned, final_trans);
                *accumulated_cloud_aligned += *aligned;
                poses[i] = final_trans * poses[i];
            }

            vis.removeAllPointClouds();

            {
                pcl::visualization::PointCloudColorHandlerRGBField<PointType> handler_rgb (accumulated_cloud_aligned);
                vis.addPointCloud<PointType> (accumulated_cloud_aligned, handler_rgb, "aligned_accum", v2);
            }

            {
                pcl::visualization::PointCloudColorHandlerRGBField<PointType> handler_rgb (accumulated_cloud);
                vis.addPointCloud<PointType> (accumulated_cloud, handler_rgb, "accum", v1);
            }

            if(vis_final_)
            {
                vis.spin();
            }
            else
            {
                vis.spinOnce(1000, true);
            }
        }
    }

    for(size_t k=0; k < range_images_.size(); k++)
    {
        //write original cloud
        {
            std::stringstream temp;
            temp << aligned_output_dir << "/cloud_";
            temp << setw( 8 ) << setfill( '0' ) << static_cast<int>(k) << ".pcd";
            std::string scene_name;
            temp >> scene_name;
            std::cout << scene_name << std::endl;
            pcl::io::savePCDFileBinary(scene_name, *range_images_[k]);
        }

        //write pose
        {
            std::stringstream temp;
            temp << aligned_output_dir << "/pose_";
            temp << setw( 8 ) << setfill( '0' ) << static_cast<int>(k) << ".txt";
            std::string scene_name;
            temp >> scene_name;
            std::cout << scene_name << std::endl;
            writeMatrixToFile(scene_name, poses[k]);
        }

        //write object indices
        {
            std::vector<int> obj_indices_original = registration_utils::maskToIndices(obj_masks_[k]);
            std::stringstream temp;
            temp << aligned_output_dir << "/object_indices_";
            temp << setw( 8 ) << setfill( '0' ) << static_cast<int>(k) << ".pcd";
            std::string scene_name;
            temp >> scene_name;
            std::cout << scene_name << std::endl;
            pcl::PointCloud<IndexPoint> obj_indices_cloud;
            obj_indices_cloud.width = obj_indices_original.size();
            obj_indices_cloud.height = 1;
            obj_indices_cloud.points.resize(obj_indices_cloud.width);
            for(size_t kk=0; kk < obj_indices_original.size(); kk++)
                obj_indices_cloud.points[kk].idx = obj_indices_original[kk];

            pcl::io::savePCDFileBinary(scene_name, obj_indices_cloud);
        }
    }

}
