
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

/*#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/icp.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <faat_pcl/registration/icp_with_gc.h>
#include <faat_pcl/3d_rec_framework/feature_wrapper/local/local_estimator.h>
#include <faat_pcl/utils/noise_models.h>*/

typedef pcl::PointXYZRGB PointType;

void highestPlane(std::vector<faat_pcl::PlaneModel<PointType> > & planes, int * highest)
{
    int table_plane_selected = 0;
    int max_inliers_found = -1;
    std::vector<int> plane_inliers_counts;
    plane_inliers_counts.resize (planes.size ());

    for (size_t i = 0; i < planes.size (); i++)
    {
        pcl::PointIndices inl = planes[i].inliers_;

        std::cout << "Number of inliers for this plane:" << inl.indices.size () << std::endl;

        plane_inliers_counts[i] = inl.indices.size ();

        if (plane_inliers_counts[i] > max_inliers_found)
        {
            table_plane_selected = i;
            max_inliers_found = plane_inliers_counts[i];
        }
    }

    size_t itt = static_cast<size_t> (table_plane_selected);

    pcl::ModelCoefficients mc = planes[itt].coefficients_;
    Eigen::Vector4f table_plane = Eigen::Vector4f (mc.values[0], mc.values[1],
                                                   mc.values[2], mc.values[3]);

    Eigen::Vector3f normal_table = Eigen::Vector3f (mc.values[0], mc.values[1], mc.values[2]);

    int inliers_count_best = plane_inliers_counts[itt];

    //check that the other planes with similar normal are not higher than the table_plane_selected
    for (size_t i = 0; i < planes.size (); i++)
    {

        pcl::ModelCoefficients mc = planes[i].coefficients_;

        Eigen::Vector4f model = Eigen::Vector4f (mc.values[0], mc.values[1],
                                                 mc.values[2], mc.values[3]);

        Eigen::Vector3f normal = Eigen::Vector3f (mc.values[0], mc.values[1], mc.values[2]);

        int inliers_count = plane_inliers_counts[i];

        std::cout << "Dot product is:" << normal.dot (normal_table) << std::endl;
        if ((normal.dot (normal_table) > 0.95) && (inliers_count_best * 0.5 <= inliers_count))
        {
            //check if this plane is higher, projecting a point on the normal direction
            std::cout << "Check if plane is higher, then change table plane" << std::endl;
            std::cout << model[3] << " " << table_plane[3] << std::endl;
            if (model[3] < table_plane[3])
            {
                PCL_WARN ("Changing table plane...");
                table_plane_selected = i;
                table_plane = model;
                normal_table = normal;
                inliers_count_best = inliers_count;
            }
        }
    }

    *highest = table_plane_selected;
}

struct PointXYZRedGreenBlue
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

    std::stringstream out_dir;
    out_dir << training_dir_out_ << "/" << model_name;
    bf::path out_dir_path = out_dir.str();
    if(!bf::exists(out_dir_path))
        bf::create_directory(out_dir_path);

    //Read the aligned clouds
    std::vector<Eigen::Matrix4f> pose_aligned_;
    {
        std::vector<std::string> files;
        std::string pattern_scenes = ".*cloud.*.pcd";

        bf::path input_dir = pcd_files_aligned_dir_;
        faat_pcl::utils::getFilesInDirectory(input_dir, files, pattern_scenes);

        std::cout << "Number of aligned scenes is:" << files.size () << std::endl;

        std::sort(files.begin(), files.end());

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
        }

    }

    //Read the original scans, split pcd_files_dir_ with comma
    {

        pcl::visualization::PCLVisualizer vis("test");
        vis.addCoordinateSystem(0.1);
        std::vector < std::string > strs_2;
        boost::split (strs_2, pcd_files_dir_, boost::is_any_of (","));
        int n_cloud=0;
        for(size_t jj=0; jj < strs_2.size(); jj++)
        {

            std::vector < std::string > strs_22;
            boost::split (strs_22, strs_2[jj], boost::is_any_of ("/\\"));
            std::string seq_num = strs_22[strs_22.size() - 1];

            std::vector<std::string> files;
            std::string pattern_scenes = ".*cloud.*.pcd";

            bf::path input_dir = strs_2[jj];
            faat_pcl::utils::getFilesInDirectory(input_dir, files, pattern_scenes);

            std::cout << "Number of scenes in directory is:" << files.size () << std::endl;

            std::sort(files.begin(), files.end());

            std::stringstream out_dir_seq;
            out_dir_seq << out_dir.str() << "/" << seq_num;
            bf::path out_dir_path = out_dir_seq.str();
            if(!bf::exists(out_dir_path))
                bf::create_directory(out_dir_path);

            //vis.removeAllPointClouds();

            for (size_t i = 0; i < files.size (); i+=step,n_cloud++)
            {
                pcl::PointCloud<PointType>::Ptr scene (new pcl::PointCloud<PointType>);
                pcl::PointCloud<PointType>::Ptr rimage (new pcl::PointCloud<PointType>);
                std::stringstream file_to_read;
                file_to_read << strs_2[jj] << "/" << files[i];
                pcl::io::loadPCDFile (file_to_read.str (), *scene);
                pcl::copyPointCloud(*scene, *rimage);

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

                /*pcl::NormalEstimationOMP<PointType, pcl::Normal> ne;
                ne.setInputCloud (scene);
                pcl::search::KdTree<PointType>::Ptr tree (new pcl::search::KdTree<PointType> ());
                ne.setSearchMethod (tree);
                ne.setRadiusSearch (0.02);
                pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
                ne.compute (*normals);

                faat_pcl::utils::noise_models::NguyenNoiseModel<PointType> nm;
                nm.setInputCloud(scene);
                nm.setInputNormals(normals);
                nm.setLateralSigma(0.003f);
                nm.setMaxAngle(65);
                nm.setUseDepthEdges(true);
                nm.compute();
                std::vector<float> weights;
                nm.getWeights(weights);

                std::vector<int> valid_indices;
                for(size_t k=0; k < weights.size(); k++)
                {
                    if(weights[k] > 0.9f)
                    {
                       valid_indices.push_back(k);
                    }
                }*/

                //pcl::PointCloud<PointType>::Ptr filtered(new pcl::PointCloud<PointType>);
                //pcl::copyPointCloud(*scene, valid_indices, *filtered);

                faat_pcl::MultiPlaneSegmentation<PointType> mps;
                mps.setInputCloud(scene);
                mps.setMinPlaneInliers(num_plane_inliers);
                mps.setResolution(0.001f);
                mps.setMergePlanes(true);
                mps.segment(true);
                std::vector<faat_pcl::PlaneModel<PointType> > planes_found;
                planes_found = mps.getModels();

                //select highest plane
                int highest = 0;
                highestPlane(planes_found, &highest);
                std::cout << "planes:" << planes_found.size() << std::endl;

                pcl::PointIndices above_plane;
                faat_pcl::PlaneModel<PointType> high_plane = planes_found[highest];

                pcl::ModelCoefficients mc = high_plane.coefficients_;
                Eigen::Vector4f table_plane = Eigen::Vector4f (mc.values[0], mc.values[1],
                                                               mc.values[2], mc.values[3]);

                for(size_t k=0; k < scene->points.size(); k++)
                {
                    Eigen::Vector3f xyz_p = scene->points[k].getVector3fMap ();

                    if (!pcl_isfinite (xyz_p[0]) || !pcl_isfinite (xyz_p[1]) || !pcl_isfinite (xyz_p[2]))
                        continue;

                    float val = xyz_p[0] * table_plane[0] + xyz_p[1] * table_plane[1] + xyz_p[2] * table_plane[2] + table_plane[3];

                    if (val > -0.02)
                    {
                        above_plane.indices.push_back(k);
                    }
                    else
                    {
                        scene->points[k].x = scene->points[k].y = scene->points[k].z = std::numeric_limits<float>::quiet_NaN();
                    }
                }

                pcl::IndicesPtr above_plane_indices_ptr;
                above_plane_indices_ptr.reset(new std::vector<int>(above_plane.indices));
                pcl::PointIndices cloud_object_indices;
                pcl::ExtractPolygonalPrismData<PointType> prism_;
                prism_.setHeightLimits(0.01,1.f);
                prism_.setIndices(above_plane_indices_ptr);
                prism_.setInputCloud (scene);
                prism_.setInputPlanarHull (high_plane.convex_hull_cloud_);
                prism_.segment (cloud_object_indices);

                pcl::PointCloud<PointType>::Ptr aligned (new pcl::PointCloud<PointType>);
                pcl::copyPointCloud(*scene, cloud_object_indices, *aligned);

                pcl::transformPointCloud(*aligned, *aligned, pose_aligned_[n_cloud]);
                std::stringstream name;
                name << "cloud_" << jj << "_" << i;
                vis.addPointCloud(aligned, name.str());
                vis.spinOnce(100, true);

                //save info

                std::stringstream temp;
                temp << out_dir_seq.str() << "/cloud_";
                temp << setw( 3 ) << setfill( '0' ) << static_cast<int>(i) << ".pcd";
                std::string scene_name;
                temp >> scene_name;
                std::cout << scene_name << std::endl;
                pcl::io::savePCDFileBinary (scene_name, *scene);

                //write pose
                {

                    std::stringstream temp;
                    temp << out_dir_seq.str() << "/pose_";
                    temp << setw( 3 ) << setfill( '0' ) << static_cast<int>(i) << ".txt";
                    std::string scene_name;
                    temp >> scene_name;
                    std::cout << scene_name << std::endl;
                    writeMatrixToFile(scene_name, pose_aligned_[n_cloud]);
                }

                {
                    std::stringstream temp;
                    temp << out_dir_seq.str() << "/object_indices_";
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
            }
        }
    }
    return 0;

    std::vector<std::string> files;
    std::string pattern_scenes = ".*cloud.*.pcd";

    bf::path input_dir = pcd_files_dir_;
    faat_pcl::utils::getFilesInDirectory(input_dir, files, pattern_scenes);

    std::cout << "Number of scenes in directory is:" << files.size () << std::endl;

    std::sort(files.begin(), files.end());

    std::vector<pcl::PointCloud<PointType>::Ptr> range_images_;
    std::vector<pcl::PointCloud<PointType>::Ptr> scenes_;
    std::vector<pcl::PointIndices> above_plane_indices_;
    std::vector<std::vector<bool> > object_masks_;

    pcl::visualization::PCLVisualizer vis("test");
    int v1,v2;
    vis.createViewPort(0,0,0.5,1,v1);
    vis.createViewPort(0.5,0,1,1,v2);
    vis.addCoordinateSystem(0.1);

    /*
      ROI
      */
    for (size_t i = 0; i < files.size (); i+=step)
    {
        pcl::PointCloud<PointType>::Ptr scene (new pcl::PointCloud<PointType>);
        pcl::PointCloud<PointType>::Ptr rimage (new pcl::PointCloud<PointType>);
        std::stringstream file_to_read;
        file_to_read << pcd_files_dir_ << "/" << files[i];
        pcl::io::loadPCDFile (file_to_read.str (), *scene);
        pcl::copyPointCloud(*scene, *rimage);

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

        scenes_.push_back(scene);

        faat_pcl::MultiPlaneSegmentation<PointType> mps;
        mps.setInputCloud(scene);
        mps.setMinPlaneInliers(num_plane_inliers);
        mps.setResolution(0.001f);
        mps.setMergePlanes(true);
        mps.segment(true);
        std::vector<faat_pcl::PlaneModel<PointType> > planes_found;
        planes_found = mps.getModels();

        //select highest plane
        int highest = 0;
        highestPlane(planes_found, &highest);
        std::cout << "planes:" << planes_found.size() << std::endl;

        pcl::PointIndices above_plane;
        faat_pcl::PlaneModel<PointType> high_plane = planes_found[highest];

        pcl::ModelCoefficients mc = high_plane.coefficients_;
        Eigen::Vector4f table_plane = Eigen::Vector4f (mc.values[0], mc.values[1],
                                                       mc.values[2], mc.values[3]);

        std::vector<bool> obj_mask(scene->points.size(), false);

        for(size_t k=0; k < scene->points.size(); k++)
        {
            Eigen::Vector3f xyz_p = scene->points[k].getVector3fMap ();

            if (!pcl_isfinite (xyz_p[0]) || !pcl_isfinite (xyz_p[1]) || !pcl_isfinite (xyz_p[2]))
                continue;

            float val = xyz_p[0] * table_plane[0] + xyz_p[1] * table_plane[1] + xyz_p[2] * table_plane[2] + table_plane[3];

            if (val > -0.02)
            {
                above_plane.indices.push_back(k);
            }

            if (val > 0.01)
            {
                obj_mask[k] = true;
            }
        }

        pcl::IndicesPtr above_plane_indices_ptr;
        above_plane_indices_ptr.reset(new std::vector<int>(above_plane.indices));
        pcl::PointIndices cloud_object_indices;
        pcl::ExtractPolygonalPrismData<PointType> prism_;
        prism_.setHeightLimits(-0.02,1.f);
        prism_.setIndices(above_plane_indices_ptr);
        prism_.setInputCloud (scene);
        prism_.setInputPlanarHull (high_plane.convex_hull_cloud_);
        prism_.segment (cloud_object_indices);

        std::vector<bool> mask;
        mask = registration_utils::indicesToMask(cloud_object_indices.indices, scene->points.size(), false);

        for(size_t kk=0; kk < mask.size(); kk++)
        {
            if(!mask[kk])
            {
                rimage->points[kk].x = rimage->points[kk].y = rimage->points[kk].z = std::numeric_limits<float>::quiet_NaN();
            }
        }

        range_images_.push_back(rimage);

        above_plane = cloud_object_indices;
        above_plane_indices_.push_back(above_plane);

        pcl::PointCloud<PointType>::Ptr above_plane_cloud (new pcl::PointCloud<PointType>);
        pcl::copyPointCloud(*scene, above_plane, *above_plane_cloud);

        vis.addPointCloud(rimage, "scene", v1);
        vis.addPointCloud(above_plane_cloud, "above_plane_cloud", v2);

        pcl::visualization::PointCloudColorHandlerCustom<PointType> handler_rgb (high_plane.plane_cloud_, 255, 0,0);
        vis.addPointCloud<PointType> (high_plane.plane_cloud_, handler_rgb, "plane", v2);

        vis.spinOnce(100,true);
        vis.removeAllPointClouds();

        //pcl::copyPointCloud(*scenes_[i], above_plane_indices_[i], *scenes_[i]);
        object_masks_.push_back(obj_mask);
    }

    /*
      FILTERING AND NORMALS COMPUTATION
      */

    /*std::vector< pcl::IndicesPtr > indices_views_;
    indices_views_.resize (scenes_.size ());

    typedef pcl::PointXYZRGBNormal PointTInternal;

    std::vector<pcl::PointCloud<PointTInternal>::Ptr> xyz_normals_;
    xyz_normals_.resize (scenes_.size ());

    for(size_t i=0; i < scenes_.size(); i++)
    {

        std::cout << "Not organized normals" << std::endl;
        pcl::NormalEstimationOMP<PointType, pcl::Normal> ne;
        ne.setInputCloud (range_images_[i]);
        pcl::search::KdTree<PointType>::Ptr tree (new pcl::search::KdTree<PointType> ());
        ne.setSearchMethod (tree);
        ne.setRadiusSearch (0.02);
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        ne.compute (*normals);

        faat_pcl::utils::noise_models::NguyenNoiseModel<PointType> nm;
        nm.setInputCloud(range_images_[i]);
        nm.setInputNormals(normals);
        nm.setLateralSigma(0.003f);
        nm.setMaxAngle(70);
        nm.setUseDepthEdges(true);
        nm.compute();
        std::vector<float> weights;
        nm.getWeights(weights);

        std::vector<int> valid_indices;
        for(size_t k=0; k < weights.size(); k++)
        {
            if(weights[k] > 0.8f)
            {
                if(object_masks_[i][k])
                    valid_indices.push_back(k);
            }
            else
            {
                range_images_[i]->points[k].x =
                range_images_[i]->points[k].y =
                range_images_[i]->points[k].z = std::numeric_limits<float>::quiet_NaN();
            }
        }

        pcl::PointCloud<PointType>::Ptr filtered(new pcl::PointCloud<PointType>);
        pcl::copyPointCloud(*scenes_[i], valid_indices, *filtered);
        scenes_[i] = filtered;

        xyz_normals_[i].reset(new pcl::PointCloud<PointTInternal>);
        pcl::copyPointCloud(*scenes_[i], *xyz_normals_[i]);
        pcl::copyPointCloud(*normals, valid_indices, *xyz_normals_[i]);

    }*/

    /*
        KEYPOINTS
      */

    /*std::vector<pcl::PointCloud<PointType>::Ptr> keypoints_;
    for(size_t i=0; i < scenes_.size(); i++)
    {

        boost::shared_ptr<faat_pcl::rec_3d_framework::UniformSamplingExtractor<PointType> > uniform_keypoint_extractor ( new faat_pcl::rec_3d_framework::UniformSamplingExtractor<PointType>);
        uniform_keypoint_extractor->setSamplingDensity (0.005f);
        uniform_keypoint_extractor->setFilterPlanar (true);
        uniform_keypoint_extractor->setThresholdPlanar(0.025);
        uniform_keypoint_extractor->setMaxDistance(2.5f);
        uniform_keypoint_extractor->setForceUnorganized(true);
        uniform_keypoint_extractor->setInputCloud(scenes_[i]);
        uniform_keypoint_extractor->setSupportRadius(0.04);
        std::vector<int> idxes;
        uniform_keypoint_extractor->compute(idxes);

        std::cout << "idxes size:" << idxes.size() << " " << scenes_[i]->points.size() << std::endl;

        indices_views_[i].reset(new std::vector<int>(idxes));

        pcl::PointCloud<PointType>::Ptr keypoints (new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*scenes_[i], idxes, *keypoints);

//        pcl::visualization::PCLVisualizer vis_icp("test");
//        int v1,v2;
//        vis_icp.createViewPort(0,0,0.5,1,v1);
//        vis_icp.createViewPort(0.5,0,1,1,v2);
//        vis_icp.addCoordinateSystem(0.1);

//        vis_icp.addPointCloud(keypoints, "keypoints", v1);
//        vis_icp.addPointCloud(scenes_[i], "scene", v2);
//        vis_icp.spin();

        keypoints_.push_back(keypoints);

    }*/

    /*
        pairwise
      */

    /*std::vector<Eigen::Matrix4f> poses_;
    poses_.push_back(Eigen::Matrix4f::Identity());

    {
        pcl::visualization::PCLVisualizer vis_icp("test");
        int v1,v2;
        vis_icp.createViewPort(0,0,0.5,1,v1);
        vis_icp.createViewPort(0.5,0,1,1,v2);
        vis_icp.addCoordinateSystem(0.1);

        for(size_t i=1; i < scenes_.size(); i++)
        {
            //pcl::PointCloud<PointType>::Ptr target (new pcl::PointCloud<PointType>());
            //pcl::PointCloud<PointType>::Ptr input (new pcl::PointCloud<PointType>());

            faat_pcl::IterativeClosestPointWithGC<PointTInternal, PointTInternal> icp;
            icp.setTransformationEpsilon (0.000001 * 0.000001);
            icp.setMinNumCorrespondences (5);
            icp.setMaxCorrespondenceDistance (0.1f);
            icp.setUseCG (true);
            icp.setSurvivalOfTheFittest (false);
            icp.setMaximumIterations(icp_iterations);
            icp.setOverlapPercentage(0.5f);
            icp.setVisFinal(vis_final_icp_);
            icp.setInliersThreshold(0.005f);
            icp.setDtVxSize(0.005f);
            icp.setSourceAndTargetIndices(indices_views_[i], indices_views_[i - 1]);
            icp.setuseColor(false);
            //icp.setSourceTargetCorrespondences(temp_correspondences);
            icp.setUseSHOT(false);

            icp.setRangeImages<PointType>(range_images_[i], range_images_[i-1], 525.f, 640, 480);
            pcl::registration::DefaultConvergenceCriteria<float>::Ptr convergence_criteria;
            convergence_criteria = icp.getConvergeCriteria ();
            convergence_criteria->setAbsoluteMSE (1e-12);
            convergence_criteria->setMaximumIterationsSimilarTransforms (icp_iterations);
            convergence_criteria->setFailureAfterMaximumIterations (false);

            icp.setInputTarget (xyz_normals_[i-1]);
            icp.setInputSource (xyz_normals_[i]);

            typename pcl::PointCloud<PointTInternal>::Ptr pp_out(new pcl::PointCloud<PointTInternal>);
            icp.align (*pp_out);
            std::vector<std::pair<float, Eigen::Matrix4f> > res;
            icp.getResults(res);

            pcl::PointCloud<PointType>::Ptr aligned (new pcl::PointCloud<PointType>);
            pcl::transformPointCloud(*range_images_[i], *aligned, res[0].second);

            vis_icp.addPointCloud(range_images_[i-1], "target");
            vis_icp.addPointCloud(range_images_[i], "input", v1);
            vis_icp.addPointCloud(aligned, "output", v2);
            vis_icp.spin();
            vis_icp.removeAllPointClouds();

            poses_.push_back(res[0].second);

        }
    }

    {
        pcl::PointCloud<PointType>::Ptr accumulated_cloud (new pcl::PointCloud<PointType>);
        pcl::PointCloud<PointType>::Ptr accumulated_cloud_aligned (new pcl::PointCloud<PointType>);

        Eigen::Matrix4f accum = Eigen::Matrix4f::Identity();

        for(size_t i=0; i < scenes_.size(); i++)
        {
            *accumulated_cloud += *scenes_[i];
            accum = accum * poses_[i];

            pcl::PointCloud<PointType>::Ptr aligned (new pcl::PointCloud<PointType>);
            pcl::transformPointCloud(*scenes_[i], *aligned, accum);

            *accumulated_cloud_aligned += *aligned;
        }

        pcl::visualization::PCLVisualizer vis_icp("test");
        int v1,v2;
        vis_icp.createViewPort(0,0,0.5,1,v1);
        vis_icp.createViewPort(0.5,0,1,1,v2);
        vis_icp.addCoordinateSystem(0.1);
        vis_icp.addPointCloud(accumulated_cloud, "input", v1);
        vis_icp.addPointCloud(accumulated_cloud_aligned, "output", v2);
        vis_icp.spin();
    }*/
}
