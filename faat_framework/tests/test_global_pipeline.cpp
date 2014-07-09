/*
 * OUR-CVFH test.
 *
 *  Created on: June 11, 2014
 *  Author: Aitor Aldoma

 Command line:
 ./bin/test_global_pipeline -pcd_file /home/aitor/data/willow_reduced/scenes/T_10/cloud_0000000000.pcd -model_dir /home/aitor/data/willow_reduced/models/ -max_z 1. -nn 3

 If you would like to do ICP to refine poses, add -icp_iterations X to the command line where (X > 0)

 */

#include <pcl/console/parse.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/organized_multi_plane_segmentation.h>
#include <pcl/segmentation/planar_polygon_fusion.h>
#include <pcl/segmentation/plane_coefficient_comparator.h>
#include <pcl/segmentation/euclidean_plane_coefficient_comparator.h>
#include <pcl/segmentation/rgb_plane_coefficient_comparator.h>
#include <pcl/segmentation/edge_aware_plane_comparator.h>
#include <pcl/segmentation/euclidean_cluster_comparator.h>
#include <pcl/segmentation/organized_connected_component_segmentation.h>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <pcl/common/common.h>
#include <boost/regex.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/our_cvfh.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>

struct IndexPoint
{
    int idx;
};

POINT_CLOUD_REGISTER_POINT_STRUCT (IndexPoint,
                                   (int, idx, idx)
                                   )

//do a segmentation that instead of the table plane, returns all indices that are not planes
template<typename PointT>
void
doSegmentation (typename pcl::PointCloud<PointT>::Ptr & xyz_points,
                pcl::PointCloud<pcl::Normal>::Ptr & normal_cloud,
                std::vector<pcl::PointIndices> & indices)
{
    Eigen::Vector4f table_plane;

    int min_cluster_size_ = 500;
    int num_plane_inliers = 1000;

    pcl::OrganizedMultiPlaneSegmentation<PointT, pcl::Normal, pcl::Label> mps;
    mps.setMinInliers (num_plane_inliers);
    mps.setAngularThreshold (0.017453 * 2.f); // 2 degrees
    mps.setDistanceThreshold (0.01); // 1cm
    mps.setInputNormals (normal_cloud);
    mps.setInputCloud (xyz_points);

    std::vector<pcl::PlanarRegion<PointT>, Eigen::aligned_allocator<pcl::PlanarRegion<PointT> > > regions;
    std::vector<pcl::ModelCoefficients> model_coefficients;
    std::vector<pcl::PointIndices> inlier_indices;
    pcl::PointCloud<pcl::Label>::Ptr labels (new pcl::PointCloud<pcl::Label>);
    std::vector<pcl::PointIndices> label_indices;
    std::vector<pcl::PointIndices> boundary_indices;
    std::vector<bool> plane_labels;

    typename pcl::PlaneRefinementComparator<PointT, pcl::Normal, pcl::Label>::Ptr ref_comp (
                new pcl::PlaneRefinementComparator<PointT,
                pcl::Normal, pcl::Label> ());
    ref_comp->setDistanceThreshold (0.01f, false);
    ref_comp->setAngularThreshold (0.017453 * 2);
    mps.setRefinementComparator (ref_comp);
    mps.segmentAndRefine (regions, model_coefficients, inlier_indices, labels, label_indices, boundary_indices);

    std::cout << "Number of planes found:" << model_coefficients.size () << std::endl;
    if(model_coefficients.size() == 0)
        return;

    int table_plane_selected = 0;
    int max_inliers_found = -1;
    std::vector<int> plane_inliers_counts;
    plane_inliers_counts.resize (model_coefficients.size ());

    for (size_t i = 0; i < model_coefficients.size (); i++)
    {
        Eigen::Vector4f table_plane = Eigen::Vector4f (model_coefficients[i].values[0], model_coefficients[i].values[1],
                                                       model_coefficients[i].values[2], model_coefficients[i].values[3]);

        std::cout << "Number of inliers for this plane:" << inlier_indices[i].indices.size () << std::endl;
        int remaining_points = 0;
        typename pcl::PointCloud<PointT>::Ptr plane_points (new pcl::PointCloud<PointT> (*xyz_points));
        for (int j = 0; j < plane_points->points.size (); j++)
        {
            Eigen::Vector3f xyz_p = plane_points->points[j].getVector3fMap ();

            if (!pcl_isfinite (xyz_p[0]) || !pcl_isfinite (xyz_p[1]) || !pcl_isfinite (xyz_p[2]))
                continue;

            float val = xyz_p[0] * table_plane[0] + xyz_p[1] * table_plane[1] + xyz_p[2] * table_plane[2] + table_plane[3];

            if (std::abs (val) > 0.01)
            {
                plane_points->points[j].x = std::numeric_limits<float>::quiet_NaN ();
                plane_points->points[j].y = std::numeric_limits<float>::quiet_NaN ();
                plane_points->points[j].z = std::numeric_limits<float>::quiet_NaN ();
            }
            else
                remaining_points++;
        }

        plane_inliers_counts[i] = remaining_points;

        if (remaining_points > max_inliers_found)
        {
            table_plane_selected = i;
            max_inliers_found = remaining_points;
        }
    }

    size_t itt = static_cast<size_t> (table_plane_selected);
    table_plane = Eigen::Vector4f (model_coefficients[itt].values[0], model_coefficients[itt].values[1],
                                   model_coefficients[itt].values[2], model_coefficients[itt].values[3]);

    Eigen::Vector3f normal_table = Eigen::Vector3f (model_coefficients[itt].values[0], model_coefficients[itt].values[1],
                                                    model_coefficients[itt].values[2]);

    int inliers_count_best = plane_inliers_counts[itt];

    //check that the other planes with similar normal are not higher than the table_plane_selected
    for (size_t i = 0; i < model_coefficients.size (); i++)
    {
        Eigen::Vector4f model = Eigen::Vector4f (model_coefficients[i].values[0], model_coefficients[i].values[1], model_coefficients[i].values[2],
                                                 model_coefficients[i].values[3]);

        Eigen::Vector3f normal = Eigen::Vector3f (model_coefficients[i].values[0], model_coefficients[i].values[1], model_coefficients[i].values[2]);

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

    table_plane = Eigen::Vector4f (model_coefficients[table_plane_selected].values[0], model_coefficients[table_plane_selected].values[1],
                                   model_coefficients[table_plane_selected].values[2], model_coefficients[table_plane_selected].values[3]);

    label_indices.resize(2);
    //create two labels, 1 one for points belonging to or under the plane, 1 for points above the plane
    for (int j = 0; j < xyz_points->points.size (); j++)
    {
        Eigen::Vector3f xyz_p = xyz_points->points[j].getVector3fMap ();

        if (!pcl_isfinite (xyz_p[0]) || !pcl_isfinite (xyz_p[1]) || !pcl_isfinite (xyz_p[2]))
            continue;

        float val = xyz_p[0] * table_plane[0] + xyz_p[1] * table_plane[1] + xyz_p[2] * table_plane[2] + table_plane[3];

        if (val >= 0.01f) //object
        {
            labels->points[j].label = 1;
            label_indices[1].indices.push_back(j);
        }
        else //plane or below
        {
            labels->points[j].label = 0;
            label_indices[0].indices.push_back(j);
        }
    }

    plane_labels.resize (2, false);
    plane_labels[0] = true;

    //cluster..
    typename pcl::EuclideanClusterComparator<PointT, pcl::Normal, pcl::Label>::Ptr
            euclidean_cluster_comparator_ (
                new pcl::EuclideanClusterComparator<
                PointT,
                pcl::Normal,
                pcl::Label> ());

    euclidean_cluster_comparator_->setInputCloud (xyz_points);
    euclidean_cluster_comparator_->setLabels (labels);
    euclidean_cluster_comparator_->setExcludeLabels (plane_labels);
    euclidean_cluster_comparator_->setDistanceThreshold (0.035f, true);

    pcl::PointCloud<pcl::Label> euclidean_labels;
    std::vector<pcl::PointIndices> euclidean_label_indices;
    pcl::OrganizedConnectedComponentSegmentation<PointT, pcl::Label> euclidean_segmentation (euclidean_cluster_comparator_);
    euclidean_segmentation.setInputCloud (xyz_points);
    euclidean_segmentation.segment (euclidean_labels, euclidean_label_indices);

    for (size_t i = 0; i < euclidean_label_indices.size (); i++)
    {
        if (euclidean_label_indices[i].indices.size () >= min_cluster_size_)
        {
            indices.push_back (euclidean_label_indices[i]);
        }
    }
}

namespace bf = boost::filesystem;

void
getDirectories (bf::path & path,
                std::vector<std::string> & directories)
{
    bf::directory_iterator end_itr;
    for (bf::directory_iterator itr (path); itr != end_itr; ++itr)
    {
        if ((bf::is_directory (*itr)))
        {
            std::string so_far = path.string () + "/" + (itr->path ().filename ()).string () + "/";
            directories.push_back(so_far);
        }
    }
}

void
getFilesInDirectory (const bf::path & path,
                     std::vector<std::string> & files,
                     const std::string & pattern)
{

    std::stringstream filter_str;
    filter_str << pattern;
    const boost::regex my_filter( filter_str.str() );

    bf::directory_iterator end_itr;
    for (bf::directory_iterator itr (path); itr != end_itr; ++itr)
    {
        if (!(bf::is_directory (*itr)))
        {
            std::string file = path.string () + "/" + (itr->path ().filename ()).string();

            boost::smatch what;
            if( !boost::regex_match( file, what, my_filter ) ) continue;

            files.push_back (file);
        }
    }
}

template<typename PointT>
class flann_model
{
public:
    typename pcl::PointCloud<PointT>::Ptr cloud_;
    typename pcl::PointCloud<PointT>::Ptr original_cloud_;
    Eigen::Matrix4f roll_transform_;
    Eigen::Vector3f centroid_;
    std::vector<float> descr_;
    std::string object_id_;
};

template<typename PointT>
class index_score
{
public:
    int idx_training_data_;
    int idx_input_;
    double score_;
    flann_model<PointT> flann_model_;
};

struct sortIndexScores
{
    template<typename PointT>
    bool
    operator() (const index_score<PointT>& d1, const index_score<PointT>& d2)
    {
        return d1.score_ < d2.score_;
    }
};

template<typename PointT>
inline void
convertToFLANN (const std::vector<flann_model<PointT> > &models, flann::Matrix<float> &data)
{
    data.rows = models.size ();
    data.cols = models[0].descr_.size (); // number of histogram bins

    flann::Matrix<float> flann_data (new float[models.size () * models[0].descr_.size ()], models.size (), models[0].descr_.size ());

    for (size_t i = 0; i < data.rows; ++i)
        for (size_t j = 0; j < data.cols; ++j)
        {
            flann_data.ptr ()[i * data.cols + j] = models[i].descr_[j];
        }

    data = flann_data;
}

template<typename PointT>
void
nearestKSearch (flann::Index<flann::L2<float> > * index, const flann_model<PointT> &model,
                int k, flann::Matrix<int> &indices, flann::Matrix<float> &distances)
{
    flann::Matrix<float> p = flann::Matrix<float> (new float[model.descr_.size ()], 1, model.descr_.size ());
    memcpy (&p.ptr ()[0], &model.descr_[0], p.cols * p.rows * sizeof(float));

    indices = flann::Matrix<int> (new int[k], 1, k);
    distances = flann::Matrix<float> (new float[k], 1, k);
    index->knnSearch (p, indices, distances, k, flann::SearchParams (512));
    delete[] p.ptr ();
}

template<typename PointT>
inline void
computeOURCVFH(typename pcl::PointCloud<PointT>::Ptr & view_cropped,
               pcl::PointCloud<pcl::VFHSignature308> & signatures,
               std::vector<Eigen::Vector3f> & centroids_in,
               std::vector<bool> & valid_roll_transforms_in,
               std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > & transforms_in)
{
    float voxel_grid_size = 0.003f;
    typename pcl::PointCloud<PointT>::Ptr processed(new pcl::PointCloud<PointT>);
    pcl::VoxelGrid<PointT> grid;
    grid.setInputCloud (view_cropped);
    grid.setLeafSize (voxel_grid_size, voxel_grid_size, voxel_grid_size);
    grid.setDownsampleAllData (true);
    grid.filter (*processed);

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimationOMP<PointT, pcl::Normal> ne;
    ne.setRadiusSearch(0.02f);
    ne.setInputCloud (processed);
    ne.compute (*normals);

    typedef typename pcl::OURCVFHEstimation<PointT, pcl::Normal, pcl::VFHSignature308> OURCVFHEstimation;
    typename pcl::search::KdTree<PointT>::Ptr cvfh_tree (new pcl::search::KdTree<PointT>);

    OURCVFHEstimation our_cvfh;
    our_cvfh.setSearchMethod (cvfh_tree);
    our_cvfh.setInputCloud (processed);
    our_cvfh.setInputNormals (normals);
    our_cvfh.setNormalizeBins (false);
    our_cvfh.setClusterTolerance (0.01);
    our_cvfh.setRadiusNormals (0.02f);
    our_cvfh.setMinPoints (50);

    our_cvfh.compute (signatures);

    our_cvfh.getCentroidClusters (centroids_in);
    our_cvfh.getTransforms (transforms_in);
    our_cvfh.getValidTransformsVec (valid_roll_transforms_in);
}

int
main (int argc, char ** argv)
{

    std::string pcd_file = "";
    std::string model_dir = "";

    float axis_threshold = 1.f;
    float max_z = 1.f;
    bool visualize_candidates_ = false;
    bool visualize_individual_poses_ = false;
    int NN_ = 5;
    int icp_iterations = 0;

    pcl::console::parse_argument (argc, argv, "-icp_iterations", icp_iterations);
    pcl::console::parse_argument (argc, argv, "-pcd_file", pcd_file);
    pcl::console::parse_argument (argc, argv, "-axis_threshold", axis_threshold);
    pcl::console::parse_argument (argc, argv, "-model_dir", model_dir);
    pcl::console::parse_argument (argc, argv, "-max_z", max_z);
    pcl::console::parse_argument (argc, argv, "-visualize_candidates", visualize_candidates_);
    pcl::console::parse_argument (argc, argv, "-vis_poses", visualize_individual_poses_);
    pcl::console::parse_argument (argc, argv, "-nn", NN_);

    typedef pcl::PointXYZRGB PointT;
    typedef pcl::Histogram<1327> FeatureT;

    if (pcd_file.compare ("") == 0)
    {
        PCL_ERROR("Set the filename of the PCD file of the scene\n");
        return -1;
    }

    if (model_dir.compare ("") == 0)
    {
        PCL_ERROR("model_dir option is empty. Required to train objects.\n");
        return -1;
    }

    //load models and setup FLANN structure
    std::string pattern_scenes = ".*cloud_.*.pcd";
    std::string pattern_indices = ".*object_indices_.*.pcd";

    bf::path models_path = model_dir;
    std::vector<std::string> directories;
    getDirectories(models_path, directories);

    std::vector<flann_model<PointT> > training_data_flann_models;

    for(size_t i=0; i < directories.size(); i++)
    {
        std::cout << directories[i] << std::endl;

        bf::path path = directories[i];
        std::vector<std::string> pcd_files, indices_files;
        getFilesInDirectory(path, indices_files, pattern_indices);
        getFilesInDirectory(path, pcd_files, pattern_scenes);

        std::cout << pcd_files.size() << " " << indices_files.size() << std::endl;
        std::sort(pcd_files.begin(), pcd_files.end());
        std::sort(indices_files.begin(), indices_files.end());

        //load files and compute OUR-CVFH

        for(size_t k=0; k < indices_files.size(); k++)
        {
            pcl::PointCloud<PointT>::Ptr view(new pcl::PointCloud<PointT>);
            pcl::io::loadPCDFile(pcd_files[k], *view);

            pcl::PointCloud<IndexPoint> obj_indices_cloud;
            pcl::io::loadPCDFile (indices_files[k], obj_indices_cloud);
            pcl::PointIndices indices;
            indices.indices.resize(obj_indices_cloud.points.size());
            for(size_t kk=0; kk < obj_indices_cloud.points.size(); kk++)
                indices.indices[kk] = obj_indices_cloud.points[kk].idx;

            pcl::PointCloud<PointT>::Ptr view_cropped(new pcl::PointCloud<PointT>);
            pcl::copyPointCloud(*view, indices, *view_cropped);

            pcl::PointCloud<pcl::VFHSignature308> cvfh_signatures;
            std::vector<Eigen::Vector3f> centroids_in;
            std::vector<bool> valid_roll_transforms_in;
            std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transforms_in;

            computeOURCVFH<PointT>(view_cropped, cvfh_signatures, centroids_in, valid_roll_transforms_in, transforms_in);

            std::cout << cvfh_signatures.points.size() << std::endl;

            for(size_t d=0; d < cvfh_signatures.points.size(); d++)
            {
                if(!valid_roll_transforms_in[d])
                    continue;

                flann_model<PointT> descr_model;
                descr_model.object_id_ = directories[i];
                descr_model.cloud_ = view_cropped;
                descr_model.original_cloud_ = view;
                descr_model.roll_transform_ = transforms_in[d];
                descr_model.centroid_ = centroids_in[d];

                int size_feat = sizeof(cvfh_signatures.points[d].histogram) / sizeof(float);
                descr_model.descr_.resize (size_feat);
                memcpy (&descr_model.descr_[0], &cvfh_signatures.points[d].histogram[0], size_feat * sizeof(float));
                training_data_flann_models.push_back(descr_model);
            }
        }
    }

    //transform to FLANN structure
    typedef flann::L2<float> DistT;
    flann::Matrix<float> flann_data_;
    flann::Index<DistT> * flann_index_;

    convertToFLANN (training_data_flann_models, flann_data_);
    flann_index_ = new flann::Index<DistT> (flann_data_, flann::LinearIndexParams ());
    flann_index_->buildIndex ();

    pcl::PointCloud<PointT>::Ptr view(new pcl::PointCloud<PointT>);
    pcl::io::loadPCDFile(pcd_file, *view);

    pcl::PointCloud<PointT>::Ptr view_orig(new pcl::PointCloud<PointT>(*view));

    pcl::PassThrough<PointT> pass_;
    pass_.setFilterLimits (0.f, max_z);
    pass_.setFilterFieldName ("z");
    pass_.setInputCloud (view);
    pass_.setKeepOrganized (true);
    pass_.filter (*view);

    //compute scene normals
    pcl::PointCloud<pcl::Normal>::Ptr normal_cloud (new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimationOMP<PointT, pcl::Normal> ne;
    ne.setRadiusSearch(0.02f);
    ne.setInputCloud (view);
    ne.compute (*normal_cloud);

    //segment the objects on top of the highest dominant plane
    std::vector<pcl::PointIndices> indices;
    doSegmentation<PointT>(view, normal_cloud, indices);

    typedef index_score<PointT> IdxScore;
    typedef std::vector<IdxScore> IndexScoreVector;
    std::vector< IndexScoreVector > candidates_for_clusters;
    candidates_for_clusters.resize(indices.size());

    //compute OUR-CVFH for the extracted objects and match against training data
    for (size_t c = 0; c < indices.size (); c++)
    {
        typename pcl::PointCloud<PointT>::Ptr cluster (new pcl::PointCloud<PointT>);
        typename pcl::PointCloud<PointT>::Ptr cluster_organized (new pcl::PointCloud<PointT>(*view));

        pcl::copyPointCloud (*view, indices[c].indices, *cluster);

        std::vector<bool> set_to_nan(view->points.size(), true);
        for(size_t kk=0; kk < indices[c].indices.size(); kk++)
            set_to_nan[indices[c].indices[kk]] = false;

        for(size_t kk=0; kk < cluster_organized->points.size(); kk++)
        {
            if(set_to_nan[kk])
            {
                cluster_organized->points[kk].x =
                cluster_organized->points[kk].y =
                cluster_organized->points[kk].z = std::numeric_limits<float>::quiet_NaN();
            }
        }

        pcl::PointCloud<pcl::VFHSignature308> ourcvfh_signatures;
        std::vector<Eigen::Vector3f> centroids_in;
        std::vector<bool> valid_roll_transforms_in;
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transforms_in;

        computeOURCVFH<PointT>(cluster, ourcvfh_signatures, centroids_in, valid_roll_transforms_in, transforms_in);
        flann::Matrix<int> indices;
        flann::Matrix<float> distances;

        std::vector<index_score<PointT> > indices_scores;

        for(size_t d=0; d < ourcvfh_signatures.points.size(); d++)
        {
            if(!valid_roll_transforms_in[d])
                continue;

            float* hist = ourcvfh_signatures.points[d].histogram;
            int size_feat = sizeof(ourcvfh_signatures.points[d].histogram) / sizeof(float);
            std::vector<float> std_hist (hist, hist + size_feat);

            flann_model<PointT> histogram;
            histogram.descr_ = std_hist;
            histogram.centroid_ = centroids_in[d];
            histogram.cloud_ = cluster;
            histogram.roll_transform_ = transforms_in[d];

            nearestKSearch<PointT> (flann_index_, histogram, NN_, indices, distances);

            for(int k=0; k < NN_; k++)
            {
                index_score<PointT> is;
                is.idx_training_data_ = indices[0][k];
                is.idx_input_ = static_cast<int>(d);
                is.score_ =  distances[0][k];
                is.flann_model_ = histogram; //save the data for the input cluster
                indices_scores.push_back (is);
            }
        }

        std::sort (indices_scores.begin (), indices_scores.end (), sortIndexScores());
        indices_scores.resize(NN_); //resize to desired NN
        candidates_for_clusters[c] = indices_scores;

        //visualize candidates for this cluster
        if(visualize_candidates_)
        {

            std::stringstream name_input_cluster;
            name_input_cluster << "cluster_clouds/cluster_" << c << ".pcd";

            pcl::io::savePCDFileBinary(name_input_cluster.str(), *cluster_organized);

            int k = static_cast<int>(indices_scores.size() + 1);
            pcl::visualization::PCLVisualizer p (argc, argv, "OUR-CVFH candidates");
            int y_s = (int)floor (sqrt ((double)k));
            int x_s = y_s + (int)ceil ((k / (double)y_s) - y_s);
            double x_step = (double)(1 / (double)x_s);
            double y_step = (double)(1 / (double)y_s);

            int viewport = 0, l = 0, m = 0;

            p.createViewPort (l * x_step, m * y_step, (l + 1) * x_step, (m + 1) * y_step, viewport);

            pcl::visualization::PointCloudColorHandlerGenericField<PointT> handler_rgb (view_orig, "z");
            p.addPointCloud<PointT> (view_orig, handler_rgb, "orig_v3", viewport);
            p.addPointCloud (cluster, "input_cluster_cloud", viewport);

            /*p.addText ("input cluster", 20, 30, 0, 1, 0, "input cluster", viewport);
            p.setShapeRenderingProperties (pcl::visualization::PCL_VISUALIZER_FONT_SIZE, 14, "input cluster", viewport);*/

            l++;

            for (int i = 0; i < (k - 1); ++i)
            {
                p.createViewPort (l * x_step, m * y_step, (l + 1) * x_step, (m + 1) * y_step, viewport);
                l++;
                if (l >= x_s)
                {
                    l = 0; m++;
                }

                std::stringstream cloud_name;
                cloud_name << "candidate_" << i;

                pcl::visualization::PointCloudColorHandlerGenericField<PointT> handler_rgb (training_data_flann_models[indices_scores[i].idx_training_data_].original_cloud_, "z");
                p.addPointCloud<PointT> (training_data_flann_models[indices_scores[i].idx_training_data_].original_cloud_, handler_rgb, cloud_name.str(), viewport);

                cloud_name << "_cluster";

                p.addPointCloud (training_data_flann_models[indices_scores[i].idx_training_data_].cloud_, cloud_name.str(), viewport);

                std::stringstream name_input_cluster;
                name_input_cluster << "cluster_clouds/candidate_" << c << "_" << i << ".pcd";
                pcl::io::savePCDFileBinary(name_input_cluster.str(), *training_data_flann_models[indices_scores[i].idx_training_data_].cloud_);
                /*std::stringstream ss;
                ss << indices_scores[i].score_;
                p.addText (ss.str (), 20, 30, 0, 1, 0, ss.str (), viewport);

                p.setShapeRenderingProperties (pcl::visualization::PCL_VISUALIZER_FONT_SIZE, 14, ss.str (), viewport);*/
            }

            p.addCoordinateSystem (0.1);
            p.setBackgroundColor(1,1,1);
            p.spin ();
        }
    }

    //estimate poses for all candidates
    std::vector<pcl::PointCloud<PointT>::Ptr> candidates;
    for(size_t c=0; c < candidates_for_clusters.size(); c++)
    {
        if(candidates_for_clusters[c].size() == 0)
            continue;

        for(size_t k=0; k < candidates_for_clusters[c].size(); k++)
        {
            index_score<PointT> is = candidates_for_clusters[c][k];
            flann_model<PointT> training = training_data_flann_models[is.idx_training_data_];
            flann_model<PointT> input_cluster = is.flann_model_;

            //estimate pose of training candidate
            Eigen::Matrix4f hom_from_OC_to_CC;
            hom_from_OC_to_CC = input_cluster.roll_transform_.inverse() *  training.roll_transform_;

            pcl::PointCloud<PointT>::Ptr candidate_aligned(new pcl::PointCloud<PointT>);
            pcl::transformPointCloud(*training.cloud_, *candidate_aligned, hom_from_OC_to_CC);

            candidates.push_back(candidate_aligned);
        }
    }

    if(visualize_individual_poses_)
    {
        for(size_t c=0; c < candidates_for_clusters.size(); c++)
        {
            if(candidates_for_clusters[c].size() == 0)
                continue;

            pcl::visualization::PCLVisualizer vis("poses");
            int v1,v2,v3;
            vis.createViewPort(0,0,0.33,1,v1);
            vis.createViewPort(0.33,0,0.66,1,v2);
            vis.createViewPort(0.66,0,1,1,v3);
            vis.addCoordinateSystem(0.2f);

            std::cout << candidates_for_clusters[c].size() << std::endl;
            for(size_t k=0; k < candidates_for_clusters[c].size(); k++)
            {
                index_score<PointT> is = candidates_for_clusters[c][k];
                flann_model<PointT> training = training_data_flann_models[is.idx_training_data_];
                flann_model<PointT> input_cluster = is.flann_model_;

                //estimate pose of training candidate
                Eigen::Matrix4f hom_from_OC_to_CC;
                hom_from_OC_to_CC = input_cluster.roll_transform_.inverse() *  training.roll_transform_;

                pcl::PointCloud<PointT>::Ptr candidate_aligned(new pcl::PointCloud<PointT>);
                pcl::transformPointCloud(*training.cloud_, *candidate_aligned, hom_from_OC_to_CC);

                vis.addPointCloud(training.cloud_, "training", v1);
                vis.addPointCloud(input_cluster.cloud_, "input", v2);
                vis.addPointCloud(candidate_aligned, "aligned", v3);
                vis.spin();
                vis.removeAllPointClouds();
            }
        }
    }

    //ICP
    if(icp_iterations > 0)
    {
        typename pcl::search::KdTree<PointT>::Ptr kdtree_scene(new pcl::search::KdTree<PointT>);
        kdtree_scene->setInputCloud(view_orig);
        float max_dist = 0.01f;

        for (size_t i = 0; i < candidates.size(); i++)
        {
            typename pcl::registration::CorrespondenceRejectorSampleConsensus<PointT>::Ptr
                    rej (new pcl::registration::CorrespondenceRejectorSampleConsensus<PointT> ());

            rej->setInputTarget (view_orig);
            rej->setMaximumIterations (1000);
            rej->setInlierThreshold (0.005f);
            rej->setInputSource (candidates[i]);

            pcl::IterativeClosestPoint<PointT, PointT> reg;
            reg.addCorrespondenceRejector (rej);
            reg.setInputTarget (view_orig); //scene
            reg.setInputSource (candidates[i]); //model
            reg.setMaximumIterations (icp_iterations);
            reg.setMaxCorrespondenceDistance (max_dist);
            reg.setSearchMethodTarget(kdtree_scene, true); //avoid building the kd-tree each time
            reg.setEuclideanFitnessEpsilon(1e-9);

            typename pcl::PointCloud<PointT>::Ptr output_ (new pcl::PointCloud<PointT> ());
            reg.align (*output_);

            candidates[i] = output_;
        }
    }

    //visualization stuff
    pcl::visualization::PCLVisualizer vis("TEST");
    int v1,v2,v3;
    vis.createViewPort(0,0,0.33,1,v1);
    vis.createViewPort(0.33,0,0.66,1,v2);
    vis.createViewPort(0.66,0,1,1,v3);

    vis.addPointCloud(view_orig, "view", v1);
    pcl::visualization::PointCloudColorHandlerCustom<PointT> handler_rgb (view_orig, 125, 125, 125);
    vis.addPointCloud<PointT> (view_orig, handler_rgb, "orig_v3", v3);

    vis.addText ("scene", 20, 30, 0, 1, 0, "scene", v1);
    vis.addText ("segmentation", 20, 30, 0, 1, 0, "segmentation", v2);
    vis.addText ("candidates aligned", 20, 30, 0, 1, 0, "candidates", v3);
    vis.setShapeRenderingProperties (pcl::visualization::PCL_VISUALIZER_FONT_SIZE, 14, "scene");
    vis.setShapeRenderingProperties (pcl::visualization::PCL_VISUALIZER_FONT_SIZE, 14, "segmentation");
    vis.setShapeRenderingProperties (pcl::visualization::PCL_VISUALIZER_FONT_SIZE, 14, "candidates");

    for (size_t c = 0; c < indices.size (); c++)
    {
        std::stringstream name;
        name << "cluster_" << c;
        typename pcl::PointCloud<PointT>::Ptr cluster (new pcl::PointCloud<PointT>);
        pcl::copyPointCloud (*view, indices[c].indices, *cluster);
        pcl::visualization::PointCloudColorHandlerRandom<PointT> handler_rgb (cluster);
        vis.addPointCloud<PointT> (cluster, handler_rgb, name.str (), v2);
    }

    for(size_t c=0; c < candidates.size(); c++)
    {
        std::stringstream name;
        name << "candidate_" << c;
        vis.addPointCloud(candidates[c], name.str(), v3);
    }

    vis.setBackgroundColor(1,1,1);
    vis.spin();
}
