/*
 * local_recognition_mian_dataset.cpp
 *
 *  Created on: Mar 24, 2012
 *      Author: aitor
 */
#include <pcl/console/parse.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/copy.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_matrix.hpp>
#include <boost/graph/undirected_graph.hpp>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/euclidean_cluster_comparator.h>
#include <pcl/segmentation/organized_connected_component_segmentation.h>
#include <v4r/PCLAddOns/NormalsEstimationNR.hh>
#include <pcl/filters/fast_bilateral.h>
#include "v4r/SurfaceSegmenter/segmentation.hpp"

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT>::Ptr PointCloudPtr;
typedef pcl::PointCloud<PointT> PointCloud;

float voxel_resolution = 0.005f;
float seed_resolution = 0.05f;
float dot_threshold_ = 0.99f;
float curv_threshold_ = 0.3f;
float cluster_tolerance_ = 0.01f;

class SmoothClusters
{
public:
    pcl::PointIndices indices_;
    Eigen::Vector3f avg_normal_;
    Eigen::Vector4f plane_;
};

bool
sortClustersBySize (const SmoothClusters & i, const SmoothClusters & j)
{
  return (i.indices_.indices.size() > j.indices_.indices.size());
}

void SupervoxelsSegmentation(PointCloudPtr & input_cloud,
                             pcl::PointCloud<pcl::Normal>::Ptr & normal_cloud,
                             std::vector<SmoothClusters> & smooth_clusters)
{
    typename pcl::SupervoxelClustering<PointT> super (voxel_resolution, seed_resolution, false);
    super.setInputCloud (input_cloud);
    super.setColorImportance (1.f);
    super.setSpatialImportance (1.f);
    super.setNormalImportance (5.f);
    super.setNormalCloud(normal_cloud);
    std::map <uint32_t, typename pcl::Supervoxel<PointT>::Ptr > supervoxel_clusters;
    pcl::console::print_highlight ("Extracting supervoxels!\n");
    super.extract (supervoxel_clusters);
    //super.refineSupervoxels(5, supervoxel_clusters);
    pcl::console::print_info ("Found %d supervoxels\n", supervoxel_clusters.size ());

    pcl::PointCloud<pcl::PointXYZL>::Ptr supervoxels_labels_cloud = super.getLabeledCloud();
    uint32_t max_label = super.getMaxLabel();

    pcl::PointCloud<pcl::PointNormal>::Ptr sv_normal_cloud = super.makeSupervoxelNormalCloud (supervoxel_clusters);

    std::vector<int> label_to_idx;
    label_to_idx.resize(max_label + 1, -1);
    typename std::map <uint32_t, typename pcl::Supervoxel<PointT>::Ptr>::iterator sv_itr,sv_itr_end;
    sv_itr = supervoxel_clusters.begin ();
    sv_itr_end = supervoxel_clusters.end ();
    int i=0;
    for ( ; sv_itr != sv_itr_end; ++sv_itr, i++)
    {
        label_to_idx[sv_itr->first] = i;
    }

    std::vector< std::vector<bool> > adjacent;
    adjacent.resize(supervoxel_clusters.size());
    for(size_t i=0; i < (supervoxel_clusters.size()); i++)
        adjacent[i].resize(supervoxel_clusters.size(), false);

    std::multimap<uint32_t, uint32_t> supervoxel_adjacency;
    super.getSupervoxelAdjacency (supervoxel_adjacency);
    //To make a graph of the supervoxel adjacency, we need to iterate through the supervoxel adjacency multimap
    std::multimap<uint32_t,uint32_t>::iterator label_itr = supervoxel_adjacency.begin ();
    std::cout << "super voxel adjacency size:" << supervoxel_adjacency.size() << std::endl;
    for ( ; label_itr != supervoxel_adjacency.end (); )
    {
        //First get the label
        uint32_t supervoxel_label = label_itr->first;
        Eigen::Vector3f normal_super_voxel = sv_normal_cloud->points[label_to_idx[supervoxel_label]].getNormalVector3fMap();
        normal_super_voxel.normalize();
        //Now we need to iterate through the adjacent supervoxels and make a point cloud of them
        std::multimap<uint32_t,uint32_t>::iterator adjacent_itr = supervoxel_adjacency.equal_range (supervoxel_label).first;
        for ( ; adjacent_itr!=supervoxel_adjacency.equal_range (supervoxel_label).second; ++adjacent_itr)
        {
            Eigen::Vector3f normal_neighbor_supervoxel = sv_normal_cloud->points[label_to_idx[adjacent_itr->second]].getNormalVector3fMap();
            normal_neighbor_supervoxel.normalize();

            if(normal_super_voxel.dot(normal_neighbor_supervoxel) > dot_threshold_)
            {
                adjacent[label_to_idx[supervoxel_label]][label_to_idx[adjacent_itr->second]] = true;
            }
        }

        //Move iterator forward to next label
        label_itr = supervoxel_adjacency.upper_bound (supervoxel_label);
    }

    typedef boost::adjacency_matrix<boost::undirectedS, int> Graph;
    Graph G(supervoxel_clusters.size());
    for(size_t i=0; i < supervoxel_clusters.size(); i++)
    {
        for(size_t j=(i+1); j < supervoxel_clusters.size(); j++)
        {
            if(adjacent[i][j])
                boost::add_edge(i, j, G);
        }
    }

    std::vector<int> components (boost::num_vertices (G));
    int n_cc = static_cast<int> (boost::connected_components (G, &components[0]));
    std::cout << "Number of connected components..." << n_cc << std::endl;

    std::vector<int> cc_sizes;
    std::vector<std::vector<int> > ccs;
    std::vector<uint32_t> original_labels_to_merged;
    original_labels_to_merged.resize(supervoxel_clusters.size());

    ccs.resize(n_cc);
    cc_sizes.resize (n_cc, 0);
    typename boost::graph_traits<Graph>::vertex_iterator vertexIt, vertexEnd;
    boost::tie (vertexIt, vertexEnd) = vertices (G);
    for (; vertexIt != vertexEnd; ++vertexIt)
    {
        int c = components[*vertexIt];
        cc_sizes[c]++;
        ccs[c].push_back(*vertexIt);
        original_labels_to_merged[*vertexIt] = c;
    }

    for(size_t i=0; i < supervoxels_labels_cloud->points.size(); i++)
    {
        //std::cout << supervoxels_labels_cloud->points[i].label << " " << label_to_idx.size() << " " << original_labels_to_merged.size() << " " << label_to_idx[supervoxels_labels_cloud->points[i].label] << std::endl;
        if(label_to_idx[supervoxels_labels_cloud->points[i].label] < 0)
            continue;

        supervoxels_labels_cloud->points[i].label = original_labels_to_merged[label_to_idx[supervoxels_labels_cloud->points[i].label]];
    }

    //create clusters
    smooth_clusters.resize(ccs.size());

    for(size_t i=0; i < supervoxels_labels_cloud->points.size(); i++)
    {
        if(supervoxels_labels_cloud->points[i].label <= 0)
            continue;

        smooth_clusters[supervoxels_labels_cloud->points[i].label].indices_.indices.push_back(i);
    }

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr supervoxels_rgb(new pcl::PointCloud<pcl::PointXYZRGBA>);
    supervoxels_rgb = super.getColoredCloud();

    {
        pcl::visualization::PCLVisualizer vis("model smooth surfaces");
        int v1,v2, v3;
        vis.createViewPort(0,0,0.33,1.0,v1);
        vis.createViewPort(0.33, 0, 0.66, 1, v2);
        vis.createViewPort(0.66, 0, 1, 1, v3);
        vis.addPointCloud<PointT>(input_cloud, "model", v1);
        vis.addPointCloudNormals<pcl::PointNormal> (sv_normal_cloud,1,0.05f, "supervoxel_normals", v2);
        vis.addPointCloud(supervoxels_rgb, "labels", v2);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr smooth_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);
        std::vector<uint32_t> label_colors;
        label_colors.reserve (max_label);
        srand (static_cast<unsigned int> (time (0)));
        while (label_colors.size () <= max_label )
        {
            uint8_t r = static_cast<uint8_t>( (rand () % 256));
            uint8_t g = static_cast<uint8_t>( (rand () % 256));
            uint8_t b = static_cast<uint8_t>( (rand () % 256));
            label_colors.push_back (static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
        }

        smooth_rgb->points.resize (supervoxels_labels_cloud->points.size ());
        smooth_rgb->width = supervoxels_labels_cloud->points.size();
        smooth_rgb->height = 1;

        {
            for(size_t i=0; i < supervoxels_labels_cloud->points.size(); i++)
            {
                Eigen::Vector3f p = supervoxels_labels_cloud->points[i].getVector3fMap();

                if(!pcl_isnan(p[0]))
                {
                    smooth_rgb->points[i].getVector3fMap() = supervoxels_labels_cloud->points[i].getVector3fMap();
                    smooth_rgb->points[i].rgb = label_colors[supervoxels_labels_cloud->points[i].label];
                }
            }
        }

        vis.addPointCloud(smooth_rgb, "labels_smooth", v3);
        vis.spin();
    }
}

void SmoothRegionsSegmentation(PointCloudPtr & surface_, pcl::PointCloud<pcl::Normal>::Ptr & normal_cloud)
{

    std::vector<pcl::PointIndices> clusters;

    pcl::ScopeTime t("clustering");
    typename pcl::EuclideanClusterComparator<PointT, pcl::Normal, pcl::Label>::Ptr
            euclidean_cluster_comparator (new pcl::EuclideanClusterComparator<PointT, pcl::Normal, pcl::Label> ());

    //create two labels, 1 one for points to be smoothly clustered, another one for the rest
    pcl::PointCloud<pcl::Label>::Ptr labels (new pcl::PointCloud<pcl::Label>);
    labels->points.resize(surface_->points.size());
    labels->width = surface_->width;
    labels->height = surface_->height;
    labels->is_dense = surface_->is_dense;

    for (size_t j = 0; j < surface_->points.size (); j++)
    {
        Eigen::Vector3f xyz_p = surface_->points[j].getVector3fMap ();
        if (!pcl_isfinite (xyz_p[0]) || !pcl_isfinite (xyz_p[1]) || !pcl_isfinite (xyz_p[2]))
        {
            labels->points[j].label = 0;
            continue;
        }

        //check normal
        Eigen::Vector3f normal = normal_cloud->points[j].getNormalVector3fMap ();
        if (!pcl_isfinite (normal[0]) || !pcl_isfinite (normal[1]) || !pcl_isfinite (normal[2]))
        {
            labels->points[j].label = 0;
            continue;
        }

        //check curvature
        float curvature = normal_cloud->points[j].curvature;
        if(curvature > (curv_threshold_)) // * (std::min(1.f,scene_cloud_->points[j].z))))
        {
            labels->points[j].label = 0;
            continue;
        }

        labels->points[j].label = 1;
    }

    std::vector<bool> excluded_labels;
    excluded_labels.resize (2, false);
    excluded_labels[0] = true;

    euclidean_cluster_comparator->setInputCloud (surface_);
    euclidean_cluster_comparator->setLabels (labels);
    euclidean_cluster_comparator->setExcludeLabels (excluded_labels);
    euclidean_cluster_comparator->setDistanceThreshold (cluster_tolerance_, true);
    euclidean_cluster_comparator->setAngularThreshold(0.017453 * 5.f); //5 degrees

    pcl::PointCloud<pcl::Label> euclidean_labels;
    pcl::OrganizedConnectedComponentSegmentation<PointT, pcl::Label> euclidean_segmentation (euclidean_cluster_comparator);
    euclidean_segmentation.setInputCloud (surface_);
    euclidean_segmentation.segment (euclidean_labels, clusters);

    {
        pcl::visualization::PCLVisualizer vis("model smooth surfaces");
        int v1, v2;
        vis.createViewPort(0,0,0.5,1.0,v1);
        vis.createViewPort(0.5, 0, 1, 1, v2);

        vis.addPointCloud<PointT>(surface_, "model", v1);

        unsigned int max_label = clusters.size() + 1;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr smooth_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);
        std::vector<uint32_t> label_colors;
        label_colors.reserve (max_label);
        srand (static_cast<unsigned int> (time (0)));
        while (label_colors.size () <= max_label )
        {
            uint8_t r = static_cast<uint8_t>( (rand () % 256));
            uint8_t g = static_cast<uint8_t>( (rand () % 256));
            uint8_t b = static_cast<uint8_t>( (rand () % 256));
            label_colors.push_back (static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
        }

        smooth_rgb->points.resize (surface_->points.size ());
        smooth_rgb->width = surface_->points.size();
        smooth_rgb->height = 1;
        smooth_rgb->is_dense = surface_->is_dense;

        for(size_t i=0; i < surface_->points.size(); i++)
        {
            smooth_rgb->points[i].getVector3fMap() = surface_->points[i].getVector3fMap();
            smooth_rgb->points[i].rgb = label_colors[0];
        }

        std::cout << "Number of clusters:" << clusters.size() << std::endl;

        for(size_t i=0; i < clusters.size(); i++)
        {

            if(clusters[i].indices.size() < 100)
                continue;

            for(size_t j=0; j < clusters[i].indices.size(); j++)
            {
                int idx_to_surface = clusters[i].indices[j];
                smooth_rgb->points[idx_to_surface].getVector3fMap() = surface_->points[idx_to_surface].getVector3fMap();
                smooth_rgb->points[idx_to_surface].rgb = label_colors[i+1];
            }
        }

        vis.addPointCloud(smooth_rgb, "labels_smooth", v2);
        vis.spin();
    }
}

void AndreasSegmentation(PointCloudPtr & surface_, pcl::PointCloud<pcl::Normal>::Ptr & normal_cloud, bool use_planes=false)
{

    std::vector<pcl::PointIndices> clusters;

    boost::shared_ptr<segmentation::Segmenter> segmenter_;

    segmenter_.reset(new segmentation::Segmenter);
    segmenter_->setModelFilename("data_rgbd_segmenter/ST-TrainAll.model.txt");
    segmenter_->setScaling("data_rgbd_segmenter/ST-TrainAll.scalingparams.txt");
    segmenter_->setUsePlanesNotNurbs(use_planes);
    segmenter_->setPointCloud(surface_);
    segmenter_->segment();

    std::vector<std::vector<int> > clusters_ = segmenter_->getSegmentedObjectsIndices();

    for (size_t i = 0; i < clusters_.size (); i++)
    {
        pcl::PointIndices indx;
        indx.indices = clusters_[i];
        clusters.push_back(indx);
    }

    {
        pcl::visualization::PCLVisualizer vis("model smooth surfaces");
        int v1, v2;
        vis.createViewPort(0,0,0.5,1.0,v1);
        vis.createViewPort(0.5, 0, 1, 1, v2);

        pcl::visualization::PointCloudColorHandlerRGBField<PointT> handler(surface_);
        vis.addPointCloud<PointT>(surface_, handler, "model", v1);

        unsigned int max_label = clusters.size() + 1;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr smooth_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);
        std::vector<uint32_t> label_colors;
        label_colors.reserve (max_label);
        srand (static_cast<unsigned int> (time (0)));
        while (label_colors.size () <= max_label )
        {
            uint8_t r = static_cast<uint8_t>( (rand () % 256));
            uint8_t g = static_cast<uint8_t>( (rand () % 256));
            uint8_t b = static_cast<uint8_t>( (rand () % 256));
            label_colors.push_back (static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
        }

        smooth_rgb->points.resize (surface_->points.size ());
        smooth_rgb->width = surface_->points.size();
        smooth_rgb->height = 1;
        smooth_rgb->is_dense = surface_->is_dense;

        for(size_t i=0; i < surface_->points.size(); i++)
        {
            smooth_rgb->points[i].getVector3fMap() = surface_->points[i].getVector3fMap();
            smooth_rgb->points[i].rgb = label_colors[0];
        }

        std::cout << "Number of clusters:" << clusters.size() << std::endl;

        for(size_t i=0; i < clusters.size(); i++)
        {

            if(clusters[i].indices.size() < 100)
                continue;

            for(size_t j=0; j < clusters[i].indices.size(); j++)
            {
                int idx_to_surface = clusters[i].indices[j];
                smooth_rgb->points[idx_to_surface].getVector3fMap() = surface_->points[idx_to_surface].getVector3fMap();
                smooth_rgb->points[idx_to_surface].rgb = label_colors[i+1];
            }
        }

        vis.addPointCloud(smooth_rgb, "labels_smooth", v2);
        vis.spin();
    }
}

void fitPlane(PointCloudPtr & smooth_cluster, Eigen::Vector4f & plane)
{
    //recompute coefficients based on distance to camera and normal?
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*smooth_cluster, centroid);
    Eigen::Vector3f c(centroid[0],centroid[1],centroid[2]);

    Eigen::MatrixXf M_w(smooth_cluster->points.size(), 3);

    float sum_w = 0.f;
    for(size_t k=0; k < smooth_cluster->points.size(); k++)
    {
        float d_c = (smooth_cluster->points[k].getVector3fMap()).norm();
        float w_k = std::max(1.f - std::abs(1.f - d_c), 0.f);
        M_w.row(k) = w_k * (smooth_cluster->points[k].getVector3fMap() - c);
        sum_w += w_k;
    }

    Eigen::Matrix3f scatter;
    scatter.setZero ();
    scatter = M_w.transpose() * M_w;

    Eigen::JacobiSVD<Eigen::MatrixXf> svd(scatter, Eigen::ComputeFullV);

    Eigen::Vector3f n = svd.matrixV().col(2);
    //flip normal if required
    if(n.dot(c*-1) < 0)
        n = n * -1.f;

    float d = n.dot(c) * -1.f;

    plane[0] = n[0];
    plane[1] = n[1];
    plane[2] = n[2];
    plane[3] = d;
}

int
main (int argc, char ** argv)
{

    std::string pcd_file = "";
    float Z_DIST_ = 1.5f;
    float radius_normals = 0.02f;
    bool use_NERN_normals = false;
    int seg_type = 0;
    float MIN_RATIO_ = 0.5f;
    bool vis_normals = false;
    bool bf = false;
    bool use_planes = false;

    pcl::console::parse_argument (argc, argv, "-use_planes", use_planes);
    pcl::console::parse_argument (argc, argv, "-vis_normals", vis_normals);
    pcl::console::parse_argument (argc, argv, "-pcd_file", pcd_file);
    pcl::console::parse_argument (argc, argv, "-seed_resolution", seed_resolution);
    pcl::console::parse_argument (argc, argv, "-dot_threshold", dot_threshold_);
    pcl::console::parse_argument (argc, argv, "-max_z", Z_DIST_);
    pcl::console::parse_argument (argc, argv, "-curv_threshold", curv_threshold_);
    pcl::console::parse_argument (argc, argv, "-cluster_tolerance", cluster_tolerance_);
    pcl::console::parse_argument (argc, argv, "-radius_normals", radius_normals);
    pcl::console::parse_argument (argc, argv, "-use_NERN_normals", use_NERN_normals);
    pcl::console::parse_argument (argc, argv, "-seg_type", seg_type);
    pcl::console::parse_argument (argc, argv, "-min_ratio", MIN_RATIO_);
    pcl::console::parse_argument (argc, argv, "-bf", bf);

    if (pcd_file.compare ("") == 0)
    {
        PCL_ERROR("Set the directory containing scenes\n");
        return -1;
    }

    PointCloudPtr input_cloud(new PointCloud);
    pcl::io::loadPCDFile(pcd_file, *input_cloud);

    if(Z_DIST_ > 0)
    {
        pcl::PassThrough<PointT> pass_;
        pass_.setFilterLimits (0.f, Z_DIST_);
        pass_.setFilterFieldName ("z");
        pass_.setInputCloud (input_cloud);
        pass_.setKeepOrganized (true);
        pass_.filter (*input_cloud);
    }

    pcl::PointCloud<pcl::Normal>::Ptr normal_cloud (new pcl::PointCloud<pcl::Normal>);

    if(use_NERN_normals)
    {

        pclA::NormalsEstimationNR::Parameter p;
        p.useOctree = true;
        p.maxIter = 10;
        p.maxPointDist = radius_normals;
        p.epsConverge = 0.0005f;

        pclA::NormalsEstimationNR nern;
        nern.setInputCloud(input_cloud);
        nern.setParameter(p);
        nern.compute();
        nern.getNormals(normal_cloud);
    }
    else
    {
        pcl::NormalEstimationOMP<PointT, pcl::Normal> ne;
        ne.setRadiusSearch(radius_normals);
        ne.setInputCloud (input_cloud);
        ne.compute (*normal_cloud);
    }

    if(bf)
    {
        pcl::FastBilateralFilter<PointT> bf;
        bf.setSigmaR(0.003f);
        bf.setSigmaS(3);
        bf.setInputCloud(input_cloud);
        bf.applyFilter(*input_cloud);
    }

    if(vis_normals)
    {
        pcl::visualization::PCLVisualizer vis("normals");
        vis.addPointCloud(input_cloud, "input");
        vis.addPointCloudNormals<PointT, pcl::Normal> (input_cloud, normal_cloud,10,0.02f, "normals");
        vis.spin();
    }

    std::vector<SmoothClusters> smooth_clusters;

    if(seg_type == 0)
    {
        SupervoxelsSegmentation(input_cloud, normal_cloud, smooth_clusters);
    }
    else if(seg_type == 1)
    {
        SmoothRegionsSegmentation(input_cloud, normal_cloud);
    }
    else if(seg_type == 2)
    {
        AndreasSegmentation(input_cloud, normal_cloud, use_planes);
    }

    std::sort(smooth_clusters.begin(), smooth_clusters.end(), sortClustersBySize);

    pcl::visualization::PCLVisualizer vis("test");
    vis.removeAllPointClouds();
    vis.addPointCloud(input_cloud, "input");

    for(size_t i=0; i < smooth_clusters.size(); i++)
    {
        std::cout << smooth_clusters[i].indices_.indices.size() << std::endl;
        PointCloudPtr smooth_clus(new PointCloud);
        pcl::copyPointCloud(*input_cloud, smooth_clusters[i].indices_, *smooth_clus);

        //fit a plane
        fitPlane(smooth_clus, smooth_clusters[i].plane_);
        std::cout << smooth_clusters[i].plane_ << std::endl;

        /*pcl::visualization::PointCloudColorHandlerCustom<PointT> handler(smooth_clus, 255,0,0);
        vis.addPointCloud(smooth_clus, handler, "smooth");
        vis.spin();
        vis.removePointCloud("smooth");*/
    }

    size_t MAX_planes_ = 20;
    for(size_t i=0; (i < smooth_clusters.size()) && (i < MAX_planes_); i++)
    {

        float d_i = smooth_clusters[i].plane_[3];
        Eigen::Vector3f n_i = Eigen::Vector3f(smooth_clusters[i].plane_[0],smooth_clusters[i].plane_[1],smooth_clusters[i].plane_[2]);

        PointCloudPtr smooth_clus(new PointCloud);
        pcl::copyPointCloud(*input_cloud, smooth_clusters[i].indices_, *smooth_clus);

        pcl::visualization::PointCloudColorHandlerCustom<PointT> handler(smooth_clus, 255,0,0);
        vis.addPointCloud(smooth_clus, handler, "smooth");

        PointCloudPtr candidates(new PointCloud);
        int n_candidates = 0;
        for(size_t j=(i+1); j < smooth_clusters.size(); j++)
        {
            float d_j = smooth_clusters[j].plane_[3];
            Eigen::Vector3f n_j = Eigen::Vector3f(smooth_clusters[j].plane_[0],smooth_clusters[j].plane_[1],smooth_clusters[j].plane_[2]);

            float dot = n_i.dot(n_j);
            float d = std::abs(d_i - d_j);
            if(dot > 0.9)
            {
                PointCloudPtr smooth_clus_j(new PointCloud);
                pcl::copyPointCloud(*input_cloud, smooth_clusters[j].indices_, *smooth_clus_j);

                int n_inliers = 0;
                for(size_t k=0; k < smooth_clus_j->points.size(); k++)
                {
                    Eigen::Vector4f p = smooth_clus_j->points[k].getVector4fMap();
                    p[3] = 1;
                    if(std::abs(p.dot(smooth_clusters[i].plane_)) <= cluster_tolerance_)
                    {
                        n_inliers++;
                    }
                }

                float ratio = n_inliers / float(smooth_clus_j->points.size());
                if(ratio > MIN_RATIO_)
                {
                    *candidates += *smooth_clus_j;
                    n_candidates++;
                }
            }
        }

        std::cout << "num candidates:" << n_candidates << std::endl;

        pcl::visualization::PointCloudColorHandlerCustom<PointT> handler_cand(candidates, 0,255,0);
        vis.addPointCloud(candidates, handler_cand, "candidates");
        vis.spin();

        vis.removePointCloud("smooth");
        vis.removePointCloud("candidates");
    }
}
