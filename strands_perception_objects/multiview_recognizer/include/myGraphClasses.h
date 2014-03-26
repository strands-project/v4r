#ifndef MYGRAPHCLASSES_H
#define MYGRAPHCLASSES_H

#include <vector>
#include <iostream>
#include <string>
#include <sstream>
#include <boost/graph/adjacency_list.hpp>
#include <boost/filesystem.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/graphviz.hpp>
//#include <boost/graph/prim_minimum_spanning_tree.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/nonfree/features2d.hpp>
//#include <pcl/common/common.h>

#include <pcl/features/normal_3d.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/pfh.h>
#include <pcl/features/vfh.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/registration/correspondence_types.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/search/impl/flann_search.hpp>
#include <pcl/segmentation/organized_multi_plane_segmentation.h>
#include <pcl/segmentation/planar_polygon_fusion.h>
#include <pcl/segmentation/plane_coefficient_comparator.h>
#include <pcl/segmentation/euclidean_plane_coefficient_comparator.h>
#include <pcl/segmentation/rgb_plane_coefficient_comparator.h>
#include <pcl/segmentation/edge_aware_plane_comparator.h>
#include <pcl/segmentation/euclidean_cluster_comparator.h>
#include <pcl/segmentation/organized_connected_component_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <faat_pcl/3d_rec_framework/defines/faat_3d_rec_framework_defines.h>
#include <faat_pcl/3d_rec_framework/feature_wrapper/local/image/sift_local_estimator.h>
#include <faat_pcl/3d_rec_framework/pc_source/model_only_source.h>
#include <faat_pcl/3d_rec_framework/pc_source/partial_pcd_source.h>
#include <faat_pcl/3d_rec_framework/pipeline/multi_pipeline_recognizer.h>
#include <faat_pcl/3d_rec_framework/segmentation/multiplane_segmentation.h>
#include <faat_pcl/recognition/hv/hv_go.h>
#include <faat_pcl/recognition/hv/hv_go_1.h>
#include <faat_pcl/recognition/hv/hv_go_3D.h>
#include <faat_pcl/registration/fast_icp_with_gc.h>
#include <faat_pcl/registration/mv_lm_icp.h>
#include <faat_pcl/registration/registration_utils.h>
// #include <faat_pcl/utils/filesystem_utils.h>
#include <faat_pcl/utils/noise_models.h>
#include <faat_pcl/utils/pcl_opencv.h>
#include <faat_pcl/utils/registration_utils.h>

#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/String.h>
#include <pcl_conversions.h>
// #include "functions.h"
//#include "visual_recognizer/Hypotheses.h"
#include "recognition_srv_definitions/recognize.h"
#include "scitos_apps_msgs/action_buttons.h"
#include "segmenter.h"

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointInT;
typedef PointInT::ConstPtr ConstPointInTPtr;
typedef boost::shared_ptr< PointInT > PointInTPtr;
typedef pcl::PointXYZRGB PointT;
typedef pcl::Histogram<128> FeatureT;
typedef flann::L1<float> DistT;

class Hypothesis
{
public:
    Hypothesis ( std::string model_id, Eigen::Matrix4f transform, std::string origin, bool extended = false, bool verified = false );
    std::string model_id_, origin_;
    Eigen::Matrix4f transform_;
    bool extended_;
    bool verified_;
};

class View
{
public:
    View();
    //View(const View &view);
    boost::shared_ptr< pcl::PointCloud<pcl::PointXYZRGB> > pScenePCl;
//    boost::shared_ptr< pcl::PointCloud<pcl::PointXYZRGBNormal> > pSceneXYZRGBNormal;
    boost::shared_ptr< pcl::PointCloud<pcl::Normal> > pSceneNormals;
    //boost::shared_ptr< pcl::PointCloud<pcl::PointXYZRGB> > pScenePCl_f; //no table plane
    //boost::shared_ptr< pcl::PointCloud<pcl::PointXYZRGB> > pScenePCl_f_ds;
    boost::shared_ptr< pcl::PointCloud<FeatureT > > pSignatures;
    boost::shared_ptr< pcl::PointIndices > pIndices_above_plane;
    boost::shared_ptr< pcl::PointCloud<pcl::PointXYZRGB> > pKeypoints;
    std::string scene_filename_;
    //std::vector<std::string> model_ids_;
    std::vector<double> modelToViewCost;
    std::vector<Hypothesis> hypothesis;
    std::vector<Hypothesis> hypothesis_single_unverified;
    Eigen::Matrix4f absolute_pose;
    pcl::PointIndices keypoints_indices_;
    bool has_been_hopped_;
    double cumulative_weight_to_new_vrtx_;
};

class myEdge
{
public:
    myEdge();
    Eigen::Matrix4f transformation;
    double edge_weight;
    std::string model_name;
    std::string source_id, target_id;
    std::vector <cv::DMatch> matches;
    bool edge_weight_has_been_calculated_;
};

using namespace boost;
namespace bf = boost::filesystem;
//typedef adjacency_list < listS, listS, undirectedS, property<vertex_distance_t, int>, property<edge_weight_t, double> > GraphMST;
//typedef graph_traits < GraphMST >::vertex_descriptor VertexMST;
//typedef graph_traits < GraphMST >::edge_descriptor EdgeMST;

//--"copy"-of-graph-to-save-custom-information------prim_minimum_spanning_tree----cannot(?)-handle-internal-bundled-properties---
typedef adjacency_list < vecS, vecS, undirectedS, View, myEdge > Graph;
typedef graph_traits < Graph >::vertex_descriptor Vertex;
typedef graph_traits < Graph >::edge_descriptor Edge;
typedef graph_traits<Graph>::vertex_iterator vertex_iter;
typedef graph_traits<Graph>::edge_iterator edge_iter;
typedef property_map<Graph, vertex_index_t>::type IndexMap;

typedef faat_pcl::rec_3d_framework::Model<PointT> ModelT;
typedef boost::shared_ptr<ModelT> ModelTPtr;
typedef typename pcl::PointCloud<PointT>::ConstPtr ConstPointInTPtr;

void
getFilesInDirect (bf::path & dir, std::string & rel_path_so_far, std::vector<std::string> & relative_paths, std::string & ext);
std::vector<int> visualization_framework ( pcl::visualization::PCLVisualizer::Ptr vis, int number_of_views, int number_of_subwindows_per_view );
void transformNormals(const pcl::PointCloud<pcl::Normal>::ConstPtr & normals_cloud,
                      pcl::PointCloud<pcl::Normal>::Ptr & normals_aligned,
                      Eigen::Matrix4f & transform);
void computeTablePlane (const boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZRGB> > & xyz_points, Eigen::Vector4f & table_plane, float z_dist=1.2f);
void filterPCl(boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZRGB> > pInputCloud, boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB> > pOutputCloud, float dist=1.5f);

class multiviewGraph
{
private:
    Graph grph_, grph_final_;
    std::vector<Edge> edges_, best_edges_;
    std::string topic_, models_dir_, training_dir_;
    ros::NodeHandle  *n_;
    ros::ServiceClient client_;
    ros::Subscriber sub_joy_, sub_pc_;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr current_cloud_;
    boost::mutex current_cloud_mutex_;
    boost::shared_ptr < faat_pcl::rec_3d_framework::ModelOnlySource<pcl::PointXYZRGBNormal, PointT> > models_source_;
    std::string object_to_be_highlighted_;
    bool visualize_output_;
    unsigned long recorded_clouds_;
    bool go_3d_;
    std::string input_cloud_dir_;
    int icp_iter_;
    int mv_keypoints_;
    int opt_type_;
    std::string gt_or_ouput_dir_;
    double chop_at_z_;
    float icp_resolution_;
    float icp_max_correspondence_distance_;
    bool do_reverse_hyp_extension;
    pcl::visualization::PCLVisualizer::Ptr vis_;
    
public:
    multiviewGraph(){
      recorded_clouds_ = 0;
      do_reverse_hyp_extension = false;
      go_3d_ = false;
      input_cloud_dir_ = "";
      mv_keypoints_ = 0;
      opt_type_ = 0;
      gt_or_ouput_dir_ = "";
      chop_at_z_ = 1.5f;
      icp_resolution_ = 0.005f;
      icp_max_correspondence_distance_ = 0.02f;
    }
    void joyCallback ( const scitos_apps_msgs::action_buttons& msg );
    void kinectCallback ( const sensor_msgs::PointCloud2& msg );
    int recognize ( pcl::PointCloud<pcl::PointXYZRGB> &cloud, const std::string scene_name = std::string() );
    void init ( int argc, char **argv );
    bool calcFeatures ( Vertex &src, Graph &grph );
    void estimateViewTransformationBySIFT ( const Vertex &src, const Vertex &trgt, Graph &grph, flann::Index<DistT > *flann_index, Eigen::Matrix4f &transformation, Edge &edge );
//    void selectLowestWeightEdgesFromParallelEdges ( const std::vector<Edge> &parallel_edges, const Graph &grph, std::vector<Edge> &single_edges );
    void extendHypothesis ( Graph &grph );
    std::vector<Hypothesis> extendHypothesisRecursive ( Graph &grph, Edge calling_out_edge);
//    void calcMST ( const std::vector<Edge> &edges, const Graph &grph, std::vector<Edge> &edges_final );
    void createEdgesFromHypothesisMatch ( Graph &grph, std::vector<Edge> &edges );
    void createEdgesFromHypothesisMatchOnline ( const Vertex new_vertex, Graph &grph, std::vector<Edge> &edges );
    void calcEdgeWeight (Graph &grph);
    void outputgraph ( Graph& map, const char* filename );
    std::vector<Vertex> my_node_reader ( std::string filename, Graph &g );
    void createBigPointCloud ( Graph & grph_final, pcl::PointCloud<pcl::PointXYZRGB>::Ptr & big_cloud );
    Vertex getFurthestVertex ( Graph &grph);
};


namespace multiview
{
void
nearestKSearch ( flann::Index<flann::L1<float> > * index,
                 float * descr, int descr_size,
                 int k,flann::Matrix<int> &indices,flann::Matrix<float> &distances );

template <typename Type>
void
convertToFLANN ( typename pcl::PointCloud<Type>::Ptr & cloud, flann::Matrix<float> &data );
}

#endif
