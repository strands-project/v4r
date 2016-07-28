#include <v4r_config.h>
#include <v4r/common/miscellaneous.h>
#include <v4r/io/filesystem.h>
#include <v4r/io/eigen.h>
#include <v4r/registration/FeatureBasedRegistration.h>
#include <v4r/registration/fast_icp_with_gc.h>
#include <pcl/console/parse.h>
#include <pcl/filters/passthrough.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/io/pcd_io.h>
#include <iostream>
#include <sstream>

#include <boost/graph/adjacency_list.hpp>
#include <boost/filesystem.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/kruskal_min_spanning_tree.hpp>

#ifdef HAVE_SIFTGPU
#include <v4r/features/sift_local_estimator.h>
#else
#include <v4r/features/opencv_sift_local_estimator.h>
#endif

struct CamConnect
{
    Eigen::Matrix4f transformation_;
    float edge_weight;
    size_t source_id_, target_id_;

    explicit CamConnect(float w) : edge_weight(w) { }

    CamConnect() : edge_weight(0.f)  {}

    bool operator<(const CamConnect& e) const { return edge_weight < e.edge_weight; }

    bool operator<=(const CamConnect& e) const { return edge_weight <= e.edge_weight; }

    bool operator>(const CamConnect& e) const { return edge_weight > e.edge_weight; }

    bool operator>=(const CamConnect& e) const { return edge_weight >= e.edge_weight ; }
};

class View
{
public:
    typedef pcl::PointXYZRGB PointT;

    std::vector<std::vector<float> >  sift_signatures_;
    std::vector<int> sift_keypoint_indices_;
    Eigen::Matrix4f camera_pose_;
    pcl::PointCloud<PointT>::Ptr  cloud_;

    View()
    {
        cloud_.reset(new pcl::PointCloud<PointT>());
    }
};

class scene_registration
{
private:
    typedef pcl::PointXYZRGB PointT;
    typedef flann::L1<float> DistT;

#ifdef HAVE_SIFTGPU
    cv::Ptr<SiftGPU> sift_;
#endif
    std::vector<View> grph_;

    typedef boost::property<boost::edge_weight_t, CamConnect> EdgeWeightProperty;
    typedef boost::adjacency_list < boost::vecS, boost::vecS, boost::undirectedS, size_t, EdgeWeightProperty> Graph;
    typedef boost::graph_traits < Graph >::vertex_descriptor Vertex;
    typedef boost::graph_traits < Graph >::edge_descriptor Edge;
    typedef boost::graph_traits<Graph>::vertex_iterator vertex_iter;
    typedef boost::property_map<Graph, boost::vertex_index_t>::type IndexMap;

    Graph gs_;

    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > absolute_poses_;

public:
    double chop_z_;

    scene_registration()
    {
        chop_z_ = std::numeric_limits<double>::max();
    }

    void
    calcEdgeWeightAndRefineTf (const typename pcl::PointCloud<PointT>::ConstPtr &cloud_src,
                                                            const typename pcl::PointCloud<PointT>::ConstPtr &cloud_dst,
                                                            const Eigen::Matrix4f &transform,
                                                            float &registration_quality,
                                                            Eigen::Matrix4f &refined_transform)
    {
        typename pcl::PointCloud<PointT>::Ptr cloud_src_wo_nan ( new pcl::PointCloud<PointT>());
        typename pcl::PointCloud<PointT>::Ptr cloud_dst_wo_nan ( new pcl::PointCloud<PointT>());

        pcl::PassThrough<PointT> pass;
        pass.setFilterLimits (0.f, 5.f);
        pass.setFilterFieldName ("z");
        pass.setInputCloud (cloud_src);
        pass.setKeepOrganized (true);
        pass.filter (*cloud_src_wo_nan);

        pcl::PassThrough<PointT> pass2;
        pass2.setFilterLimits (0.f, 5.f);
        pass2.setFilterFieldName ("z");
        pass2.setInputCloud (cloud_dst);
        pass2.setKeepOrganized (true);
        pass2.filter (*cloud_dst_wo_nan);

        float w_after_icp_ = std::numeric_limits<float>::max ();
        const float best_overlap_ = 0.75f;

        v4r::FastIterativeClosestPointWithGC<PointT> icp;
        icp.setMaxCorrespondenceDistance ( 0.02f );
        icp.setInputSource ( cloud_src_wo_nan );
        icp.setInputTarget ( cloud_dst_wo_nan );
        icp.setUseNormals (true);
        icp.useStandardCG (true);
        icp.setNoCG(true);
        icp.setOverlapPercentage (best_overlap_);
        icp.setKeepMaxHypotheses (5);
        icp.setMaximumIterations (10);
        icp.align (transform);
        w_after_icp_ = icp.getFinalTransformation ( refined_transform );

        if ( w_after_icp_ < 0 || !pcl_isfinite ( w_after_icp_ ) )
            w_after_icp_ = std::numeric_limits<float>::max ();
        else
            w_after_icp_ = best_overlap_ - w_after_icp_;

        //    transform = icp_trans; // refined transformation
        registration_quality = w_after_icp_;
    }

    void
    calcSiftFeatures (const typename pcl::PointCloud<PointT>::Ptr &cloud_src,
                      std::vector<int> &sift_keypoint_indices,
                      std::vector<std::vector<float> > &sift_signatures)
    {
        v4r::SIFTLocalEstimation<PointT> estimator;
        estimator.setInputCloud(cloud_src);
        estimator.compute(sift_signatures);
        sift_keypoint_indices = estimator.getKeypointIndices();
    }


    void
    computeAbsolutePosesRecursive (const Graph & grph,
                                  const Vertex start,
                                  const Eigen::Matrix4f &accum,
                                  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > & absolute_poses,
                                  std::vector<bool> &hop_list)
    {
        boost::property_map<Graph, boost::edge_weight_t>::type weightmap = boost::get(boost::edge_weight, gs_);
        boost::graph_traits<Graph>::out_edge_iterator ei, ei_end;
        for (boost::tie (ei, ei_end) = boost::out_edges (start, grph); ei != ei_end; ++ei)
        {
            Vertex targ = boost::target (*ei, grph);
            size_t target_id = boost::target (*ei, grph);

            if(hop_list[target_id])
               continue;

            hop_list[target_id] = true;
            CamConnect my_e = weightmap[*ei];
            Eigen::Matrix4f intern_accum;
            Eigen::Matrix4f trans = my_e.transformation_;
            if( my_e.target_id_ != target_id)
            {
                Eigen::Matrix4f trans_inv;
                trans_inv = trans.inverse();
                trans = trans_inv;
            }
            intern_accum = accum * trans;
            absolute_poses[target_id] = intern_accum;
            computeAbsolutePosesRecursive (grph, targ, intern_accum, absolute_poses, hop_list);
        }
    }

    void
    computeAbsolutePoses (const Graph & grph,
                         std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > & absolute_poses,
                          const Eigen::Matrix4f &initial_transform = Eigen::Matrix4f::Identity())
    {
      size_t num_frames = boost::num_vertices(grph);
      absolute_poses.resize( num_frames );
      std::vector<bool> hop_list (num_frames, false);
      Vertex source_view = 0;
      hop_list[0] = true;
      Eigen::Matrix4f accum = initial_transform;
      absolute_poses[0] = accum;
      computeAbsolutePosesRecursive (grph, source_view, accum, absolute_poses, hop_list);
    }

    void addView(const pcl::PointCloud<PointT>::ConstPtr &cloud)
    {
        View view;

        pcl::PassThrough<PointT> pass;
        pass.setFilterLimits (0.f, chop_z_);
        pass.setFilterFieldName ("z");
        pass.setInputCloud (cloud);
        pass.setKeepOrganized (true);
        pass.filter (*view.cloud_);
        calcSiftFeatures( view.cloud_, view.sift_keypoint_indices_, view.sift_signatures_);
        grph_.push_back(view);
    }

    void buildGraph()
    {
        for (size_t view_a = 0; view_a < grph_.size(); view_a++)
        {
            for (size_t view_b = 0; view_b < view_a; view_b++)
            {
                std::vector<CamConnect> transforms;
                CamConnect edge;
                edge.source_id_ = view_b;
                edge.target_id_ = view_a;

                boost::shared_ptr<flann::Index<DistT> > flann_index;
                v4r::convertToFLANN<DistT>(grph_[view_b].sift_signatures_, flann_index );

                std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > sift_transforms =
                v4r::Registration::FeatureBasedRegistration<PointT>::estimateViewTransformationBySIFT(
                            *grph_[view_a].cloud_, *grph_[view_b].cloud_,
                            grph_[view_a].sift_keypoint_indices_, grph_[view_b].sift_keypoint_indices_,
                            grph_[view_a].sift_signatures_, grph_[view_b].sift_signatures_);

                for(size_t sift_tf_id = 0; sift_tf_id < sift_transforms.size(); sift_tf_id++)
                {
                    edge.transformation_ = sift_transforms[sift_tf_id];
                    transforms.push_back(edge);
                }


                size_t best_transform_id = 0;
                float lowest_edge_weight = std::numeric_limits<float>::max();
                for ( size_t trans_id = 0; trans_id < transforms.size(); trans_id++ )
                {
                    Eigen::Matrix4f icp_refined_trans;
                    calcEdgeWeightAndRefineTf(grph_[view_a].cloud_, grph_[view_b].cloud_, transforms[ trans_id ].transformation_, transforms[ trans_id ].edge_weight, icp_refined_trans);
                    transforms[ trans_id ].transformation_ = icp_refined_trans,
                    std::cout << "Edge weight is " << transforms[ trans_id ].edge_weight << " for edge connecting vertex " <<
                                 transforms[ trans_id ].source_id_ << " and " << transforms[ trans_id ].target_id_ << std::endl;

                    if(transforms[ trans_id ].edge_weight < lowest_edge_weight)
                    {
                        lowest_edge_weight = transforms[ trans_id ].edge_weight;
                        best_transform_id = trans_id;
                    }
                }
                boost::add_edge (transforms[best_transform_id].source_id_, transforms[best_transform_id].target_id_, transforms[best_transform_id], gs_);
            }
        }
    }

    void compute_mst()
    {
        boost::property_map<Graph, boost::edge_weight_t>::type weightmap = boost::get(boost::edge_weight, gs_);
        std::vector < Edge > spanning_tree;
        boost::kruskal_minimum_spanning_tree(gs_, std::back_inserter(spanning_tree));

        Graph grph_mst;
        std::cout << "Print the edges in the MST:" << std::endl;
        for (std::vector < Edge >::iterator ei = spanning_tree.begin(); ei != spanning_tree.end(); ++ei)
        {
            CamConnect my_e = weightmap[*ei];
            std::cout << "[" << source(*ei, gs_) << "->" << target(*ei, gs_) << "] with weight " << my_e.edge_weight << std::endl;
            boost::add_edge(source(*ei, gs_), target(*ei, gs_), weightmap[*ei], grph_mst);
        }

        computeAbsolutePoses(grph_mst, absolute_poses_);

        for(size_t view_id=0; view_id<absolute_poses_.size(); view_id++)
        {
            grph_[ view_id ].camera_pose_ = absolute_poses_ [ view_id ];
        }
    }


    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >
    getAbsolutePoses()
    {
        return absolute_poses_;
    }
};



int main (int argc, char ** argv)
{
    typedef pcl::PointXYZRGB PointT;
    std::string path;
    bool save_pose;
    double chop_z = std::numeric_limits<float>::max();
    pcl::console::parse_argument (argc, argv, "-path", path);
    pcl::console::parse_argument (argc, argv, "-chop_z", chop_z);
    pcl::console::parse_argument (argc, argv, "-save_pose", save_pose);

    std::cout << "Processing all point clouds in folder " << path;

    std::vector< std::string> sub_folder_names = v4r::io::getFoldersInDirectory(path);
    if(sub_folder_names.empty() )
        sub_folder_names.push_back("");

    for (size_t sub_folder_id=0; sub_folder_id < sub_folder_names.size(); sub_folder_id++)
    {
        const std::string path_sub = path + "/" + sub_folder_names[sub_folder_id];
        std::cout << "Processing all point clouds in folder " << path_sub;

        scene_registration sr;
        sr.chop_z_ = chop_z;

        std::vector < std::string > files_intern = v4r::io::getFilesInDirectory (path_sub, ".*.pcd", true);
        if (files_intern.empty())
        {
            std::cerr << "No files in directory: " << path_sub << ". Usage " << argv[0] << " -path <folder name> -rows <number of rows used for displaying files>" << std::endl;
            return -1;
        }

        for (size_t i=0; i < files_intern.size(); i++)
        {
            const std::string full_path = path_sub + "/"  + files_intern[i];
            pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
            pcl::io::loadPCDFile (full_path, *cloud);
            sr.addView(cloud);
        }

        sr.buildGraph();
        sr.compute_mst();
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > abs_poses = sr.getAbsolutePoses();

        if(save_pose)
        {
            for (size_t i=0; i < files_intern.size(); i++)
            {
                const std::string full_path = path_sub + "/"  + files_intern[i];
                pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
                pcl::io::loadPCDFile (full_path, *cloud);
                v4r::setCloudPose(abs_poses[i], *cloud);
                pcl::io::savePCDFileBinary(full_path, *cloud);
            }
        }
    }
}
