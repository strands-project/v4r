#include "v4r/registration/MultiSessionModelling.h"
#include "v4r/common/visibility_reasoning.h"
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <v4r/common/miscellaneous.h>
#include <boost/graph/kruskal_min_spanning_tree.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/biconnected_components.hpp>
#include <boost/graph/prim_minimum_spanning_tree.hpp>

template<class PointT>
v4r::Registration::MultiSessionModelling<PointT>::MultiSessionModelling()
{

}

template<class PointT>
float
v4r::Registration::MultiSessionModelling<PointT>::computeFSV (PointCloudTPtr & cloud,
                                                              pcl::PointCloud<pcl::Normal>::Ptr & normals,
                                                              std::vector<int> & indices,
                                                              Eigen::Matrix4f & pose,
                                                              PointCloudTPtr & range_image)
{
    v4r::VisibilityReasoning<PointT> vr (525.f, 640, 480);
    vr.setThresholdTSS (0.01f);

    PointCloudTPtr model(new pcl::PointCloud<PointT>());
    pcl::transformPointCloud(*cloud, indices, *model, pose);

    pcl::PointCloud<pcl::Normal>::Ptr model_normals (new pcl::PointCloud<pcl::Normal>);
    v4r::transformNormals(*normals, *model_normals, indices, pose);

    //Eigen::Matrix4f identity = Eigen::Matrix4f::Identity();

    float fsv = vr.computeFSVWithNormals (range_image, model, model_normals);

    //the more points on the same surface the more reliable this is, so increase fsv if not many points
    int ss = vr.getFSVUsedPoints();
    float ratio = 1.f - ss / static_cast<float>(model->points.size());

    //std::cout << model->points.size() << " " << indices.size() << " fsv:" << fsv << std::endl;

    /*if(fsv > 0.1)
    {*/
    /*    ///debug
        pcl::visualization::PCLVisualizer vis("correspondences");
        //int v1,v2;
        //vis.createViewPort(0,0,0.5,1,v1);
        //vis.createViewPort(0.5,0,1,1,v2);

        pcl::visualization::PointCloudColorHandlerCustom<PointT> handler(model, 255, 0, 0);
        vis.addPointCloud(range_image, "cloud_2");
        vis.addPointCloud(model, handler, "cloud_1");

        vis.spin();
    */
    /*}*/

    return fsv * ratio;
}

template<class PointT>
void
v4r::Registration::MultiSessionModelling<PointT>::computeCost(EdgeBetweenPartialModels & edge)
{
    //fsv computation
    //take all views of edge_i with respective indices and compute FSV with respect to the views of j_
    //do i need normals?

    float fsv = 0.f;
    int non_valid = 0;
    int total = 0;
    for(size_t ii=session_ranges_[edge.i_].first; ii <= session_ranges_[edge.i_].second; ii++)
    {
        PointCloudTPtr cloud = clouds_[ii];
        std::vector<int> & indices = getIndices(ii);
        Eigen::Matrix4f pose = edge.transformation_.inverse() * getPose(ii);

        for(size_t jj=session_ranges_[edge.j_].first; jj <= session_ranges_[edge.j_].second; jj++)
        {
            Eigen::Matrix4f pose_to_jj = poses_[jj].inverse() * pose;
            float fsv_loc = computeFSV(cloud, normals_[ii], indices, pose_to_jj, clouds_[jj]);
            if(fsv_loc < 0)
            {
                non_valid++;
            }
            else
            {
                fsv += fsv_loc;
            }

            total++;
        }
    }

    if(non_valid == total)
    {
        std::cout << "No valid edge..." << std::endl;
        edge.cost_ = std::numeric_limits<float>::infinity();
    }
    else
    {
        edge.cost_ = fsv / (total - non_valid);
    }
}

template<class PointT>
void
v4r::Registration::MultiSessionModelling<PointT>::compute()
{

    for(size_t a=0; a < reg_algos_.size(); a++)
    {
        reg_algos_[a]->setMSM(this);
        reg_algos_[a]->initialize(session_ranges_);
    }

    std::vector< std::vector< std::vector<EdgeBetweenPartialModels> > > edges;
    edges.resize(session_ranges_.size());
    for(size_t i=0; i < session_ranges_.size(); i++)
    {
        edges[i].resize(session_ranges_.size());
    }

    //for each session pair, call the class that registers two partial models and returns a set of poses
    for(size_t i=0; i < session_ranges_.size(); i++)
    {
        std::pair<int, int> pair_i = session_ranges_[i];
        for(size_t j=(i+1); j < session_ranges_.size(); j++)
        {
            std::pair<int, int> pair_j = session_ranges_[j];

            for(size_t a=0; a < reg_algos_.size(); a++)
            {
                reg_algos_[a]->setSessions(pair_i, pair_j);
                reg_algos_[a]->compute(i,j);

                //poses transform the RF of pair_j to the RF of pair_i
                std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > poses;
                reg_algos_[a]->getPoses(poses);

                std::cout << "poses between " << j << " and " << i << " :" << poses.size() << std::endl;

                for(size_t p=0; p < poses.size(); p++)
                {
                    EdgeBetweenPartialModels ebpm;
                    ebpm.transformation_ = poses[p];
                    ebpm.cost_ = -1;
                    ebpm.i_ = i;
                    ebpm.j_ = j;

                    edges[i][j].push_back(ebpm);
                }
            }
        }
    }

    //for each edge in the graph, compute a score quantifying quality of alignment
    //overlap? number of outliers? fsv average?

    for(size_t i=0; i < session_ranges_.size(); i++)
    {
        for(size_t j=(i+1); j < session_ranges_.size(); j++)
        {
            if(edges[i][j].size() > 0)
            {
                std::cout << "Number of edges:" << edges[i][j].size() << std::endl;
                //iterate over the edges and compute cost_ (the higher, the worse)

#pragma omp parallel for schedule(dynamic, 1) num_threads(4)
                for(size_t k=0; k < edges[i][j].size(); k++)
                {
                    computeCost(edges[i][j][k]);
                }
            }
        }
    }

    //final alignment between partial models? MST
    int final_alignment = 0; //0-MST, 1-something fancy?

    switch(final_alignment)
    {
    case 0:
    {

        std::vector<PointCloudTPtr> partial_model_clouds(session_ranges_.size());

        for(size_t i=0; i < session_ranges_.size(); i++)
        {
            partial_model_clouds[i].reset(new pcl::PointCloud<PointT>);

            for(int t=session_ranges_[i].first; t <= session_ranges_[i].second; t++)
            {
                typename pcl::PointCloud<PointT>::Ptr transformed(new pcl::PointCloud<PointT>);
                pcl::copyPointCloud(*getCloud(static_cast<size_t>(t)), getIndices(static_cast<size_t>(t)), *transformed);
                Eigen::Matrix4f pose_inv = getPose(static_cast<size_t>(t));
                pcl::transformPointCloud(*transformed, *transformed, pose_inv);
                *partial_model_clouds[i] += *transformed;
            }

        }

        Graph G;
        for(size_t i=0; i < session_ranges_.size(); i++)
        {
            boost::add_vertex((int)i, G);
        }

        for(size_t i=0; i < session_ranges_.size(); i++)
        {
            for(size_t j=(i+1); j < session_ranges_.size(); j++)
            {
                if(edges[i][j].size() > 0)
                {
                    float min_cost = std::numeric_limits<float>::infinity();
                    int min_k = -1;

                    for(size_t k=0; k < edges[i][j].size(); k++)
                    {
                        //std::cout << "cost:" << edges[i][j][k].cost_ << std::endl;
                        if(edges[i][j][k].cost_ < min_cost)
                        {
                            min_cost = edges[i][j][k].cost_;
                            min_k = static_cast<int>(k);
                        }
                    }

                    std::cout << j << " => " << i << " " << min_cost << " " << min_k << std::endl;
                    //add edge to graph... transformation maps from j to i (otherwise invert)
                    myEdge e;
                    e.edge_weight = edges[i][j][min_k].cost_;
                    e.transformation = edges[i][j][min_k].transformation_;
                    e.source_id = j;
                    e.target_id = i;
                    boost::add_edge ((int)j, (int)i, e, G);

                    /*std::stringstream title_str;
                    title_str << "best pw from " << j << " " << i;
                    pcl::visualization::PCLVisualizer vis(title_str.str().c_str());
                    vis.addCoordinateSystem(0.1);
                    for(size_t k=0; k < edges[i][j].size(); k++)
                    {
                        if(edges[i][j][k].cost_ * 0.25 <= min_cost)
                        {
                            std::cout << "cost:" << edges[i][j][k].cost_ << std::endl;

                            {
                                pcl::visualization::PointCloudColorHandlerRGBField<PointT> handler(partial_model_clouds[i]);
                                vis.addPointCloud(partial_model_clouds[i], handler, "cloud_i");
                            }

                            typename pcl::PointCloud<PointT>::Ptr transformed(new pcl::PointCloud<PointT>);
                            pcl::transformPointCloud(*partial_model_clouds[j], *transformed, edges[i][j][k].transformation_);

                            {
                                pcl::visualization::PointCloudColorHandlerRGBField<PointT> handler(transformed);
                                vis.addPointCloud(transformed, handler, "cloud_j");
                            }

                            vis.spin();
                            vis.removeAllPointClouds();
                        }
                    }*/

                }
            }
        }

        boost::property_map<Graph, boost::edge_weight_t>::type weightmap = boost::get(boost::edge_weight, G);
        Graph MST;
        for(size_t i=0; i < session_ranges_.size(); i++)
            boost::add_vertex((int)i, MST);

        //there is a bug in boost 1.54 that causes negative weights error (even though weights are positive)
        /*std::vector < boost::graph_traits<Graph>::vertex_descriptor > p (boost::num_vertices (G));
        boost::prim_minimum_spanning_tree (G, &p[0]);

        for (std::size_t i = 0; i != p.size (); ++i)
        {
            if (p[i] != i)
                std::cout << "parent[" << i << "] = " << p[i] << std::endl;
            else
                std::cout << "parent[" << i << "] = no parent" << std::endl;
        }


        typedef typename Graph::edge_iterator EdgeIterator;
        std::pair<EdgeIterator, EdgeIterator> edges = boost::edges(G);
        EdgeIterator edge;

        for (edge = edges.first; edge != edges.second; edge++)
        {
            typename boost::graph_traits<Graph>::vertex_descriptor s, t;
            s = boost::source(*edge, G);
            t = boost::target(*edge, G);

            if(p[s] == t || p[t] == s)
            {
                //edge in prim
                boost::add_edge ((int)s, (int)t, weightmap[*edge], MST);
            }
        }*/

        std::vector < Edge > spanning_tree;
        boost::kruskal_minimum_spanning_tree(G, std::back_inserter(spanning_tree));

        std::cout << "Print the edges in the MST:" << std::endl;
        for (std::vector < Edge >::iterator ei = spanning_tree.begin(); ei != spanning_tree.end(); ++ei)
        {
            std::cout << source(*ei, G) << " " << target(*ei, G) << std::endl;
            boost::add_edge(source(*ei, G), target(*ei, G), weightmap[*ei], MST);
        }


        std::cout << boost::num_edges(MST) << " " << boost::num_vertices(MST) << std::endl;
        output_session_poses_.resize(boost::num_vertices(MST));

        computeAbsolutePoses(MST, output_session_poses_);

        //finally, define output_cloud_poses_
        output_cloud_poses_.resize(clouds_.size());

        for(size_t i=0; i < output_session_poses_.size(); i++)
        {
            std::cout << output_session_poses_[i] << std::endl;
            for(int j=session_ranges_[i].first; j <= session_ranges_[i].second; j++)
            {
                Eigen::Matrix4f trans = output_session_poses_[i] * poses_[j];
                output_cloud_poses_[j] = trans;
            }
        }

        break;
    }

    default:
    {
        break;
    }
    }
}

template class V4R_EXPORTS v4r::Registration::MultiSessionModelling<pcl::PointXYZRGB>;

