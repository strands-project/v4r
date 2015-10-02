#include "v4r/recognition/boost_graph_extension.h"
#include <boost/graph/graphviz.hpp>

struct my_graph_writer
{
	void
	operator() ( std::ostream& out ) const
	{
		out << "node [shape=circle color=blue]" << std::endl;
		// just an example, showing that local options override global
		out << "edge [color=red]" << std::endl;
	}
} myGraphWrite;

struct my_edge_writer
{
    my_edge_writer ( MVGraph& g_ ) :
		g ( g_ )
	{
	}
	;
	template<class Edge>
	void
	operator() ( std::ostream& out, Edge e )
	{
		// just an example, showing that local options override global
		out << " [color=purple]" << std::endl;
		out << " [label=\"" << g[e].edge_weight << boost::filesystem::path ( g[e].model_name ).stem ().string () << "\"]" << std::endl;
	}
	;
    MVGraph g;
};

struct my_node_writer
{
    typedef pcl::PointXYZRGB PointT;
    my_node_writer ( MVGraph& g_ ) :
		g ( g_ )
	{
	}
	;
	template<class Vertex>
	void
	operator() ( std::ostream& out, Vertex v )
	{
        out << " [label=\"" << g[v].id_ << "(" << g[v].id_ << ")\"]"
		    << std::endl;
        out << " [file=\"" << g[v].id_ << "\"]" << std::endl;
        out << " [index=\"" << g[v].id_ << "\"]" << std::endl;

        for ( std::vector<Hypothesis<PointT> >::iterator it_hyp = g[v].hypothesis_sv_.begin (); it_hyp != g[v].hypothesis_sv_.end (); ++it_hyp )
		{
		    out << " [hypothesis_model_id=\"" << it_hyp->model_id_ << "\"]" << std::endl;
		    out << " [hypothesis_transform=\"" << it_hyp->transform_ ( 0, 0 ) << " " << it_hyp->transform_ ( 0, 1 ) << " " << it_hyp->transform_ ( 0, 2 )
		        << " " << it_hyp->transform_ ( 0, 3 ) << " " << it_hyp->transform_ ( 1, 0 ) << " " << it_hyp->transform_ ( 1, 1 ) << " "
		        << it_hyp->transform_ ( 1, 2 ) << " " << it_hyp->transform_ ( 1, 3 ) << " " << it_hyp->transform_ ( 2, 0 ) << " " << it_hyp->transform_ ( 2, 1 )
		        << " " << it_hyp->transform_ ( 2, 2 ) << " " << it_hyp->transform_ ( 2, 3 ) << " " << it_hyp->transform_ ( 3, 0 ) << " "
		        << it_hyp->transform_ ( 3, 1 ) << " " << it_hyp->transform_ ( 3, 2 ) << " " << it_hyp->transform_ ( 3, 3 ) << " " << "\"]" << std::endl;
		    out << " [hypothesis_origin=\"" << it_hyp->origin_ << "\"]" << std::endl;
		    out << " [hypothesis_verified=\"" << it_hyp->verified_ << "\"]" << std::endl;
		}
	}
	;
    MVGraph g;
};

void outputgraph ( MVGraph& map, const char* filename )
{
	std::ofstream gout;
	gout.open ( filename );
	write_graphviz ( gout, map, my_node_writer ( map ), my_edge_writer ( map ), myGraphWrite );
}

View::View ()
{
    pScenePCl.reset ( new pcl::PointCloud<pcl::PointXYZRGB> );
    scene_f_.reset ( new pcl::PointCloud<pcl::PointXYZRGB> );
    pSceneNormals.reset ( new pcl::PointCloud<pcl::Normal> );
    kp_normals_.reset ( new pcl::PointCloud<pcl::Normal> );
//    pIndices_above_plane.reset ( new pcl::PointIndices );
    pSiftSignatures_.reset ( new pcl::PointCloud<FeatureT> );
    has_been_hopped_ = false;
    cumulative_weight_to_new_vrtx_ = 0;
}

Edge::Edge()
{
    edge_weight = std::numeric_limits<float>::max ();
    model_name = "";
//    edge_weight_has_been_calculated_ = false;
//    std::vector <cv::DMatch> matches;
}

//void shallowCopyVertexIntoOtherGraph(const Vertex vrtx_src, const Graph grph_src, Vertex &vrtx_target, Graph &grph_target)
//{
//    grph_target[vrtx_target].pScenePCl = grph_src[vrtx_src].pScenePCl;
//    grph_target[vrtx_target].pScenePCl_f = grph_src[vrtx_src].pScenePCl_f;
//    grph_target[vrtx_target].pSceneNormals = grph_src[vrtx_src].pSceneNormals;
//    grph_target[vrtx_target].filteredSceneIndices_ = grph_src[vrtx_src].filteredSceneIndices_;
//    grph_target[vrtx_target].pKeypointsMultipipe_ = grph_src[vrtx_src].pKeypointsMultipipe_;
//    grph_target[vrtx_target].hypotheses_ = grph_src[vrtx_src].hypotheses_;
//    grph_target[vrtx_target].pSiftSignatures_ = grph_src[vrtx_src].pSiftSignatures_;
//    grph_target[vrtx_target].sift_keypoints_scales = grph_src[vrtx_src].sift_keypoints_scales;
//    grph_target[vrtx_target].siftKeypointIndices_ = grph_src[vrtx_src].siftKeypointIndices_;
//    grph_target[vrtx_target].hypothesis_sv_ = grph_src[vrtx_src].hypothesis_sv_;
//    grph_target[vrtx_target].hypothesis_mv_ = grph_src[vrtx_src].hypothesis_mv_;
//    grph_target[vrtx_target].absolute_pose = grph_src[vrtx_src].absolute_pose;
//    grph_target[vrtx_target].transform_to_world_co_system_ = grph_src[vrtx_src].transform_to_world_co_system_;
//    grph_target[vrtx_target].has_been_hopped_ = grph_src[vrtx_src].has_been_hopped_;
//    grph_target[vrtx_target].cumulative_weight_to_new_vrtx_ = grph_src[vrtx_src].cumulative_weight_to_new_vrtx_;
//    grph_target[vrtx_target].keypointIndices_ = grph_src[vrtx_src].keypointIndices_;
//}

//void copyEdgeIntoOtherGraph(const Edge edge_src, const Graph grph_src, Edge &edge_target, Graph &grph_target)
//{
//    grph_target[edge_target].transformation = grph_src[edge_src].transformation;
//    grph_target[edge_target].edge_weight = grph_src[edge_src].edge_weight;
//    grph_target[edge_target].model_name = grph_src[edge_src].model_name;
//    grph_target[edge_target].source_id = grph_src[edge_src].source_id;
//    grph_target[edge_target].target_id = grph_src[edge_src].target_id;
//    grph_target[edge_target].edge_weight_has_been_calculated_ = grph_src[edge_src].edge_weight_has_been_calculated_;
//}

void pruneGraph (MVGraph &grph, size_t num_remaining_vertices)
{
    while(num_vertices(grph) > num_remaining_vertices)
    {
        ViewD vrtxToKill = getFurthestVertex(grph);

        std::vector<EdgeD> edges_to_be_removed;
        graph_traits<MVGraph>::out_edge_iterator out_i, out_end;
        for ( tie ( out_i, out_end ) = out_edges ( vrtxToKill, grph ); out_i != out_end; ++out_i )
        {
            edges_to_be_removed.push_back(*out_i);
        }

        for(size_t remover_id = 0; remover_id < edges_to_be_removed.size(); remover_id++)
        {
            remove_edge(edges_to_be_removed[remover_id], grph);
        }

        edges_to_be_removed.clear();   // This should fix a problem with newer boost versions that in_edges and out_edges are not treated seperately any more for undirected edges

        graph_traits<MVGraph>::in_edge_iterator in_i, in_end;
        for ( tie ( in_i, in_end ) = in_edges ( vrtxToKill, grph ); in_i != in_end; ++in_i )
        {
            edges_to_be_removed.push_back(*in_i);
        }
        for(size_t remover_id = 0; remover_id < edges_to_be_removed.size(); remover_id++)
        {
            remove_edge(edges_to_be_removed[remover_id], grph);
        }

        remove_vertex(vrtxToKill, grph);
    }
}

void resetHopStatus(MVGraph &grph)
{
    for (std::pair<vertex_iter, vertex_iter> vp = vertices ( grph ); vp.first != vp.second; ++vp.first )
    {
        grph[*vp.first].has_been_hopped_ = false;
    }

}

ViewD getFurthestVertex ( MVGraph &grph)
{
    std::pair<vertex_iter, vertex_iter> vp; //vp.first = running iterator..... vp.second = last iterator

    vp = vertices ( grph );
    ViewD furthest_vrtx = *vp.first;
    ++vp.first;

    for (; vp.first != vp.second; ++vp.first )
    {
        if(grph[*vp.first].cumulative_weight_to_new_vrtx_ > grph[furthest_vrtx].cumulative_weight_to_new_vrtx_)
        {
            furthest_vrtx = *vp.first;
        }
    }

    return furthest_vrtx;
}

//std::vector<Vertex> my_node_reader ( std::string filename, Graph &g )
//{
//    std::string fn, model_id, line, tf_str, origin, verified;
//    Eigen::Matrix4f tf;
//    std::ifstream myfile;
//    std::vector<Vertex> vertices_temp_v;

//    myfile.open ( filename.c_str () );

//    if ( myfile.is_open () )
//    {
//        while ( myfile.good () )
//        {
//            std::getline ( myfile, line );

//            int found = -1;
//            std::string searchstring ( "[file=\"" );
//            found = line.find ( searchstring );

//            if ( found > -1 )
//            {
//                Vertex v = boost::add_vertex ( g );
//                vertices_temp_v.push_back ( v );

//                fn = line.substr ( found + searchstring.length () );
//                fn.erase ( fn.end () - 2, fn.end () );

//                int read_state = 0;
//                while ( myfile.good () && read_state > -1 )
//                {
//                    std::getline ( myfile, line );

//                    searchstring = ";";
//                    found = line.find ( searchstring );
//                    if ( found > -1 )
//                    {
//                        read_state = -1;
//                        break;
//                    }
//                    else
//                    {
//                        searchstring = "[hypothesis_model_id=\"";
//                        found = line.find ( searchstring );
//                        if ( found > -1 )
//                        {
//                            model_id = line.substr ( found + searchstring.length () );
//                            model_id.erase ( model_id.end () - 2, model_id.end () );
//                            read_state++;
//                        }

//                        searchstring = "[hypothesis_transform=\"";
//                        found = line.find ( searchstring );
//                        if ( found > -1 )
//                        {
//                            tf_str = line.substr ( found + searchstring.length () );
//                            tf_str.erase ( tf_str.end () - 2, tf_str.end () );

//                            std::stringstream ( tf_str ) >> tf ( 0, 0 ) >> tf ( 0, 1 ) >> tf ( 0, 2 ) >> tf ( 0, 3 ) >> tf ( 1, 0 ) >> tf ( 1, 1 ) >> tf ( 1, 2 ) >> tf ( 1, 3 )
//                                                         >> tf ( 2, 0 ) >> tf ( 2, 1 ) >> tf ( 2, 2 ) >> tf ( 2, 3 ) >> tf ( 3, 0 ) >> tf ( 3, 1 ) >> tf ( 3, 2 ) >> tf ( 3, 3 );
//                            read_state++;

//                        }
//                        searchstring = "[hypothesis_origin=\"";
//                        found = line.find ( searchstring );
//                        if ( found > -1 )
//                        {
//                            origin = line.substr ( found + searchstring.length () );
//                            origin.erase ( origin.end () - 2, origin.end () );
//                            read_state++;

//                            searchstring = "[hypothesis_verified=\"";
//                            found = line.find ( searchstring );
//                            if ( found > -1 )
//                            {
//                                verified = line.substr ( found + searchstring.length () );
//                                verified.erase ( verified.end () - 2, verified.end () );
//                                read_state++;
//                            }
//                        }
//                    }
//                    if ( read_state >= 4 )
//                    {
//                        read_state = 0;
//                        Hypothesis hypothesis ( model_id, tf, origin, false, atoi ( verified.c_str() ) );
//                        g[v].hypothesis.push_back ( hypothesis );

//                        g[v].scene_filename = fn;
//                        g[v].pScenePCl.reset ( new pcl::PointCloud<pcl::PointXYZRGB> );
//                        pcl::io::loadPCDFile ( g[v].scene_filename, * ( g[v].pScenePCl ) );

//                    }
//                }
//            }
//        }
//        myfile.close ();
//    }
//    return vertices_temp_v;
//}

