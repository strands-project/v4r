#include "myGraphClasses.h"
#include <pcl/common/transforms.h>

void
getFilesInDirect ( bf::path & dir, std::string & rel_path_so_far, std::vector<std::string> & relative_paths, std::string & ext )
{
    bf::directory_iterator end_itr;
    for ( bf::directory_iterator itr ( dir ); itr != end_itr; ++itr )
    {
        //check if its a directory, then get models in it
        if ( bf::is_directory ( *itr ) )
        {
#if BOOST_FILESYSTEM_VERSION == 3
            std::string so_far = rel_path_so_far + ( itr->path ().filename () ).string() + "/";
#else
            std::string so_far = rel_path_so_far + ( itr->path () ).filename () + "/";
#endif

            bf::path curr_path = itr->path ();
            getFilesInDirect ( curr_path, so_far, relative_paths, ext );
        }
        else
        {
            //check that it is a ply file and then add, otherwise ignore..
            std::vector < std::string > strs;
#if BOOST_FILESYSTEM_VERSION == 3
            std::string file = ( itr->path ().filename () ).string();
#else
            std::string file = ( itr->path () ).filename ();
#endif

            boost::split ( strs, file, boost::is_any_of ( "." ) );
            std::string extension = strs[strs.size () - 1];

            if ( extension.compare ( ext ) == 0 )
            {
#if BOOST_FILESYSTEM_VERSION == 3
                std::string path = rel_path_so_far + ( itr->path ().filename () ).string();
#else
                std::string path = rel_path_so_far + ( itr->path () ).filename ();
#endif

                relative_paths.push_back ( path );
            }
        }
    }
}



//std::vector<Vertex> multiviewGraph::
//my_node_reader ( std::string filename, Graph &g )
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

struct my_node_writer
{
    my_node_writer ( Graph& g_ ) :
        g ( g_ )
    {
    }
    ;
    template<class Vertex>
    void
    operator() ( std::ostream& out, Vertex v )
    {
        out << " [label=\"" << g[v].scene_filename_ << "(" << boost::filesystem::path ( g[v].scene_filename_ ).stem ().string () << ")\"]"
            << std::endl;
        out << " [file=\"" << g[v].scene_filename_ << "\"]" << std::endl;
        out << " [index=\"" << g[v].scene_filename_ << "\"]" << std::endl;

        for ( std::vector<Hypothesis>::iterator it_hyp = g[v].hypothesis.begin (); it_hyp != g[v].hypothesis.end (); ++it_hyp )
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
    Graph g;
};

struct my_edge_writer
{
    my_edge_writer ( Graph& g_ ) :
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
    Graph g;
};

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

void multiviewGraph::
outputgraph ( Graph& map, const char* filename )
{
    std::ofstream gout;
    gout.open ( filename );
    write_graphviz ( gout, map, my_node_writer ( map ), my_edge_writer ( map ), myGraphWrite );
}

inline void
createBigPointCloudRecursive ( Graph & grph_final, pcl::PointCloud<pcl::PointXYZRGB>::Ptr & big_cloud, Vertex start, Vertex coming_from,
                               Eigen::Matrix4f accum )
{
    if ( boost::degree ( start, grph_final ) == 1 )
    {
        //check if target is like coming_from
        boost::graph_traits<Graph>::out_edge_iterator ei, ei_end;
        for ( boost::tie ( ei, ei_end ) = boost::out_edges ( start, grph_final ); ei != ei_end; ++ei )
        {
            if ( target ( *ei, grph_final ) == coming_from )
                return;
        }
    }

    boost::graph_traits<Graph>::out_edge_iterator ei, ei_end;
    std::vector < boost::graph_traits<Graph>::out_edge_iterator > edges;
    for ( boost::tie ( ei, ei_end ) = boost::out_edges ( start, grph_final ); ei != ei_end; ++ei )
    {

        if ( target ( *ei, grph_final ) == coming_from )
        {
            continue;
        }

        edges.push_back ( ei );
    }

    for ( size_t i = 0; i < edges.size (); i++ )
    {
        Eigen::Matrix4f internal_accum;
        Edge e = *edges[i];
        Vertex src = boost::source ( e, grph_final );
        Vertex targ = boost::target ( e, grph_final );
        Eigen::Matrix4f transform;
        if ( grph_final[e].source_id.compare( grph_final[src].scene_filename_ ) == 0)
        {
            PCL_WARN ( "inverse" );
            transform = grph_final[e].transformation.inverse ();
        }
        else
        {
            PCL_WARN ( "normal" );
            transform = grph_final[e].transformation;
        }

        internal_accum = accum * transform;
        std::cout << internal_accum << std::endl;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr trans ( new pcl::PointCloud<pcl::PointXYZRGB> );
        pcl::transformPointCloud ( *grph_final[targ].pScenePCl, *trans, internal_accum );
        *big_cloud += *trans;
        grph_final[targ].absolute_pose = internal_accum;
        createBigPointCloudRecursive ( grph_final, big_cloud, targ, start, internal_accum );
    }
}

void  multiviewGraph::
createBigPointCloud ( Graph & grph_final, pcl::PointCloud<pcl::PointXYZRGB>::Ptr & big_cloud )
{
    std::pair<vertex_iter, vertex_iter> vp;
    vp = vertices ( grph_final );
    Eigen::Matrix4f accum;
    accum.setIdentity ();
    *big_cloud += *grph_final[*vp.first].pScenePCl;
    grph_final[*vp.first].absolute_pose = accum;
    createBigPointCloudRecursive ( grph_final, big_cloud, *vp.first, *vp.first, accum );
}

bool multiviewGraph::
calcFeatures ( Vertex &src, Graph &grph )
{
    boost::shared_ptr < faat_pcl::rec_3d_framework::SIFTLocalEstimation<PointT, FeatureT> > estimator;
    estimator.reset ( new faat_pcl::rec_3d_framework::SIFTLocalEstimation<PointT, FeatureT> );

    PointInTPtr processed ( new PointInT );
    estimator->setIndices ( * ( grph[src].pIndices_above_plane ) );
    return estimator->estimate ( grph[src].pScenePCl, processed, grph[src].pKeypoints, grph[src].pSignatures );

    //----display-keypoints--------------------
    /*pcl::visualization::PCLVisualizer::Ptr vis_temp (new pcl::visualization::PCLVisualizer);
     pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler_rgb_verified (grph[*it_vrtx].pScenePCl);
     vis_temp->addPointCloud<pcl::PointXYZRGB> (grph[*it_vrtx].pScenePCl, handler_rgb_verified, "Hypothesis_1");
     pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler_rgb_verified2 (grph[*it_vrtx].pKeypoints);

     for (size_t keyId = 0; keyId < grph[*it_vrtx].pKeypoints->size (); keyId++)
     {
     std::stringstream sphere_name;
     sphere_name << "sphere_" << keyId;
     vis_temp->addSphere<pcl::PointXYZRGB> (grph[*it_vrtx].pKeypoints->at (keyId), 0.01, sphere_name.str ());
     }
     vis_temp->spin ();*/
}

void transformNormals ( const pcl::PointCloud<pcl::Normal>::ConstPtr & normals_cloud,
                        pcl::PointCloud<pcl::Normal>::Ptr & normals_aligned,
                        Eigen::Matrix4f & transform )
{
    normals_aligned.reset ( new pcl::PointCloud<pcl::Normal> );
    normals_aligned->points.resize ( normals_cloud->points.size () );
    normals_aligned->width = normals_cloud->width;
    normals_aligned->height = normals_cloud->height;
    for ( size_t k = 0; k < normals_cloud->points.size (); k++ )
    {
        Eigen::Vector3f nt ( normals_cloud->points[k].normal_x, normals_cloud->points[k].normal_y, normals_cloud->points[k].normal_z );
        normals_aligned->points[k].normal_x = static_cast<float> ( transform ( 0, 0 ) * nt[0] + transform ( 0, 1 ) * nt[1]
                + transform ( 0, 2 ) * nt[2] );
        normals_aligned->points[k].normal_y = static_cast<float> ( transform ( 1, 0 ) * nt[0] + transform ( 1, 1 ) * nt[1]
                + transform ( 1, 2 ) * nt[2] );
        normals_aligned->points[k].normal_z = static_cast<float> ( transform ( 2, 0 ) * nt[0] + transform ( 2, 1 ) * nt[1]
                + transform ( 2, 2 ) * nt[2] );
    }
}

void computeTablePlane ( const boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZRGB> > & xyz_points, Eigen::Vector4f & table_plane, float z_dist )
{
    pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
    ne.setNormalEstimationMethod ( ne.COVARIANCE_MATRIX );
    ne.setMaxDepthChangeFactor ( 0.02f );
    ne.setNormalSmoothingSize ( 20.0f );
    ne.setBorderPolicy ( pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::Normal>::BORDER_POLICY_IGNORE );
    ne.setInputCloud ( xyz_points );
    pcl::PointCloud<pcl::Normal>::Ptr normal_cloud ( new pcl::PointCloud<pcl::Normal> );
    ne.compute ( *normal_cloud );

    int num_plane_inliers = 5000;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr xyz_points_andy ( new pcl::PointCloud<pcl::PointXYZRGB> );
    pcl::PassThrough<pcl::PointXYZRGB> pass_;
    pass_.setFilterLimits ( 0.f, z_dist );
    pass_.setFilterFieldName ( "z" );
    pass_.setInputCloud ( xyz_points );
    pass_.setKeepOrganized ( true );
    pass_.filter ( *xyz_points_andy );

    pcl::OrganizedMultiPlaneSegmentation<pcl::PointXYZRGB, pcl::Normal, pcl::Label> mps;
    mps.setMinInliers ( num_plane_inliers );
    mps.setAngularThreshold ( 0.017453 * 1.5f ); // 2 degrees
    mps.setDistanceThreshold ( 0.01 ); // 1cm
    mps.setInputNormals ( normal_cloud );
    mps.setInputCloud ( xyz_points_andy );

    std::vector<pcl::PlanarRegion<pcl::PointXYZRGB>, Eigen::aligned_allocator<pcl::PlanarRegion<pcl::PointXYZRGB> > > regions;
    std::vector<pcl::ModelCoefficients> model_coefficients;
    std::vector<pcl::PointIndices> inlier_indices;
    pcl::PointCloud<pcl::Label>::Ptr labels ( new pcl::PointCloud<pcl::Label> );
    std::vector<pcl::PointIndices> label_indices;
    std::vector<pcl::PointIndices> boundary_indices;

    pcl::PlaneRefinementComparator<pcl::PointXYZRGB, pcl::Normal, pcl::Label>::Ptr ref_comp (
                new pcl::PlaneRefinementComparator<pcl::PointXYZRGB,
                pcl::Normal, pcl::Label> () );
    ref_comp->setDistanceThreshold ( 0.01f, true );
    ref_comp->setAngularThreshold ( 0.017453 * 10 );
    mps.setRefinementComparator ( ref_comp );
    mps.segmentAndRefine ( regions, model_coefficients, inlier_indices, labels, label_indices, boundary_indices );

    std::cout << "Number of planes found:" << model_coefficients.size () << std::endl;

    int table_plane_selected = 0;
    int max_inliers_found = -1;
    std::vector<int> plane_inliers_counts;
    plane_inliers_counts.resize ( model_coefficients.size () );

    for ( size_t i = 0; i < model_coefficients.size (); i++ )
    {
        Eigen::Vector4f table_plane = Eigen::Vector4f ( model_coefficients[i].values[0], model_coefficients[i].values[1], model_coefficients[i].values[2],
                model_coefficients[i].values[3] );

        std::cout << "Number of inliers for this plane:" << inlier_indices[i].indices.size () << std::endl;
        int remaining_points = 0;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr plane_points ( new pcl::PointCloud<pcl::PointXYZRGB> ( *xyz_points_andy ) );
        for ( int j = 0; j < plane_points->points.size (); j++ )
        {
            Eigen::Vector3f xyz_p = plane_points->points[j].getVector3fMap ();

            if ( !pcl_isfinite ( xyz_p[0] ) || !pcl_isfinite ( xyz_p[1] ) || !pcl_isfinite ( xyz_p[2] ) )
                continue;

            float val = xyz_p[0] * table_plane[0] + xyz_p[1] * table_plane[1] + xyz_p[2] * table_plane[2] + table_plane[3];

            if ( std::abs ( val ) > 0.01 )
            {
                plane_points->points[j].x = std::numeric_limits<float>::quiet_NaN ();
                plane_points->points[j].y = std::numeric_limits<float>::quiet_NaN ();
                plane_points->points[j].z = std::numeric_limits<float>::quiet_NaN ();
            }
            else
                remaining_points++;
        }

        plane_inliers_counts[i] = remaining_points;

        if ( remaining_points > max_inliers_found )
        {
            table_plane_selected = i;
            max_inliers_found = remaining_points;
        }
    }

    size_t itt = static_cast<size_t> ( table_plane_selected );
    table_plane = Eigen::Vector4f ( model_coefficients[itt].values[0], model_coefficients[itt].values[1], model_coefficients[itt].values[2],
            model_coefficients[itt].values[3] );

    Eigen::Vector3f normal_table = Eigen::Vector3f ( model_coefficients[itt].values[0], model_coefficients[itt].values[1],
            model_coefficients[itt].values[2] );

    int inliers_count_best = plane_inliers_counts[itt];

    //check that the other planes with similar normal are not higher than the table_plane_selected
    for ( size_t i = 0; i < model_coefficients.size (); i++ )
    {
        Eigen::Vector4f model = Eigen::Vector4f ( model_coefficients[i].values[0], model_coefficients[i].values[1], model_coefficients[i].values[2],
                model_coefficients[i].values[3] );

        Eigen::Vector3f normal = Eigen::Vector3f ( model_coefficients[i].values[0], model_coefficients[i].values[1], model_coefficients[i].values[2] );

        int inliers_count = plane_inliers_counts[i];

        std::cout << "Dot product is:" << normal.dot ( normal_table ) << std::endl;
        if ( ( normal.dot ( normal_table ) > 0.95 ) && ( inliers_count_best * 0.5 <= inliers_count ) )
        {
            //check if this plane is higher, projecting a point on the normal direction
            std::cout << "Check if plane is higher, then change table plane" << std::endl;
            std::cout << model[3] << " " << table_plane[3] << std::endl;
            if ( model[3] < table_plane[3] )
            {
                PCL_WARN ( "Changing table plane..." );
                table_plane_selected = i;
                table_plane = model;
                normal_table = normal;
                inliers_count_best = inliers_count;
            }
        }
    }

    /*table_plane = Eigen::Vector4f (model_coefficients[table_plane_selected].values[0], model_coefficients[table_plane_selected].values[1],
    model_coefficients[table_plane_selected].values[2], model_coefficients[table_plane_selected].values[3]);*/

    std::cout << "Table plane computed... " << std::endl;
}


void
filterPCl ( const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr pInputCloud, boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB> > pOutputCloud,
            float dist )
{
    Eigen::Vector4f table_plane;
    computeTablePlane ( pInputCloud, table_plane );

    pcl::PassThrough<pcl::PointXYZRGB> pass;
    pass.setFilterLimits ( 0.f, dist );
    pass.setFilterFieldName ( "z" );
    pass.setInputCloud ( pInputCloud );
    pass.setKeepOrganized ( true );
    pass.filter ( *pOutputCloud );

#pragma omp parallel for
    for ( size_t kk = 0; kk < pOutputCloud->points.size (); kk++ )
    {
        Eigen::Vector3f xyz_p = pOutputCloud->points[kk].getVector3fMap ();

        if ( !pcl_isfinite ( xyz_p[0] ) || !pcl_isfinite ( xyz_p[1] ) || !pcl_isfinite ( xyz_p[2] ) )
            continue;

        float val = xyz_p[0] * table_plane[0] + xyz_p[1] * table_plane[1] + xyz_p[2] * table_plane[2] + table_plane[3];

        if ( val <= 0.01 )
        {
            pOutputCloud->points[kk].x = std::numeric_limits<float>::quiet_NaN ();
            pOutputCloud->points[kk].y = std::numeric_limits<float>::quiet_NaN ();
            pOutputCloud->points[kk].z = std::numeric_limits<float>::quiet_NaN ();
        }

    }
}


void
multiview::nearestKSearch ( flann::Index<flann::L1<float> > * index, float * descr, int descr_size, int k, flann::Matrix<int> &indices,
                            flann::Matrix<float> &distances )
{
    flann::Matrix<float> p = flann::Matrix<float> ( new float[descr_size], 1, descr_size );
    memcpy ( &p.ptr () [0], &descr[0], p.cols * p.rows * sizeof ( float ) );

    index->knnSearch ( p, indices, distances, k, flann::SearchParams ( 128 ) );
    delete[] p.ptr ();
}

template<typename Type>
void
multiview::convertToFLANN ( typename pcl::PointCloud<Type>::Ptr & cloud, flann::Matrix<float> &data )
{
    data.rows = cloud->points.size ();
    data.cols = sizeof ( cloud->points[0].histogram ) / sizeof ( float ); // number of histogram bins

    std::cout << data.rows << " " << data.cols << std::endl;

    flann::Matrix<float> flann_data ( new float[data.rows * data.cols], data.rows, data.cols );

    for ( size_t i = 0; i < data.rows; ++i )
        for ( size_t j = 0; j < data.cols; ++j )
        {
            flann_data.ptr () [i * data.cols + j] = cloud->points[i].histogram[j];
        }

    data = flann_data;
}

template void
multiview::convertToFLANN<pcl::Histogram<128> > ( pcl::PointCloud<pcl::Histogram<128> >::Ptr & cloud, flann::Matrix<float> &data ); // explicit instantiation.


void multiviewGraph::
estimateViewTransformationBySIFT ( const Vertex &src, const Vertex &trgt, Graph &grph, flann::Index<DistT> *flann_index, Eigen::Matrix4f &transformation,
                                   Edge &edge )
{
    int K = 1;
    flann::Matrix<int> indices = flann::Matrix<int> ( new int[K], 1, K );
    flann::Matrix<float> distances = flann::Matrix<float> ( new float[K], 1, K );

    pcl::CorrespondencesPtr temp_correspondences ( new pcl::Correspondences );
    PCL_INFO ( "Calculate transform via SIFT between scene %s and %s for a keypoint size of %ld", grph[src].scene_filename_.c_str(), grph[trgt].scene_filename_.c_str(), grph[src].pKeypoints->size () );
    for ( size_t keypointId = 0; keypointId < grph[src].pKeypoints->size (); keypointId++ )
    {
        FeatureT searchFeature = grph[src].pSignatures->at ( keypointId );
        int size_feat = sizeof ( searchFeature.histogram ) / sizeof ( float );
        multiview::nearestKSearch ( flann_index, searchFeature.histogram, size_feat, K, indices, distances );

        pcl::Correspondence corr;
        corr.distance = distances[0][0];
        corr.index_query = keypointId;
        corr.index_match = indices[0][0];
        temp_correspondences->push_back ( corr );
    }
    pcl::registration::CorrespondenceRejectorSampleConsensus<PointT>::Ptr rej;
    rej.reset ( new pcl::registration::CorrespondenceRejectorSampleConsensus<PointT> () );
    pcl::CorrespondencesPtr after_rej_correspondences ( new pcl::Correspondences () );

    rej->setMaximumIterations ( 50000 );
    rej->setInlierThreshold ( 0.02 );
    rej->setInputTarget ( grph[trgt].pKeypoints );
    rej->setInputSource ( grph[src].pKeypoints );
    rej->setInputCorrespondences ( temp_correspondences );
    rej->getCorrespondences ( *after_rej_correspondences );

    transformation = rej->getBestTransformation ();
    pcl::registration::TransformationEstimationSVD<PointT, PointT> t_est;
    t_est.estimateRigidTransformation ( *grph[src].pKeypoints, *grph[trgt].pKeypoints, *after_rej_correspondences, transformation );

    std::cout << "size of corr before " << temp_correspondences->size () << "; after: " << after_rej_correspondences->size () << std::endl;

    bool b;
    tie ( edge, b ) = add_edge ( trgt, src, grph );
    grph[edge].transformation = transformation;
    grph[edge].model_name = std::string ( "scene_to_scene" );
    grph[edge].source_id = grph[src].scene_filename_;
    grph[edge].target_id = grph[trgt].scene_filename_;


          pcl::visualization::PCLVisualizer::Ptr vis_temp2 (new pcl::visualization::PCLVisualizer);
          pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler_rgb_verified (grph[trgt].pScenePCl);
          vis_temp2->addPointCloud<pcl::PointXYZRGB> (grph[trgt].pScenePCl, handler_rgb_verified, "Hypothesis_1");
          PointInTPtr transformed_PCl (new pcl::PointCloud<pcl::PointXYZRGB>);
          pcl::transformPointCloud (*grph[src].pScenePCl, *transformed_PCl, grph[edge].transformation);
          pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler_rgb_verified2 (transformed_PCl);
          vis_temp2->addPointCloud<pcl::PointXYZRGB> (transformed_PCl, handler_rgb_verified2, "Hypothesis_2");
          vis_temp2->spin ();
}

//void multiviewGraph::
//selectLowestWeightEdgesFromParallelEdges ( const std::vector<Edge> &parallel_edges, const Graph &grph, std::vector<Edge> &single_edges )
//{
//    for ( size_t edgeVec_id = 0; edgeVec_id < parallel_edges.size (); edgeVec_id++ )
//    {
//        Vertex vrtx_src, vrtx_trgt;
//        vrtx_src = source ( parallel_edges[edgeVec_id], grph );
//        vrtx_trgt = target ( parallel_edges[edgeVec_id], grph );

//        bool found = false;
//        for ( size_t edges_lowestWeight_id = 0; edges_lowestWeight_id < single_edges.size (); edges_lowestWeight_id++ ) //select edge with lowest weight amongst parallel edges
//        {

//            //check if edge already exists in final graph between these two vertices
//            if ( ( ( ( boost::get ( vertex_index, grph, source ( single_edges[edges_lowestWeight_id], grph ) ) == boost::get ( vertex_index, grph, vrtx_src ) )
//                     && ( boost::get ( vertex_index, grph, target ( single_edges[edges_lowestWeight_id], grph ) ) == boost::get ( vertex_index, grph, vrtx_trgt ) ) )
//                   || ( ( boost::get ( vertex_index, grph, source ( single_edges[edges_lowestWeight_id], grph ) ) == boost::get ( vertex_index, grph, vrtx_trgt ) )
//                        && ( boost::get ( vertex_index, grph, target ( single_edges[edges_lowestWeight_id], grph ) ) == boost::get ( vertex_index, grph, vrtx_src ) ) ) ) )
//            {
//                found = true;
//                if ( grph[parallel_edges[edgeVec_id]].edge_weight < grph[single_edges[edges_lowestWeight_id]].edge_weight ) //check for lowest edge cost - if lower than currently lowest weight, then replace
//                {
//                    single_edges[edges_lowestWeight_id] = parallel_edges[edgeVec_id];
//                }
//                break;
//            }
//        }
//        if ( !found )
//            single_edges.push_back ( parallel_edges[edgeVec_id] );
//    }
//}

Vertex multiviewGraph::getFurthestVertex ( Graph &grph)
{
    std::pair<vertex_iter, vertex_iter> vp; //vp.first = running iterator..... vp.second = last iterator

    vp = vertices ( grph );
    Vertex furthest_vrtx = *vp.first;
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

std::vector<Hypothesis> multiviewGraph::
extendHypothesisRecursive ( Graph &grph, Edge calling_out_edge) //is directed edge (so the source of calling_edge is calling vertex)
{
    std::vector<Hypothesis> hyp_vec;

    typename graph_traits<Graph>::out_edge_iterator out_i, out_end;
    Vertex current_vertex = target ( calling_out_edge, grph );
    Vertex src = source ( calling_out_edge, grph );

    grph[current_vertex].has_been_hopped_ = true;
    grph[current_vertex].cumulative_weight_to_new_vrtx_ = grph[src].cumulative_weight_to_new_vrtx_ + grph[calling_out_edge].edge_weight;

    ROS_INFO("Current Vertex %s has a cumulative weight of %lf.", grph[current_vertex].scene_filename_.c_str(), grph[current_vertex].cumulative_weight_to_new_vrtx_);
    for ( tie ( out_i, out_end ) = out_edges ( current_vertex, grph ); out_i != out_end; ++out_i )
    {
        Vertex new_trgt = target ( *out_i, grph );

        if ( grph[new_trgt].has_been_hopped_ )
        {
            ROS_INFO("Vertex %s has already been hopped.", grph[new_trgt].scene_filename_.c_str());
            continue;
        }
        ROS_INFO("Hopping to vertex %s...", grph[new_trgt].scene_filename_.c_str());
        std::vector<Hypothesis> new_hypotheses = extendHypothesisRecursive ( grph, *out_i);
        for(std::vector<Hypothesis>::iterator it_new_hyp = new_hypotheses.begin(); it_new_hyp !=new_hypotheses.end(); ++it_new_hyp)
        {
            if ( grph[calling_out_edge].source_id.compare( grph[src].scene_filename_ ) == 0)
            {
                it_new_hyp->transform_ = grph[calling_out_edge].transformation.inverse () * it_new_hyp->transform_ ;
            }
            else if (grph[calling_out_edge].target_id.compare( grph[src].scene_filename_ ) == 0)
            {
                it_new_hyp->transform_ = grph[calling_out_edge].transformation * it_new_hyp->transform_;
            }
            else
            {
                ROS_WARN("Something is messed up with the transformation.");
            }
            hyp_vec.push_back(*it_new_hyp);
        }
    }

    for ( std::vector<Hypothesis>::const_iterator it_hyp = grph[current_vertex].hypothesis.begin (); it_hyp != grph[current_vertex].hypothesis.end (); ++it_hyp )
    {
        if(!it_hyp->verified_)
            continue;

        Eigen::Matrix4f tf;
        if ( grph[calling_out_edge].source_id.compare( grph[src].scene_filename_ ) == 0)
        {
            tf = grph[calling_out_edge].transformation.inverse () * it_hyp->transform_;
        }
        else if (grph[calling_out_edge].target_id.compare( grph[src].scene_filename_ ) == 0)
        {
            tf = grph[calling_out_edge].transformation * it_hyp->transform_;
        }
        else
        {
            ROS_WARN("Something is messed up with the transformation.");
        }
        Hypothesis ht_temp ( it_hyp->model_id_, tf, it_hyp->origin_, true );
        hyp_vec.push_back(ht_temp);
    }

    return hyp_vec;
}


void multiviewGraph::
extendHypothesis ( Graph &grph )
{
    bool something_has_been_updated = true;
    std::pair<vertex_iter, vertex_iter> vp; //vp.first = running iterator..... vp.second = last iterator

    while ( something_has_been_updated )
    {
        something_has_been_updated = false;
        for ( vp = vertices ( grph ); vp.first != vp.second; ++vp.first )
        {
            typename graph_traits<Graph>::out_edge_iterator out_i, out_end;
            for ( tie ( out_i, out_end ) = out_edges ( *vp.first, grph ); out_i != out_end; ++out_i )
            {
                Edge e = *out_i;
                Vertex src = source ( e, grph ), targ = target ( e, grph );

                if ( grph[src].scene_filename_ != grph[*vp.first].scene_filename_ )
                    PCL_WARN("something's wrong");

                size_t hypothesis_length_before_extension = grph[src].hypothesis.size ();

                for ( std::vector<Hypothesis>::iterator it_hypB = grph[targ].hypothesis.begin (); it_hypB != grph[targ].hypothesis.end (); ++it_hypB )
                {
                    bool hypotheses_from_view_exist = false;

                    //---check-if-hypotheses-from-updating-view-already-exist-in-current-view------------------------
                    for ( size_t id_hypA = 0; id_hypA < hypothesis_length_before_extension; id_hypA++ )
                    {
                        if ( grph[src].hypothesis[id_hypA].origin_ == it_hypB->origin_ )
                        {
                            hypotheses_from_view_exist = true;
                        }
                    }
                    if ( !hypotheses_from_view_exist )
                    {
                        Eigen::Matrix4f tf;
                        if ( grph[e].source_id.compare( grph[*vp.first].scene_filename_ ) == 0 )
                        {
                            tf = grph[e].transformation.inverse () * it_hypB->transform_;
                        }
                        else
                        {
                            tf = grph[e].transformation * it_hypB->transform_;
                        }

                        Hypothesis ht_temp ( it_hypB->model_id_, tf, it_hypB->origin_, true );
                        grph[*vp.first].hypothesis.push_back ( ht_temp );
                        something_has_been_updated = true;
                    }
                }
            }
        }
    }
}

//void multiviewGraph::
//calcMST ( const std::vector<Edge> &edges, const Graph &grph, std::vector<Edge> &edges_final )
//{
//    GraphMST grphMST;
//    std::vector<VertexMST> verticesMST_v;
//    std::vector<Edge> edges_lowestWeight;

//    for ( std::pair<vertex_iter, vertex_iter> vp = vertices ( grph ); vp.first != vp.second; ++vp.first )
//    {
//        VertexMST vrtxMST = boost::add_vertex ( grphMST );
//        verticesMST_v.push_back ( vrtxMST );
//    }

//    selectLowestWeightEdgesFromParallelEdges ( edges, grph, edges_lowestWeight );

//    //---create-input-for-Minimum-Spanning-Tree-calculation-------------------------------
//    for ( size_t edgeVec_id = 0; edgeVec_id < edges_lowestWeight.size (); edgeVec_id++ )
//    {
//        Vertex vrtx_src, vrtx_trgt;
//        vrtx_src = source ( edges_lowestWeight[edgeVec_id], grph );
//        vrtx_trgt = target ( edges_lowestWeight[edgeVec_id], grph );
//        add_edge ( verticesMST_v[get ( vertex_index, grph, vrtx_src )], verticesMST_v[get ( vertex_index, grph, vrtx_trgt )],
//                grph[edges_lowestWeight[edgeVec_id]].edge_weight, grphMST );
//    }

//#if defined(BOOST_MSVC) && BOOST_MSVC <= 1300
//    std::cout << "Boost Version not supported (you are using BOOST_MSVC Version: " << BOOST_MSVC << ", BOOST_MSVC > 1300 needed)" << std::endl;
//#else
//    std::vector < graph_traits<GraphMST>::vertex_descriptor > p ( num_vertices ( grphMST ) );
//    prim_minimum_spanning_tree ( grphMST, &p[0] );

//    dynamic_properties dp;
//    dp.property ( "node_id", get ( vertex_index, grphMST ) );
//    dp.property ( "weight", get ( edge_weight, grphMST ) );
//    std::cout << "Result Prims Algorithm: \n======================" << std::endl;
//    write_graphviz_dp ( std::cout, grphMST, dp, "node_id" );
//    std::cout << " There are " << boost::num_edges ( grphMST ) << " edges in the graph grph." << std::endl;

//#endif

//    for ( std::size_t i = 0; i != p.size (); ++i )
//    {
//        if ( p[i] != i )
//            std::cout << "parent[" << i << "] = " << p[i] << std::endl;
//        else
//            std::cout << "parent[" << i << "] = no parent" << std::endl;
//    }

//    for ( size_t edgeVec_id = 0; edgeVec_id < edges_lowestWeight.size (); edgeVec_id++ )
//    {
//        Vertex vrtx_src, vrtx_trgt;
//        vrtx_src = source ( edges_lowestWeight[edgeVec_id], grph );
//        vrtx_trgt = target ( edges_lowestWeight[edgeVec_id], grph );
//        if ( p[boost::get ( vertex_index, grph, vrtx_src )] == boost::get ( vertex_index, grph, vrtx_trgt ) || p[boost::get ( vertex_index, grph, vrtx_trgt )]
//             == boost::get ( vertex_index, grph, vrtx_src ) ) //check if edge represents an edge of Prim's Minimum Spanning Tree
//        {
//            edges_final.push_back ( edges_lowestWeight[edgeVec_id] );
//        }
//    }
//}

void multiviewGraph::
createEdgesFromHypothesisMatch (Graph &grph, std::vector<Edge> &edges )
{
    vertex_iter vertexItA, vertexEndA;
    for (boost::tie(vertexItA, vertexEndA) = vertices(grph_); vertexItA != vertexEndA; ++vertexItA)
    {
        for ( size_t hypVec_id = 0; hypVec_id < grph[*vertexItA].hypothesis.size (); hypVec_id++ )
        {
            vertex_iter vertexItB, vertexEndB;
            for (boost::tie(vertexItB, vertexEndB) = vertices(grph_); vertexItB != vertexItA; ++vertexItB)
            {
                //was for ( std::vector<Vertex>::const_iterator it_vrtxB = vertices_v.begin (); it_vrtxB != it_vrtxA; ++it_vrtxB )
                for ( std::vector<Hypothesis>::iterator it_hypB = grph[*vertexItB].hypothesis.begin (); it_hypB != grph[*vertexItB].hypothesis.end (); ++it_hypB )
                {
                    if ( it_hypB->model_id_.compare ( grph[*vertexItA].hypothesis[hypVec_id].model_id_ ) == 0 ) //model exists in other file -> create connection
                    {
                        Eigen::Matrix4f tf_temp = it_hypB->transform_ * grph[*vertexItA].hypothesis[hypVec_id].transform_.inverse (); //might be the other way around

                        //link views by an edge (for other graph)
                        Edge e_cpy;
                        bool b;
                        tie ( e_cpy, b ) = add_edge ( *vertexItA, *vertexItB, grph );
                        grph[e_cpy].transformation = tf_temp;
                        grph[e_cpy].model_name = grph[*vertexItA].hypothesis[hypVec_id].model_id_;
                        grph[e_cpy].source_id = grph[*vertexItA].scene_filename_;
                        grph[e_cpy].target_id = grph[*vertexItB].scene_filename_;
                        grph[e_cpy].edge_weight = std::numeric_limits<double>::max ();
                        edges.push_back ( e_cpy );

                        std::cout << "Creating Edge from " << grph[*vertexItA].scene_filename_ << " to " << grph[*vertexItB].scene_filename_
                                  << std::endl;
                    }
                }
            }
        }
    }
}

void multiviewGraph::
createEdgesFromHypothesisMatchOnline ( const Vertex new_vertex, Graph &grph, std::vector<Edge> &edges )
{
    vertex_iter vertexItA, vertexEndA;
    for (boost::tie(vertexItA, vertexEndA) = vertices(grph_); vertexItA != vertexEndA; ++vertexItA)
    {
        if ( grph[*vertexItA].scene_filename_.compare( grph[new_vertex].scene_filename_ ) == 0 )
        {
            continue;
        }

        ROS_INFO("Checking vertex %s, which has %ld hypotheses.", grph[*vertexItA].scene_filename_.c_str(), grph[*vertexItA].hypothesis.size());
        //for ( size_t hypVec_id = 0; hypVec_id < grph[*it_vrtxA].hypothesis.size (); hypVec_id++ )
        for ( std::vector<Hypothesis>::iterator it_hypA = grph[*vertexItA].hypothesis.begin (); it_hypA != grph[*vertexItA].hypothesis.end (); ++it_hypA )
        {
            for ( std::vector<Hypothesis>::iterator it_hypB = grph[new_vertex].hypothesis.begin (); it_hypB != grph[new_vertex].hypothesis.end (); ++it_hypB )
            {
                if ( it_hypB->model_id_.compare (it_hypA->model_id_ ) == 0 ) //model exists in other file (same id) --> create connection
                {
                    Eigen::Matrix4f tf_temp = it_hypB->transform_ * it_hypA->transform_.inverse (); //might be the other way around

                    //link views by an edge (for other graph)
                    Edge e_cpy;
                    bool b;
                    tie ( e_cpy, b ) = add_edge ( *vertexItA, new_vertex, grph );
                    grph[e_cpy].transformation = tf_temp;
                    grph[e_cpy].model_name = it_hypA->model_id_;
                    grph[e_cpy].source_id = grph[*vertexItA].scene_filename_;
                    grph[e_cpy].target_id = grph[new_vertex].scene_filename_;
                    grph[e_cpy].edge_weight = std::numeric_limits<double>::max ();
                    edges.push_back ( e_cpy );

                    std::cout << "Creating edge from " << grph[*vertexItA].scene_filename_ << " to " << grph[new_vertex].scene_filename_
                              << " for model match " << grph[e_cpy].model_name
                              << std::endl;
                    pcl::visualization::PCLVisualizer::Ptr vis_temp (new pcl::visualization::PCLVisualizer);
                    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler_rgb_verified (grph[new_vertex].pScenePCl);
                    vis_temp->addPointCloud<pcl::PointXYZRGB> (grph[new_vertex].pScenePCl, handler_rgb_verified, "VrtxA");

                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pAlignedPCl ( new pcl::PointCloud<pcl::PointXYZRGB> );
                    pcl::transformPointCloud (*grph[*vertexItA].pScenePCl, *pAlignedPCl, tf_temp);
                    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler_rgb_verified2 (pAlignedPCl);
                    vis_temp->addPointCloud<pcl::PointXYZRGB> (pAlignedPCl, handler_rgb_verified2, "pAlignedPCl");

                    vis_temp->spin();
                }
            }
        }
    }
}

View::View ()
{
    pScenePCl.reset ( new pcl::PointCloud<pcl::PointXYZRGB> );
    //pScenePCl_f.reset ( new pcl::PointCloud<pcl::PointXYZRGB> );
    pSceneNormals.reset ( new pcl::PointCloud<pcl::Normal> );
    //    pSceneXYZRGBNormal.reset ( new pcl::PointCloud<pcl::PointXYZRGBNormal> );
    //pScenePCl_f_ds.reset ( new pcl::PointCloud<pcl::PointXYZRGB> );
    pIndices_above_plane.reset ( new pcl::PointIndices );
    pSignatures.reset ( new pcl::PointCloud<FeatureT> );
    has_been_hopped_ = false;
    cumulative_weight_to_new_vrtx_ = 0;
}

Hypothesis::Hypothesis ( const std::string model_id, const Eigen::Matrix4f transform, const std::string origin, const bool extended, const bool verified )
{
    model_id_ = model_id;
    transform_ = transform;
    origin_ = origin;
    extended_ = extended;
    verified_ = verified;
}

myEdge::myEdge()
{
    edge_weight = std::numeric_limits<float>::max ();
    model_name = "";
    source_id = "";
    target_id = "";
    edge_weight_has_been_calculated_ = false;
    std::vector <cv::DMatch> matches;
}

void multiviewGraph::
calcEdgeWeight ( Graph &grph)
{
    //----calculate-edge-weight---------------------------------------------------------
    //pcl::visualization::PCLVisualizer::Ptr vis_temp ( new pcl::visualization::PCLVisualizer );
    //std::vector<int> viewportNr_temp = visualization_framework ( vis_temp, 2, 1 );
    std::pair<edge_iter, edge_iter> ep = edges(grph);
    for (; ep.first!=ep.second; ++ep.first) //std::vector<Edge>::iterator edge_it = edges.begin(); edge_it!=edges.end(); ++edge_it)
    {
        if(grph[*ep.first].edge_weight_has_been_calculated_)
            continue;
        //        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pTargetNormalPCl ( new pcl::PointCloud<pcl::PointXYZRGBNormal> );
        //        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pTargetPCl ( new pcl::PointCloud<pcl::PointXYZRGB> );
        //        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pSourcePCl ( new pcl::PointCloud<pcl::PointXYZRGB> );
        //        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pSourceNormalPCl ( new pcl::PointCloud<pcl::PointXYZRGBNormal> );


        double edge_weight;
        Vertex vrtx_src, vrtx_trgt;
        vrtx_src = source ( *ep.first, grph );
        vrtx_trgt = target ( *ep.first, grph );

        Eigen::Matrix4f transform;
        if ( grph[*ep.first].source_id.compare( grph[vrtx_src].scene_filename_ ))
        {
            transform = grph[*ep.first].transformation;
        }
        else
        {
            transform = grph[*ep.first].transformation.inverse ();
        }

        float w_after_icp_ = std::numeric_limits<float>::max ();

        //        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pTargetPCl_ficp ( new pcl::PointCloud<pcl::PointXYZRGB> );
        //        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pSourcePCl_ficp ( new pcl::PointCloud<pcl::PointXYZRGB> );

        float best_overlap_ = 0.75f;
        Eigen::Matrix4f icp_trans;
        faat_pcl::registration::FastIterativeClosestPointWithGC<pcl::PointXYZRGB> icp;
        icp.setMaxCorrespondenceDistance ( 0.02f );
        icp.setInputSource ( grph[vrtx_src].pScenePCl );
        icp.setInputTarget ( grph[vrtx_trgt].pScenePCl );
        icp.setUseNormals ( true );
        icp.useStandardCG ( true );
        icp.setOverlapPercentage ( best_overlap_ );
        icp.setKeepMaxHypotheses ( 1 );
        icp.setMaximumIterations ( 5 );
        icp.align ( transform );	// THERE IS A PROBLEM WITH THE VISUALIZER - CREATES VIS_Window WITHOUT ANY PURPOSE
        w_after_icp_ = icp.getFinalTransformation ( icp_trans );
        if ( w_after_icp_ < 0 || !pcl_isfinite ( w_after_icp_ ) )
        {
            w_after_icp_ = std::numeric_limits<float>::max ();
        }
        else
        {
            w_after_icp_ = best_overlap_ - w_after_icp_;
        }

        //        pcl::transformPointCloudWithNormals ( * ( grph[vrtx_src].pSceneXYZRGBNormal ), *pTargetNormalPCl, icp_trans );
        //        pcl::copyPointCloud ( * ( grph[vrtx_trgt].pSceneXYZRGBNormal ), *pSourceNormalPCl );

        if ( grph[*ep.first].source_id.compare( grph[vrtx_src].scene_filename_ ))
        {
            PCL_WARN ( "Normal...\n" );
            //icp trans is aligning source to target
            //transform is aligning source to target
            //grph[edges[edge_id]].transformation = icp_trans * grph[edges[edge_id]].transformation;
            grph[*ep.first].transformation = icp_trans;
        }
        else
        {
            //transform is aligning target to source
            //icp trans is aligning source to target
            PCL_WARN ( "Inverse...\n" );
            //grph[edges[edge_id]].transformation = icp_trans.inverse() * grph[edges[edge_id]].transformation;
            grph[*ep.first].transformation = icp_trans.inverse ();
        }

        edge_weight = w_after_icp_;


        std::cout << "WEIGHT IS: " << edge_weight << " coming from edge connecting " << grph[*ep.first].source_id
                  << " and " << grph[*ep.first].target_id << " by object_id: " << grph[*ep.first].model_name
                  << std::endl;

        grph[*ep.first].edge_weight = edge_weight;
        grph[*ep.first].edge_weight_has_been_calculated_ = true;

        /*vis_temp->removeAllPointClouds();
         pcl::visualization::PointCloudColorHandlerRGBField < pcl::PointXYZRGBNormal > handler_rgb_verified (pSourceNormalPCl);
         vis_temp->addPointCloud<pcl::PointXYZRGBNormal> (pSourceNormalPCl, handler_rgb_verified, "Hypothesis_1");
         pcl::visualization::PointCloudColorHandlerRGBField < pcl::PointXYZRGBNormal > handler_rgb_verified2 (pTargetNormalPCl);
         vis_temp->addPointCloud<pcl::PointXYZRGBNormal> (pTargetNormalPCl, handler_rgb_verified2, "Hypothesis_2");

         if(unusedCloud->points.size() > 0)
         {
         pcl::visualization::PointCloudColorHandlerCustom < pcl::PointXYZ > handler_rgb_verified2 (unusedCloud, 255, 0, 0);
         vis_temp->addPointCloud<pcl::PointXYZ> (unusedCloud, handler_rgb_verified2, "unused");
         }
         vis_temp->spin();*/
    }
}

void multiviewGraph::joyCallback ( const scitos_apps_msgs::action_buttons& msg )
{
    ROS_INFO ( "Button pressed." );
    recognize ( *current_cloud_ );
}

void multiviewGraph::kinectCallback ( const sensor_msgs::PointCloud2& msg )
{
    current_cloud_mutex_.lock();
    pcl::fromROSMsg ( msg, *current_cloud_ );
    current_cloud_mutex_.unlock();
}

int multiviewGraph::recognize ( pcl::PointCloud<pcl::PointXYZRGB> &cloud, const std::string scene_name )
{


    recognition_srv_definitions::recognize srv;

    sensor_msgs::PointCloud2 output;

    current_cloud_mutex_.lock();
    if (current_cloud_->empty())
    {
        PCL_WARN("Point cloud is empty!");
        return -1;
    }
    ROS_INFO ( "I am recognizing" );
    Vertex vrtx = boost::add_vertex ( grph_ );
    Vertex vrtx_final = boost::add_vertex ( grph_final_ );
    pcl::toROSMsg ( *current_cloud_, output );
    //*(grph_[vrtx].pScenePCl) = *current_cloud_;

    if(chop_at_z_ > 0)
    {
        pcl::PassThrough<PointT> pass_;
        pass_.setFilterLimits ( 0.f, chop_at_z_ );
        pass_.setFilterFieldName ( "z" );
        pass_.setInputCloud ( current_cloud_ );
        pass_.setKeepOrganized ( true );
        pass_.filter ( *grph_[vrtx].pScenePCl );
    }
    current_cloud_mutex_.unlock();

    if ( !scene_name.empty() )
    {
        grph_[vrtx].scene_filename_ = scene_name;
    }
    else
    {
        std::stringstream filename;
        filename << "scene_" << recorded_clouds_;
        grph_[vrtx].scene_filename_ = filename.str();
    }
    srv.request.cloud = output;
    recorded_clouds_++;

    if ( client_.call ( srv ) )
    {
        //std::vector<Eigen::Matrix4f> transforms;
        //std::vector<std::string> model_ids;
        std::vector<Edge> new_edges;

        if ( srv.response.ids.size() == 0 )
        {
            ROS_INFO ( "I didn't detect any object in the current scene." );
        }
        else
        {
            for ( size_t i=0; i<srv.response.ids.size(); i++ )
            {
                std_msgs::String object_id = ( std_msgs::String ) srv.response.ids[i];
                ROS_INFO ( "I detected object %s in the scene.", object_id.data.c_str() );
                //model_ids.push_back ( srv.response.ids[i].data );

                Eigen::Matrix4f tt;
                tt.setIdentity ( 4,4 );

                tt ( 0,3 ) = srv.response.transforms[i].translation.x;
                tt ( 1,3 ) = srv.response.transforms[i].translation.y;
                tt ( 2,3 ) = srv.response.transforms[i].translation.z;
                Eigen::Quaternionf q ( srv.response.transforms[i].rotation.w,
                                       srv.response.transforms[i].rotation.x,
                                       srv.response.transforms[i].rotation.y,
                                       srv.response.transforms[i].rotation.z );

                Eigen::Matrix3f rot = q.toRotationMatrix();
                tt.block<3,3> ( 0,0 ) = rot;

                //transforms.push_back ( tt );

                std::stringstream model_name;
                model_name << models_dir_ << srv.response.ids[i].data;
                Hypothesis hypothesis ( model_name.str(), tt, grph_[vrtx].scene_filename_, false );
                grph_[vrtx].hypothesis.push_back ( hypothesis );
            }
        }

        //---Normal estimation actually redundant because is also calculated in the single view recognition service
        pcl::NormalEstimationOMP<PointT, pcl::Normal> ne;
        ne.setRadiusSearch ( 0.02f );
        ne.setInputCloud ( grph_[vrtx].pScenePCl );
        ne.compute ( * ( grph_[vrtx].pSceneNormals ) );


        ROS_WARN("There are %ld vertices in the grph.", num_vertices(grph_));
        vertex_iter vertexIt, vertexEnd;
        boost::tie(vertexIt, vertexEnd) = vertices(grph_);
        for (; vertexIt != vertexEnd; ++vertexIt){
            std::cout << grph_[*vertexIt].scene_filename_ << std::endl;
        }

        createEdgesFromHypothesisMatchOnline ( vrtx, grph_, new_edges );

        //----------create-edges-between-views-by-SIFT-----------------------------------
        calcFeatures ( vrtx, grph_ );

        if (0)//num_vertices(grph_)>1)
        {
            flann::Matrix<float> flann_data;
            flann::Index<DistT> *flann_index;
            multiview::convertToFLANN<pcl::Histogram<128> > ( grph_[vrtx].pSignatures, flann_data );
            flann_index = new flann::Index<DistT> ( flann_data, flann::KDTreeIndexParams ( 4 ) );
            flann_index->buildIndex ();

            //#pragma omp parallel for
            vertex_iter vertexIt, vertexEnd;
            for (boost::tie(vertexIt, vertexEnd) = vertices(grph_); vertexIt != vertexEnd; ++vertexIt)
            {
                Eigen::Matrix4f transformation;
                Edge edge;
                if( grph_final_[*vertexIt].scene_filename_.compare ( grph_final_[vrtx].scene_filename_ ) != 0 )
                {
                    estimateViewTransformationBySIFT ( *vertexIt, vrtx, grph_, flann_index, transformation, edge );
                    new_edges.push_back ( edge );
                }
            }
            delete flann_index;
        }
        //----------END-create-edges-between-views-by-SIFT-----------------------------------

        calcEdgeWeight (grph_);
        outputgraph ( grph_, "complete_graph.dot" );

        //---copy-vertices-to-graph_final----------------------------
        grph_final_[vrtx_final].pScenePCl = grph_[vrtx].pScenePCl;
        //grph_final[vrtx_final].pScenePCl_f = grph[vrtx].pScenePCl_f;
        //grph_final[vrtx_final].pSceneNormal = grph[vrtx].pSceneNormal;
        grph_final_[vrtx_final].pSceneNormals = grph_[vrtx].pSceneNormals;
        //grph_final[vrtx_final].pScenePCl_f_ds = grph[vrtx].pScenePCl_f_ds;
        grph_final_[vrtx_final].scene_filename_ = grph_[vrtx].scene_filename_;
        grph_final_[vrtx_final].hypothesis = grph_[vrtx].hypothesis;
        grph_final_[vrtx_final].pKeypoints = grph_[vrtx].pKeypoints;
        grph_final_[vrtx_final].keypoints_indices_ = grph_[vrtx].keypoints_indices_;

        //std::vector<Edge> edges_final;
        //calcMST (edges_, grph_, edges_final);

        //        outputgraph (grph_, hypothesis_file.c_str ());

        //---copy-edges-that-are-left-after-MST-calculation-to-the-final-graph-------------------------------------------------------------
        //#pragma omp parallel for

        if ( num_vertices(grph_) > 1 )
        {
            //----find best edge from the freshly inserted ones and add it to the final graph
            Edge best_edge;
            best_edge = new_edges[0];

            for ( size_t new_edge_id = 0; new_edge_id < new_edges.size(); new_edge_id ++ )
            {
                edges_.push_back ( new_edges[new_edge_id] );

                if ( grph_[new_edges[new_edge_id]].edge_weight < grph_[best_edge].edge_weight )
                {
                    best_edge = new_edges[new_edge_id];
                }
            }

            Vertex vrtx_src, vrtx_trgt;
            vrtx_src = source ( best_edge, grph_ );
            vrtx_trgt = target ( best_edge, grph_ );

            Edge e_cpy;
            bool b;
            tie ( e_cpy, b ) = add_edge ( vrtx_src, vrtx_trgt, grph_final_ );

            grph_final_[e_cpy].transformation = grph_[best_edge].transformation;
            grph_final_[e_cpy].edge_weight = grph_[best_edge].edge_weight;
            grph_final_[e_cpy].model_name = grph_[best_edge].model_name;
            grph_final_[e_cpy].source_id = grph_[best_edge].source_id;
            grph_final_[e_cpy].target_id = grph_[best_edge].target_id;
            grph_final_[e_cpy].edge_weight_has_been_calculated_ = grph_[best_edge].edge_weight_has_been_calculated_;
            //}
            best_edges_.push_back ( best_edge );
            outputgraph ( grph_final_, "prune_graph.dot" );

            //------Extend-hypotheses-from-other-view(s)------------------------------------------
            typename graph_traits<Graph>::out_edge_iterator out_i, out_end;

            tie ( out_i, out_end ) = out_edges ( vrtx_final, grph_final_ );    //There should only be one edge

            std::cout << "Best edge: " << grph_final_[*out_i].edge_weight << " coming from edge " << grph_final_[*out_i].model_name << std::endl;
            //            if ( out_i != out_end-1 )
            //            {
            //                PCL_WARN("Something's crazy.");
            //            }
            grph_final_[vrtx_final].has_been_hopped_ = true;
            std::vector<Hypothesis> all_hypotheses = extendHypothesisRecursive ( grph_final_, *out_i );

            for(std::vector<Hypothesis>::const_iterator it_all_hyp = all_hypotheses.begin(); it_all_hyp != all_hypotheses.end(); ++it_all_hyp)
            {
                grph_[vrtx].hypothesis.push_back(*it_all_hyp);
                grph_final_[vrtx_final].hypothesis.push_back(*it_all_hyp);
            }
            ROS_INFO ("There are %ld hypotheses in the current view after extension, whereby %ld have been extended.",
                      grph_final_[vrtx_final].hypothesis.size(), all_hypotheses.size());

            grph_[vrtx].cumulative_weight_to_new_vrtx_ = grph_final_[vrtx_final].cumulative_weight_to_new_vrtx_;
        }


        std::pair<vertex_iter, vertex_iter> vp;
        for ( vp = vertices ( grph_final_ ); vp.first != vp.second; ++vp.first )
        {
            //-reset-hop-status
            grph_final_ [*vp.first].has_been_hopped_ = false;
        }


        //Refine hypotheses that where extended
        if ( icp_iter_ > 0 )
        {
            std::pair<vertex_iter, vertex_iter> vp;
            bool current_iteration_done = false;
            for ( vp = vertices ( grph_final_ ); (vp.first != vp.second) && (!current_iteration_done); ++vp.first )
            {
                Vertex vrtx_tmp;

                if ( ! do_reverse_hyp_extension)	// It only applies ICP on the latest scene point cloud
                {
                    vrtx_tmp = vrtx_final;
                    current_iteration_done = true;
                }
                else
                {
                    vrtx_tmp = *vp.first;
                }
                std::cout << "Hypotheses in this frame after extension:" << grph_final_[vrtx_tmp].hypothesis.size () << std::endl;

#pragma omp parallel for num_threads(8)
                for ( size_t kk = 0; kk < grph_final_[vrtx_tmp].hypothesis.size (); kk++ )
                {
                    if ( !grph_final_[vrtx_tmp].hypothesis[kk].extended_ )
                        continue;

                    std::vector < std::string > strs_2;
                    boost::split ( strs_2, grph_final_[vrtx_tmp].hypothesis[kk].model_id_, boost::is_any_of ( "/\\" ) );
                    ModelTPtr model;
                    bool found = models_source_->getModelById ( strs_2[strs_2.size () - 1], model );

                    if ( found )
                    {
                        boost::shared_ptr < distance_field::PropagationDistanceField<pcl::PointXYZRGB> > dt;
                        model->getVGDT ( dt );

                        faat_pcl::rec_3d_framework::VoxelBasedCorrespondenceEstimation<pcl::PointXYZRGB, pcl::PointXYZRGB>::Ptr
                                est (
                                    new faat_pcl::rec_3d_framework::VoxelBasedCorrespondenceEstimation<
                                    pcl::PointXYZRGB,
                                    pcl::PointXYZRGB> () );

                        pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZRGB>::Ptr
                                rej (
                                    new pcl::registration::CorrespondenceRejectorSampleConsensus<
                                    pcl::PointXYZRGB> () );

                        Eigen::Matrix4f scene_to_model_trans = grph_final_[vrtx_tmp].hypothesis[kk].transform_.inverse ();

                        pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud;
                        dt->getInputCloud ( cloud );

                        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_voxelized_icp_transformed ( new pcl::PointCloud<pcl::PointXYZRGB> () );
                        pcl::transformPointCloud ( *grph_final_[vrtx_tmp].pScenePCl, *cloud_voxelized_icp_transformed, scene_to_model_trans );

                        est->setVoxelRepresentationTarget ( dt );
                        est->setInputSource ( cloud_voxelized_icp_transformed );
                        est->setInputTarget ( cloud );
                        est->setMaxCorrespondenceDistance ( icp_max_correspondence_distance_ );
                        est->setMaxColorDistance ( -1, -1 );

                        rej->setInputTarget ( cloud );
                        rej->setMaximumIterations ( 1000 );
                        rej->setInlierThreshold ( 0.005f );
                        rej->setInputSource ( cloud_voxelized_icp_transformed );

                        pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> reg;
                        reg.setCorrespondenceEstimation ( est );
                        reg.addCorrespondenceRejector ( rej );
                        reg.setInputTarget ( cloud ); //model
                        reg.setInputSource ( cloud_voxelized_icp_transformed ); //scene
                        reg.setMaximumIterations ( icp_iter_ );
                        reg.setEuclideanFitnessEpsilon ( 1e-12 );
                        reg.setTransformationEpsilon ( 0.0001f * 0.0001f );

                        pcl::registration::DefaultConvergenceCriteria<float>::Ptr convergence_criteria;
                        convergence_criteria = reg.getConvergeCriteria ();
                        convergence_criteria->setAbsoluteMSE ( 1e-12 );
                        convergence_criteria->setMaximumIterationsSimilarTransforms ( 15 );
                        convergence_criteria->setFailureAfterMaximumIterations ( false );

                        PointInTPtr output ( new pcl::PointCloud<pcl::PointXYZRGB> () );
                        reg.align ( *output );
                        Eigen::Matrix4f trans, icp_trans;
                        trans = reg.getFinalTransformation () * scene_to_model_trans;
                        icp_trans = trans.inverse ();
                        grph_final_[vrtx_tmp].hypothesis[kk].transform_ = icp_trans;
                    }
                }
            }
        }


        //        if ( go_3d_ )
        //        {
        //            pcl::PointCloud<pcl::PointXYZRGB>::Ptr big_cloud ( new pcl::PointCloud<pcl::PointXYZRGB> );
        //            createBigPointCloud ( grph_final_, big_cloud );

        //            float leaf = 0.005f;
        //            pcl::PointCloud<pcl::PointXYZRGB>::Ptr big_cloud_vx ( new pcl::PointCloud<pcl::PointXYZRGB> );
        //            pcl::VoxelGrid<pcl::PointXYZRGB> sor;
        //            sor.setLeafSize ( leaf, leaf, leaf );
        //            sor.setInputCloud ( big_cloud );
        //            sor.filter ( *big_cloud_vx );

        //            std::pair<vertex_iter, vertex_iter> vp;
        //            std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> object_clouds;
        //            std::vector<pcl::PointCloud<pcl::Normal>::Ptr> normals_clouds;
        //            std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> scene_clouds;
        //            for ( vp = vertices ( grph_final_ ); vp.first != vp.second; ++vp.first )
        //            {

        //                std::cout << grph_final_[*vp.first].pKeypoints->points.size() << std::endl;
        //                Eigen::Vector4f table_plane;
        //                computeTablePlane ( grph_final_[*vp.first].pScenePCl, table_plane );
        //                pcl::PointCloud<pcl::PointXYZRGB>::Ptr trans_cloud ( new pcl::PointCloud<pcl::PointXYZRGB> ( *grph_final_[*vp.first].pScenePCl ) );

        //                pcl::PointIndices above_plane;
        //                for ( size_t kk = 0; kk < trans_cloud->points.size (); kk++ )
        //                {

        //                    Eigen::Vector3f xyz_p = trans_cloud->points[kk].getVector3fMap ();

        //                    if ( !pcl_isfinite ( xyz_p[0] ) || !pcl_isfinite ( xyz_p[1] ) || !pcl_isfinite ( xyz_p[2] ) )
        //                        continue;

        //                    float val = xyz_p[0] * table_plane[0] + xyz_p[1] * table_plane[1] + xyz_p[2] * table_plane[2] + table_plane[3];

        //                    if ( val <= -0.01 )
        //                    {
        //                        trans_cloud->points[kk].x = std::numeric_limits<float>::quiet_NaN ();
        //                        trans_cloud->points[kk].y = std::numeric_limits<float>::quiet_NaN ();
        //                        trans_cloud->points[kk].z = std::numeric_limits<float>::quiet_NaN ();
        //                    }
        //                    else
        //                    {
        //                        above_plane.indices.push_back ( kk );
        //                    }
        //                }

        //                pcl::PointCloud<pcl::Normal>::Ptr normal_cloud ( new pcl::PointCloud<pcl::Normal> ( *grph_final_[*vp.first].pSceneNormals ) );
        //                pcl::PointCloud<pcl::Normal>::Ptr normal_cloud_trans ( new pcl::PointCloud<pcl::Normal>() );
        //                transformNormals ( normal_cloud, normal_cloud_trans, grph_final_[*vp.first].absolute_pose );

        //                if ( mv_keypoints_ == 0 )
        //                    //using SIFT keypoints
        //                {
        //                    //compute indices to original cloud (to address normals) that are not farther away that 1.3
        //                    std::vector<int> sift_indices = grph_final_[*vp.first].keypoints_indices_.indices;
        //                    std::vector<int> original_indices;

        //                    for ( size_t kk=0; kk < sift_indices.size(); kk++ )
        //                    {
        //                        if ( trans_cloud->points[sift_indices[kk]].z > 1.3f )
        //                            continue;

        //                        if ( !pcl_isfinite ( trans_cloud->points[sift_indices[kk]].z ) )
        //                            continue;

        //                        original_indices.push_back ( sift_indices[kk] );
        //                    }

        //                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr trans_cloud2 ( new pcl::PointCloud<pcl::PointXYZRGB>() );
        //                    pcl::copyPointCloud ( *trans_cloud, original_indices, *trans_cloud2 );

        //                    pcl::transformPointCloud ( *trans_cloud2, *trans_cloud, grph_final_[*vp.first].absolute_pose );
        //                    pcl::copyPointCloud ( *normal_cloud_trans, original_indices, *normal_cloud );

        //                    std::cout << normal_cloud->points.size() << " " << trans_cloud->points.size() << std::endl;
        //                }
        //                else if ( mv_keypoints_ == 1 )
        //                {
        //                    //using RGB edges
        //                    std::vector<int> edge_indices;
        //                    registration_utils::getRGBEdges<pcl::PointXYZRGB> ( grph_final_[*vp.first].pScenePCl, edge_indices, 100, 150, 1.3f );
        //                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr trans_cloud2 ( new pcl::PointCloud<pcl::PointXYZRGB> );
        //                    pcl::copyPointCloud ( *grph_final_[*vp.first].pScenePCl, edge_indices, *trans_cloud2 );
        //                    pcl::transformPointCloud ( *trans_cloud2, *trans_cloud, grph_final_[*vp.first].absolute_pose );
        //                    pcl::copyPointCloud ( *normal_cloud_trans, edge_indices, *normal_cloud );
        //                }
        //                else if ( mv_keypoints_ == 2 )
        //                {
        //                    //using harris corners, not correctly implemented!
        //                    /*cv::Mat_ < cv::Vec3b > colorImage;
        //                    PCLOpenCV::ConvertPCLCloud2Image<pcl::PointXYZRGB> (grph_final[*vp.first].pScenePCl, colorImage);
        //                    cv::Mat grayImage;
        //                    cv::cvtColor (colorImage, grayImage, CV_BGR2GRAY);
        //                    int blockSize = 2;
        //                    int apertureSize = 3;
        //                    double k = 0.04;
        //                    cv::Mat dst, dst_norm, dst_norm_scaled;
        //                    cv::cornerHarris (grayImage, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);

        //                    cv::normalize (dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat ());
        //                    cv::convertScaleAbs (dst_norm, dst_norm_scaled);

        //                    std::vector<int> edge_indices;
        //                    float thresh = 150;
        //                    /// Drawing a circle around corners
        //                    for (int j = 0; j < dst_norm.rows; j++)
        //                    {
        //                        for (int i = 0; i < dst_norm.cols; i++)
        //                        {
        //                            if ((int)dst_norm.at<float> (j, i) > thresh)
        //                            {
        //                                int u, v;
        //                                v = i;
        //                                u = j;
        //                                if (pcl_isfinite(grph_final[*vp.first].pScenePCl->at(u,v).z)
        //                                        && pcl_isfinite(grph_final[*vp.first].pScenePCl->at(u,v).x)
        //                                        && pcl_isfinite(grph_final[*vp.first].pScenePCl->at(u,v).y))
        //                                {
        //                                    edge_indices.push_back (u * grph_final[*vp.first].pScenePCl->width + v);
        //                                }
        //                            }
        //                        }
        //                    }

        //                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr trans_cloud2 (new pcl::PointCloud<pcl::PointXYZRGB>);
        //                    pcl::copyPointCloud (*grph_final[*vp.first].pScenePCl, edge_indices, *trans_cloud2);
        //                    pcl::transformPointCloud (*trans_cloud2, *trans_cloud, grph_final[*vp.first].absolute_pose);*/
        //                }

        //                pcl::PointCloud<pcl::PointXYZRGB>::Ptr trans_cloud2 ( new pcl::PointCloud<pcl::PointXYZRGB> ( *trans_cloud ) );
        //                pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_cloud ( new pcl::PointCloud<pcl::PointXYZRGB> ( *grph_final_[*vp.first].pScenePCl ) );

        //                float w_t = 0.9f;
        //                bool depth_edges = true;
        //                float max_angle = 70.f;
        //                float lateral_sigma = 0.002f;

        //                faat_pcl::utils::noise_models::NguyenNoiseModel<pcl::PointXYZRGB> nm;
        //                nm.setInputCloud ( scene_cloud );
        //                nm.setInputNormals ( normal_cloud );
        //                nm.setLateralSigma ( lateral_sigma );
        //                nm.setMaxAngle ( max_angle );
        //                nm.setUseDepthEdges ( depth_edges );
        //                nm.compute();

        //                pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered;
        //                nm.getFilteredCloudRemovingPoints ( filtered, w_t );

        //                pcl::transformPointCloud ( *filtered, *scene_cloud, grph_final_[*vp.first].absolute_pose );

        //                scene_clouds.push_back ( scene_cloud );
        //                object_clouds.push_back ( trans_cloud2 );
        //                normals_clouds.push_back ( normal_cloud );
        //            }

        //            //refine registered scene clouds simulatenously and adapt transforms
        //            std::vector < std::vector<bool> > A;
        //            A.resize ( object_clouds.size () );
        //            for ( size_t i = 0; i < object_clouds.size (); i++ )
        //                A[i].resize ( object_clouds.size (), true );

        //            faat_pcl::registration_utils::computeOverlapMatrix<pcl::PointXYZRGB> ( object_clouds, A, 0.01, false );

        //            for ( size_t i = 0; i < object_clouds.size (); i++ )
        //            {
        //                for ( size_t j = 0; j < object_clouds.size (); j++ )
        //                    std::cout << ( int ) A[i][j] << " ";
        //                std::cout << std::endl;
        //            }

        //            float dt_size = 0.002f;
        //            float inlier_threshold = 0.005f;
        //            float max_corresp_dist = 0.01f;

        //            std::vector < Eigen::Matrix4f > transformations;
        //            transformations.resize ( object_clouds.size(), Eigen::Matrix4f::Identity() );

        //            {
        //                faat_pcl::registration::MVNonLinearICP<PointT> icp_nl ( dt_size );
        //                icp_nl.setInlierThreshold ( inlier_threshold );
        //                icp_nl.setMaxCorrespondenceDistance ( max_corresp_dist );
        //                icp_nl.setClouds ( object_clouds );
        //                icp_nl.setInputNormals ( normals_clouds );
        //                icp_nl.setVisIntermediate ( false );
        //                icp_nl.setSparseSolver ( true );
        //                icp_nl.setMaxIterations ( 5 );
        //                icp_nl.setAdjacencyMatrix ( A );
        //                icp_nl.setMinDot ( 0.5f );
        //                icp_nl.compute ();
        //                icp_nl.getTransformation ( transformations );
        //            }

        //            pcl::PointCloud<pcl::PointXYZRGB>::Ptr big_cloud_before_mv ( new pcl::PointCloud<pcl::PointXYZRGB> );
        //            pcl::PointCloud<pcl::PointXYZRGB>::Ptr big_cloud_after_mv ( new pcl::PointCloud<pcl::PointXYZRGB> );
        //            pcl::PointCloud<pcl::PointXYZRGB>::Ptr big_cloud_points_used_for_mv ( new pcl::PointCloud<pcl::PointXYZRGB> );

        //            for ( size_t i=0; i < object_clouds.size(); i++ )
        //            {
        //                pcl::PointCloud<pcl::PointXYZRGB>::Ptr trans_cloud ( new pcl::PointCloud<pcl::PointXYZRGB> );
        //                pcl::transformPointCloud ( *scene_clouds[i], *trans_cloud, transformations[i] );
        //                *big_cloud_after_mv += *trans_cloud;
        //                *big_cloud_before_mv += *scene_clouds[i];
        //                {
        //                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr trans_cloud ( new pcl::PointCloud<pcl::PointXYZRGB> );
        //                    pcl::transformPointCloud ( *object_clouds[i], *trans_cloud, transformations[i] );
        //                    *big_cloud_points_used_for_mv += *trans_cloud;
        //                }
        //            }

        //            pcl::PointCloud<pcl::PointXYZRGB>::Ptr big_cloud_vx_after_mv ( new pcl::PointCloud<pcl::PointXYZRGB> );
        //            pcl::PointCloud<pcl::PointXYZRGB>::Ptr big_cloud_vx_before_mv ( new pcl::PointCloud<pcl::PointXYZRGB> );
        //            {
        //                pcl::VoxelGrid<pcl::PointXYZRGB> sor;
        //                sor.setLeafSize ( leaf, leaf, leaf );

        //                sor.setInputCloud ( big_cloud_after_mv );
        //                sor.filter ( *big_cloud_vx_after_mv );

        //                sor.setInputCloud ( big_cloud_before_mv );
        //                sor.filter ( *big_cloud_vx_before_mv );
        //            }

        //            if ( visualize_output_ )
        //            {
        //                pcl::visualization::PCLVisualizer vis ( "registered cloud" );
        //                int v1, v2, v3;
        //                vis.createViewPort ( 0, 0, 0.33, 1, v1 );
        //                vis.createViewPort ( 0.33, 0, 0.66, 1, v2 );
        //                vis.createViewPort ( 0.66, 0, 1, 1, v3 );

        //                pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler ( big_cloud_points_used_for_mv );
        //                vis.addPointCloud ( big_cloud_points_used_for_mv, handler, "big", v1 );
        //                pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler2 ( big_cloud_vx_after_mv );
        //                vis.addPointCloud ( big_cloud_vx_after_mv, handler2, "big_mv", v2 );
        //                pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler3 ( big_cloud_vx_before_mv );
        //                vis.addPointCloud ( big_cloud_vx_before_mv, handler3, "keypoints_mv", v3 );

        //                vis.spin ();
        //            }

        //            //visualize the model hypotheses
        //            std::vector<pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr> aligned_models;
        //            std::vector < std::string > ids;
        //            std::vector<pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr> occlusion_clouds;
        //            std::vector < Eigen::Matrix4f > transforms_to_global;
        //            std::vector< Eigen::Matrix4f > hypotheses_poses_in_global_frame;

        //            std::vector<int> hyp_index_to_vp;
        //            int kk=0;
        //            for ( vp = vertices ( grph_final_ ); vp.first != vp.second; ++vp.first, kk++ )
        //            {
        //                std::cout << *vp.first << " " << kk << std::endl;
        //                occlusion_clouds.push_back ( grph_final_[*vp.first].pScenePCl );
        //                transforms_to_global.push_back ( transformations[kk] * grph_final_[*vp.first].absolute_pose );

        //                for ( std::vector<Hypothesis>::iterator it_hyp = grph_final_[*vp.first].hypothesis.begin (); it_hyp != grph_final_[*vp.first].hypothesis.end (); ++it_hyp )
        //                {
        //                    int vector_id = it_hyp - grph_final_[*vp.first].hypothesis.begin ();

        //                    if ( it_hyp->extended_ )
        //                        continue;

        //                    PointInTPtr pModelPCl ( new pcl::PointCloud<pcl::PointXYZRGB> );
        //                    PointInTPtr pModelPCl2 ( new pcl::PointCloud<pcl::PointXYZRGB> );
        //                    pcl::io::loadPCDFile ( it_hyp->model_id_, * ( pModelPCl ) );

        //                    Eigen::Matrix4f trans = transformations[kk] * grph_final_[*vp.first].absolute_pose * it_hyp->transform_;
        //                    pcl::transformPointCloud ( *pModelPCl, *pModelPCl, trans );

        //                    pcl::VoxelGrid<pcl::PointXYZRGB> sor;
        //                    sor.setLeafSize ( leaf, leaf, leaf );
        //                    sor.setInputCloud ( pModelPCl );
        //                    sor.filter ( *pModelPCl2 );

        //                    hypotheses_poses_in_global_frame.push_back ( trans );
        //                    aligned_models.push_back ( pModelPCl2 );
        //                    ids.push_back ( it_hyp->model_id_ );
        //                    hyp_index_to_vp.push_back ( *vp.first );
        //                }
        //            }

        //            std::cout << "number of hypotheses for GO3D:" << aligned_models.size() << std::endl;
        //            if ( aligned_models.size() > 0 )
        //            {

        //                //Refine aligned models with ICP
        //                for ( size_t kk=0; kk < aligned_models.size(); kk++ )
        //                {
        //                    pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
        //                    icp.setInputTarget ( big_cloud_vx_after_mv );
        //                    icp.setInputSource ( aligned_models[kk] );
        //                    icp.setMaxCorrespondenceDistance ( 0.015f );
        //                    icp.setMaximumIterations ( 15 );
        //                    icp.setRANSACIterations ( 1000 );
        //                    icp.setEuclideanFitnessEpsilon ( 1e-9 );
        //                    icp.setTransformationEpsilon ( 1e-9 );
        //                    pcl::PointCloud < PointT >::Ptr model_aligned ( new pcl::PointCloud<PointT> );
        //                    icp.align ( *model_aligned );

        //                    Eigen::Matrix4f icp_trans;
        //                    icp_trans = icp.getFinalTransformation();
        //                    hypotheses_poses_in_global_frame[kk] = icp_trans * hypotheses_poses_in_global_frame[kk];
        //                    aligned_models[kk] = model_aligned;
        //                }

        //                /*pcl::io::savePCDFileBinary("big_cloud.pcd", *big_cloud_after_mv);
        //                for(size_t kk=0; kk < occlusion_clouds.size(); kk++)
        //                {
        //                    {
        //                        std::stringstream original_cloud_name;
        //                        original_cloud_name << "cloud_" << std::setw(4) << std::setfill('0') << kk << ".pcd";
        //                        pcl::io::savePCDFileBinary(original_cloud_name.str(), *occlusion_clouds[kk]);
        //                    }

        //                    {
        //                        std::stringstream original_cloud_name;
        //                        original_cloud_name << "transformation_" << std::setw(4) << std::setfill('0') << kk << ".txt";
        //                        faat_pcl::utils::writeMatrixToFile(original_cloud_name.str(), transforms_to_global[kk]);
        //                    }
        //                }

        //                for(size_t kk=0; kk < aligned_models.size(); kk++)
        //                {
        //                    {
        //                        std::stringstream original_cloud_name;
        //                        original_cloud_name << "model_" << std::setw(4) << std::setfill('0') << kk << ".pcd";
        //                        pcl::io::savePCDFileBinary(original_cloud_name.str(), *aligned_models[kk]);
        //                    }
        //                }*/

        //                //Instantiate HV go 3D, reimplement addModels that will reason about occlusions
        //                //Set occlusion cloudS!!
        //                //Set the absolute poses so we can go from the global coordinate system to the occlusion clouds
        //                //TODO: Normals might be a problem!! We need normals from the models and normals from the scene, correctly oriented!
        //                //right now, all normals from the scene will be oriented towards some weird 0, same for models actually
        //                faat_pcl::GO3D<PointT, PointT> go;
        //                go.setResolution ( leaf );
        //                go.setAbsolutePoses ( transforms_to_global );
        //                go.setOcclusionsClouds ( occlusion_clouds );
        //                go.setZBufferSelfOcclusionResolution ( 250 );
        //                go.setInlierThreshold ( 0.0075f );
        //                go.setRadiusClutter ( 0.035f );
        //                go.setDetectClutter ( false ); //Attention, detect clutter turned off!
        //                go.setRegularizer ( 3.f );
        //                go.setClutterRegularizer ( 3.f );
        //                go.setHypPenalty ( 0.f );
        //                go.setIgnoreColor ( false );
        //                go.setColorSigma ( 0.25f );
        //                go.setOptimizerType ( opt_type_ );
        //                go.setObjectIds ( ids );
        //                //go.setSceneCloud (big_cloud);
        //                go.setSceneCloud ( big_cloud_after_mv );
        //                go.addModels ( aligned_models, true );
        //                go.verify ();
        //                std::vector<bool> mask;
        //                go.getMask ( mask );

        //                if ( visualize_output_ )
        //                {
        //                    pcl::visualization::PCLVisualizer vis ( "registered cloud" );
        //                    int v1, v2, v3, v4;
        //                    vis.createViewPort ( 0, 0, 0.5, 0.5, v1 );
        //                    vis.createViewPort ( 0.5, 0, 1, 0.5, v2 );
        //                    vis.createViewPort ( 0, 0.5, 0.5, 1, v3 );
        //                    vis.createViewPort ( 0.5, 0.5, 1, 1, v4 );

        //                    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler ( big_cloud_after_mv );
        //                    vis.addPointCloud ( big_cloud_after_mv, handler, "big", v1 );

        //                    /*pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler (big_cloud_vx_after_mv);
        //                    vis.addPointCloud (big_cloud_vx_after_mv, handler, "big", v1);*/

        //                    for ( size_t i=0; i < aligned_models.size(); i++ )
        //                    {
        //                        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler_rgb_verified ( aligned_models[i] );
        //                        std::stringstream name;
        //                        name << "Hypothesis_model_" << i;
        //                        vis.addPointCloud<pcl::PointXYZRGB> ( aligned_models[i], handler_rgb_verified, name.str (), v2 );
        //                    }

        //                    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr smooth_cloud_ =  go.getSmoothClustersRGBCloud();
        //                    if ( smooth_cloud_ )
        //                    {
        //                        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> random_handler ( smooth_cloud_ );
        //                        vis.addPointCloud<pcl::PointXYZRGBA> ( smooth_cloud_, random_handler, "smooth_cloud", v4 );
        //                    }

        //                    for ( size_t i = 0; i < mask.size (); i++ )
        //                    {
        //                        if ( mask[i] )
        //                        {
        //                            std::cout << "Verified:" << ids[i] << std::endl;
        //                            pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler_rgb_verified ( aligned_models[i] );
        //                            std::stringstream name;
        //                            name << "verified" << i;
        //                            vis.addPointCloud<pcl::PointXYZRGB> ( aligned_models[i], handler_rgb_verified, name.str (), v3 );
        //                        }
        //                    }

        //                    vis.spin ();
        //                }

        //                for ( size_t i = 0; i < mask.size (); i++ )
        //                {
        //                    if ( mask[i] )
        //                    {
        //                        int k=0;
        //                        for ( vp = vertices ( grph_final_ ); vp.first != vp.second; ++vp.first, k++ )
        //                        {
        //                            //hypotheses_poses_in_global_frame[i] transforms from object coordinates to global coordinate system
        //                            //transforms_to_global aligns a single frame to the global coordinate system
        //                            //transformation would then be a transformation transforming first the object to global coordinate system
        //                            //concatenated with the inverse of transforms_to_global[k]
        //                            Eigen::Matrix4f t = transforms_to_global[k].inverse() * hypotheses_poses_in_global_frame[i];
        //                            std::string origin = "3d go";
        //                            Hypothesis hyp ( ids[i], t, origin, true, true );
        //                            grph_final_[*vp.first].hypothesis.push_back ( hyp );
        //                        }
        //                    }
        //                }
        //            }
        //        }
        //        else
        {
            //---Verify-extended-hypotheses-and-visualize------------------------
            bool current_iteration_done = false;

            std::pair<vertex_iter, vertex_iter> vp;
            for ( vp = vertices ( grph_final_ ); (vp.first != vp.second) && (!current_iteration_done); ++vp.first )
            {

                Vertex vrtx_tmp;

                if ( ! do_reverse_hyp_extension)	// It only applies ICP on the latest scene point cloud
                {
                    vrtx_tmp = vrtx_final;
                    current_iteration_done = true;
                }
                else
                {
                    vrtx_tmp = *vp.first;
                }

                std::vector<pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr> aligned_models;
                std::vector < std::string > ids;

                for ( std::vector<Hypothesis>::iterator it_hyp = grph_final_[vrtx_tmp].hypothesis.begin (); it_hyp != grph_final_[vrtx_tmp].hypothesis.end (); ++it_hyp )
                {
                    PointInTPtr pModelPCl ( new pcl::PointCloud<pcl::PointXYZRGB> );
                    PointInTPtr pModelPClTransformed ( new pcl::PointCloud<pcl::PointXYZRGB> );
                    PointInTPtr pModelPCl2 ( new pcl::PointCloud<pcl::PointXYZRGB> );
                    pcl::io::loadPCDFile ( it_hyp->model_id_, * ( pModelPCl ) );

                    pcl::transformPointCloud ( *pModelPCl, *pModelPClTransformed, it_hyp->transform_ );

                    pcl::VoxelGrid<pcl::PointXYZRGB> sor;
                    float leaf = 0.005f;
                    sor.setLeafSize ( leaf, leaf, leaf );
                    sor.setInputCloud ( pModelPClTransformed );
                    sor.filter ( *pModelPCl2 );

                    aligned_models.push_back ( pModelPCl2 );
                    ids.push_back ( it_hyp->model_id_ );
                }
                std::cout << "View " << grph_final_[vrtx_tmp].scene_filename_ << " has " << grph_final_[vrtx_tmp].hypothesis.size ()
                          << " hypothesis. " << std::endl;

                //initialize go
                float go_resolution_ = 0.005f;
                bool add_planes = true;
                float assembled_resolution = 0.003f;
                float color_sigma = 0.5f;

                boost::shared_ptr<faat_pcl::GlobalHypothesesVerification_1<PointT, PointT> > go (
                            new faat_pcl::GlobalHypothesesVerification_1<PointT,
                            PointT> );

                go->setSmoothSegParameters ( 0.1, 0.035, 0.005 );
                //go->setRadiusNormals(0.03f);
                go->setResolution ( go_resolution_ );
                go->setInlierThreshold ( 0.01 );
                go->setRadiusClutter ( 0.03f );
                go->setRegularizer ( 2 );
                go->setClutterRegularizer ( 5 );
                go->setDetectClutter ( true );
                go->setOcclusionThreshold ( 0.01f );
                go->setOptimizerType ( 0 );
                go->setUseReplaceMoves ( true );
                go->setRadiusNormals ( 0.02 );
                go->setRequiresNormals ( false );
                go->setInitialStatus ( false );
                go->setIgnoreColor ( false );
                go->setColorSigma ( color_sigma );
                go->setUseSuperVoxels ( false );


                //Multiplane segmentation
                faat_pcl::MultiPlaneSegmentation<PointT> mps;
                mps.setInputCloud ( grph_final_[vrtx_tmp].pScenePCl );
                mps.setMinPlaneInliers ( 1000 );
                mps.setResolution ( go_resolution_ );
                mps.setNormals ( grph_final_[vrtx_tmp].pSceneNormals );
                mps.setMergePlanes ( true );
                std::vector<faat_pcl::PlaneModel<PointT> > planes_found;
                mps.segment();
                planes_found = mps.getModels();

                if ( planes_found.size() == 0 && grph_final_[vrtx_tmp].pScenePCl->isOrganized() )
                {
                    PCL_WARN ( "No planes found, doing segmentation with standard method\n" );
                    mps.segment ( true );
                    planes_found = mps.getModels();
                }

                std::vector<pcl::PointIndices> indices;
                Eigen::Vector4f table_plane;
                doSegmentation<PointT> ( grph_final_[vrtx_tmp].pScenePCl, grph_final_[vrtx_tmp].pSceneNormals, indices, table_plane );

                std::vector<int> indices_above_plane;
                for ( int k = 0; k < grph_final_[vrtx_tmp].pScenePCl->points.size (); k++ )
                {
                    Eigen::Vector3f xyz_p = grph_final_[vrtx_tmp].pScenePCl->points[k].getVector3fMap ();
                    if ( !pcl_isfinite ( xyz_p[0] ) || !pcl_isfinite ( xyz_p[1] ) || !pcl_isfinite ( xyz_p[2] ) )
                        continue;

                    float val = xyz_p[0] * table_plane[0] + xyz_p[1] * table_plane[1] + xyz_p[2] * table_plane[2] + table_plane[3];
                    if ( val >= 0.01 )
                        indices_above_plane.push_back ( static_cast<int> ( k ) );
                }

                std::vector<std::string> model_ids;
                typename pcl::PointCloud<PointT>::Ptr occlusion_cloud ( new pcl::PointCloud<PointT> ( *grph_final_[vrtx_tmp].pScenePCl ) );
                go->setSceneCloud ( grph_final_[vrtx_tmp].pScenePCl );
                go->setNormalsForClutterTerm ( grph_final_[vrtx_tmp].pSceneNormals );
                go->setOcclusionCloud ( occlusion_cloud );
                //addModels
                go->addModels ( aligned_models, true );
                //append planar models
                if ( add_planes )
                {
                    go->addPlanarModels ( planes_found );
                    for ( size_t kk=0; kk < planes_found.size(); kk++ )
                    {
                        std::stringstream plane_id;
                        plane_id << "plane_" << kk;
                        model_ids.push_back ( plane_id.str() );
                    }
                }

                go->setObjectIds ( model_ids );
                //verify
                {
                    pcl::ScopeTime t ( "Go verify" );
                    go->verify ();
                }
                std::vector<bool> mask_hv;
                go->getMask ( mask_hv );


                for ( size_t hyp_id = 0; hyp_id < aligned_models.size(); hyp_id++ )
                {
                    std::cout << hyp_id << "is" << static_cast<int> ( mask_hv[hyp_id] ) << std::endl;
                    //std::cout << static_cast<int> (mask_hv[j]) << std::endl;
                    if ( !mask_hv[hyp_id] )
                    {
                        grph_final_[vrtx_tmp].hypothesis[hyp_id].verified_ = false;
                    }
                    else
                    {
                        grph_final_[vrtx_tmp].hypothesis[hyp_id].verified_ = true;
                    }

                }
            }
        }

        //        outputgraph ( grph_final_, "Final_with_Hypothesis_extension.dot" );

        if ( visualize_output_ ) //-------Visualize Scene Cloud--------------------
        {
            std::vector<int> viewportNr;
            vis_->removeAllPointClouds();
            viewportNr = visualization_framework ( vis_, num_vertices(grph_), 4 );

            std::pair<vertex_iter, vertex_iter> vp;
            int view_id = -1;
            for ( vp = vertices ( grph_final_ ); vp.first != vp.second; ++vp.first )
            {
                view_id++;
                pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler_rgb ( grph_final_[*vp.first].pScenePCl );
                std::stringstream cloud_name;
                cloud_name << "scene_cloud_" << grph_final_[*vp.first].scene_filename_;
                vis_->addPointCloud<pcl::PointXYZRGB> ( grph_final_[*vp.first].pScenePCl, handler_rgb, cloud_name.str (), viewportNr[view_id * 4 + 0] );

                for ( size_t hyp_id = 0; hyp_id < grph_final_[*vp.first].hypothesis.size(); hyp_id++ )
                {
                    //visualize models
                    std::string model_id = grph_final_[*vp.first].hypothesis[hyp_id].model_id_;
                    std::string origin = grph_final_[*vp.first].hypothesis[hyp_id].origin_;
                    Eigen::Matrix4f trans = grph_final_[*vp.first].hypothesis[hyp_id].transform_;

                    std::stringstream name;
                    name << cloud_name.str() << "___hypothesis_" << hyp_id << "___origin_" << origin;

                    // 		ModelTPtr m;

                    // 		models_source_->getModelById(model_id, m);
                    //
                    // 		ConstPointInTPtr model_cloud = m->getAssembled (0.003f);
                    typename pcl::PointCloud<PointT>::Ptr pModelPCl ( new pcl::PointCloud<PointT> );
                    typename pcl::PointCloud<PointT>::Ptr model_aligned ( new pcl::PointCloud<PointT> );

                    pcl::io::loadPCDFile ( model_id, *pModelPCl );
                    pcl::transformPointCloud ( *pModelPCl, *model_aligned, trans );

                    pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb_handler ( model_aligned );
                    vis_->addPointCloud<PointT> ( model_aligned, rgb_handler, name.str (), viewportNr[view_id * 4 +2] );

                    if ( grph_final_[*vp.first].hypothesis[hyp_id].origin_.compare ( grph_final_[*vp.first].scene_filename_ ) == 0 )	//--show-hypotheses-from-single-view
                    {
                        name << "__extended";
                        pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb_handler2 ( model_aligned );
                        vis_->addPointCloud<PointT> ( model_aligned, rgb_handler, name.str (), viewportNr[view_id * 4 + 1] );
                    }

                    if ( grph_final_[*vp.first].hypothesis[hyp_id].verified_ )	//--show-verified-extended-hypotheses
                    {
                        name << "__verified";
                        pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb_handler3 ( model_aligned );
                        vis_->addPointCloud<PointT> ( model_aligned, rgb_handler, name.str (), viewportNr[view_id * 4 + 3] );
                    }

                }
            }
            vis_->spin ();
            //vis->getInteractorStyle()->saveScreenshot ( "singleview.png" );
        }

        //---CLEAN-UP-GRAPH

        if(num_vertices(grph_) > 2)
        {
            Vertex vrtxToKill = getFurthestVertex(grph_);

            std::vector<Edge> edges_to_be_removed;
            typename graph_traits<Graph>::out_edge_iterator out_i, out_end;
            for ( tie ( out_i, out_end ) = out_edges ( vrtxToKill, grph_ ); out_i != out_end; ++out_i )
            {
                edges_to_be_removed.push_back(*out_i);
            }
            typename graph_traits<Graph>::in_edge_iterator in_i, in_end;
            for ( tie ( in_i, in_end ) = in_edges ( vrtxToKill, grph_ ); in_i != in_end; ++in_i )
            {
                edges_to_be_removed.push_back(*in_i);
            }


            for(size_t remover_id = 0; remover_id < edges_to_be_removed.size(); remover_id++)
            {
                remove_edge(edges_to_be_removed[remover_id], grph_);
            }

            remove_vertex(vrtxToKill, grph_);

            Vertex vrtxToKill_final = getFurthestVertex(grph_final_);

            std::vector<Edge> edges_to_be_removed_final;

            for ( tie ( out_i, out_end ) = out_edges ( vrtxToKill_final, grph_final_ ); out_i != out_end; ++out_i )
            {
                edges_to_be_removed_final.push_back(*out_i);
            }
            for ( tie ( in_i, in_end ) = in_edges ( vrtxToKill_final, grph_final_ ); in_i != in_end; ++in_i )
            {
                edges_to_be_removed_final.push_back(*in_i);
            }

            for(size_t remover_id = 0; remover_id < edges_to_be_removed_final.size(); remover_id++)
            {
                remove_edge(edges_to_be_removed_final[remover_id], grph_final_);
            }

            remove_vertex(vrtxToKill_final, grph_final_);
            outputgraph ( grph_final_, "final_after_deleting_old_vertex.dot" );
            outputgraph ( grph_, "grph_after_deleting_old_vertex.dot" );
        }
    }
    else
    {
        ROS_ERROR ( "Failed to call service" );
        return 1;
    }


    return ( 0 );
}

std::vector<int>
visualization_framework ( pcl::visualization::PCLVisualizer::Ptr vis, int number_of_views, int number_of_subwindows_per_view )
{
    std::vector<int> viewportNr ( number_of_views * number_of_subwindows_per_view, 0 );

    for ( size_t i = 0; i < number_of_views; i++ )
    {
        for ( size_t j = 0; j < number_of_subwindows_per_view; j++ )
        {
            vis->createViewPort ( float ( i ) / number_of_views, float ( j ) / number_of_subwindows_per_view, ( float ( i ) + 1.0 ) / number_of_views,
                                  float ( j + 1 ) / number_of_subwindows_per_view, viewportNr[number_of_subwindows_per_view * i + j] );

            vis->setBackgroundColor ( float ( j * ( ( i % 2 ) / 10.0 + 1 ) ) / number_of_subwindows_per_view,
                                      float ( j * ( ( i % 2 ) / 10.0 + 1 ) ) / number_of_subwindows_per_view,
                                      float ( j * ( ( i % 2 ) / 10.0 + 1 ) ) / number_of_subwindows_per_view, viewportNr[number_of_subwindows_per_view * i + j] );

            std::stringstream window_id;
            window_id << "(" << i << ", " << j << ")";
            vis->addText ( window_id.str (), 10, 10, window_id.str (), viewportNr[i * number_of_subwindows_per_view + j] );
        }
    }
    return viewportNr;
}


void multiviewGraph::init ( int argc, char **argv )
{
    ros::init ( argc, argv, "multiview_graph" );
    current_cloud_.reset ( new pcl::PointCloud<pcl::PointXYZRGB>() );
    std::string icp_iterations, mv_keypoints, opt_type;
    n_ = new ros::NodeHandle ( "~" );
    n_->getParam ( "topic", topic_ );
    n_->getParam ( "models_dir", models_dir_ );
    n_->getParam ( "training_dir", training_dir_ );
    n_->getParam ( "visualize_output", visualize_output_ );
    n_->getParam ( "go_3d", go_3d_ );
    n_->getParam ( "input_cloud_dir", input_cloud_dir_ );
    n_->getParam ( "gt_or_output_dir", gt_or_ouput_dir_ );
    n_->getParam ( "icp_iterations", icp_iter_ );
    n_->getParam ( "mv_keypoints", mv_keypoints_ );
    n_->getParam ( "opt_type", opt_type_ );
    n_->getParam ( "chop_z", chop_at_z_ );
    client_ = n_->serviceClient<recognition_srv_definitions::recognize> ( "/mp_recognition" );

    //load models for visualization
    models_source_.reset ( new faat_pcl::rec_3d_framework::ModelOnlySource<pcl::PointXYZRGBNormal, PointT> );
    models_source_->setPath ( models_dir_ );
    models_source_->setLoadViews ( false );
    models_source_->setLoadIntoMemory ( false );
    std::string training_dir = "not_needed";
    models_source_->generate ( training_dir );
    models_source_->createVoxelGridAndDistanceTransform ( icp_resolution_ );
    ROS_INFO ( "Models loaded from %s", models_dir_.c_str() );

    object_to_be_highlighted_ = "";

    if ( visualize_output_ ) //-------Visualize Scene Cloud--------------------
    {
        vis_.reset ( new pcl::visualization::PCLVisualizer ( "vis1" ) );
        vis_->setWindowName ( "Recognition from Multiple Views" );
    }
    
    bf::path obj_name_path = input_cloud_dir_;
    if ( bf::exists ( obj_name_path ) )
    {
        std::vector < std::string > files_intern;
        std::string start = "";
        std::string ext = std::string ( "pcd" );
        getFilesInDirect ( obj_name_path, start, files_intern, ext );
        long nrFiles = files_intern.size ();
        if ( nrFiles > 0 )
        {
            ROS_INFO ( "Start offline recognition of %ld PCD files...", nrFiles );
            for ( size_t filevec_id = 0; filevec_id < nrFiles; filevec_id++ )
            {
                pcl::io::loadPCDFile ( input_cloud_dir_ + files_intern[filevec_id], *current_cloud_ );
                recognize ( *current_cloud_ , files_intern[filevec_id] );
            }

        }
        else
        {
            ROS_INFO ( "There are no PCD files in directory %s...", input_cloud_dir_.c_str() );
        }
    }
    else
    {
        sub_pc_  = n_->subscribe ( topic_, 1, &multiviewGraph::kinectCallback, this );
        sub_joy_ = n_->subscribe ( "/teleop_joystick/action_buttons", 1, &multiviewGraph::joyCallback, this );
        ROS_INFO ( "Start online recognition of topic %s by pressing a button...", topic_.c_str() );
        ros::spin();
    }
}
