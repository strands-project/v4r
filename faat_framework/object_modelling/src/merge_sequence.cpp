#include <faat_pcl/object_modelling/merge_sequence.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/copy.hpp>
#include <boost/graph/prim_minimum_spanning_tree.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_matrix.hpp>

#include <faat_pcl/registration/icp_with_gc.h>
#include <pcl/surface/convex_hull.h>

template<typename ModelPointT>
void
faat_pcl::modelling::MergeSequences<ModelPointT>::mergeTriangles(pcl::PolygonMesh::Ptr & mesh_out,
                                                                  PointTPtr & model_cloud,
                                                                  std::vector<stablePlane> & stable_planes)
{
    pcl::PointCloud<pcl::PointXYZ> hull_cloud;
    pcl::fromPCLPointCloud2(mesh_out->cloud, hull_cloud);
    //pcl::copyPointCloud(mesh_out->cloud, hull_cloud);

    std::cout << "Number of polygons:" << mesh_out->polygons.size() << std::endl;

    std::vector<Eigen::Vector3f> normals;
    std::vector<float> areas;
    std::vector<Eigen::Vector3f> centers;

    normals.resize(mesh_out->polygons.size());
    areas.resize(mesh_out->polygons.size());
    centers.resize(mesh_out->polygons.size());

    for(size_t i=0; i < mesh_out->polygons.size(); i++)
    {
        Eigen::Vector3f v1, v2;
        v1 = hull_cloud.points[mesh_out->polygons[i].vertices[1]].getVector3fMap() - hull_cloud.points[mesh_out->polygons[i].vertices[0]].getVector3fMap();
        v2 = hull_cloud.points[mesh_out->polygons[i].vertices[2]].getVector3fMap() - hull_cloud.points[mesh_out->polygons[i].vertices[0]].getVector3fMap();
        Eigen::Vector3f cross_v1_v2 = v2.cross(v1);
        float area = 0.5f * cross_v1_v2.norm();
        areas[i] = area;

        Eigen::Vector3f center;
        center.setZero();
        for(size_t k=0; k < mesh_out->polygons[i].vertices.size(); k++)
            center = center + hull_cloud.points[mesh_out->polygons[i].vertices[k]].getVector3fMap();

        center = center / 3.f;
        centers[i] = center;

        Eigen::Vector3f normal = cross_v1_v2;
        normal.normalize();

        //make sure that the normal obtained from the face is pointing in the right direction, otherwise flip it
        PointTPtr triangle_points(new pcl::PointCloud<ModelPointT>);
        ModelPointT p;
        p.getVector4fMap() = hull_cloud.points[mesh_out->polygons[i].vertices[0]].getVector4fMap();
        triangle_points->push_back(p);

        std::vector<int> indices_to_cloud;
        getIndicesFromCloud<ModelPointT>(model_cloud, triangle_points, indices_to_cloud);
        Eigen::Vector3f normal_p = model_cloud->points[indices_to_cloud[0]].getNormalVector3fMap();
        if(normal_p.dot(normal) < 0)
            normal = normal * -1.f;

        normals[i] = normal;
    }

    float good_dot = 0.995f;
    int good_normals_pairs=0;
    float min_area = 1e-4;
    std::vector<std::pair<int,int> > lines;

    for(size_t i=0; i < normals.size(); i++)
    {
        if(areas[i] < min_area)
            continue;

        if(checkForGround(centers[i], normals[i]))
            continue;

        for(size_t j=(i+1); j < normals.size(); j++)
        {
            if(areas[j] < min_area)
                continue;

            if(checkForGround(centers[j], normals[j]))
                continue;

            float dot = normals[i].dot(normals[j]);
            if(dot < good_dot)
                continue;

            if(!checkShareVertex(mesh_out->polygons[i].vertices,mesh_out->polygons[j].vertices))
                continue;

            good_normals_pairs++;
            lines.push_back(std::make_pair((int)i,(int)j));
        }
    }

    std::cout << "good_normals_pairs: " << good_normals_pairs << std::endl;

    //Build graph (good_normals_pairs edges, #vertices?)
    typedef boost::adjacency_matrix<boost::undirectedS, int> Graph;
    Graph G(mesh_out->polygons.size());
    for(size_t i=0; i < lines.size(); i++)
    {
        boost::add_edge(lines[i].first, lines[i].second, G);
    }

    std::vector<int> components (boost::num_vertices (G));
    int n_cc = static_cast<int> (boost::connected_components (G, &components[0]));
    std::cout << "Number of connected components..." << n_cc << std::endl;

    std::vector< std::vector<int> > unique_vertices_per_cc;
    std::vector<int> cc_sizes;
    std::vector<float> cc_areas;
    cc_sizes.resize (n_cc, 0);
    cc_areas.resize (n_cc, 0);
    unique_vertices_per_cc.resize (n_cc);

    typename boost::graph_traits<Graph>::vertex_iterator vertexIt, vertexEnd;
    boost::tie (vertexIt, vertexEnd) = vertices (G);
    for (; vertexIt != vertexEnd; ++vertexIt)
    {
      int c = components[*vertexIt];
      unique_vertices_per_cc[c].push_back(*vertexIt);
      cc_sizes[c]++;
      cc_areas[c]+=areas[*vertexIt];
    }

    for(size_t i=0; i < cc_sizes.size(); i++)
    {
        if(cc_areas[i] < min_area)
            continue;

        std::cout << "size:" << cc_sizes[i] << " area:" << cc_areas[i] << std::endl;
        stablePlane sp;
        sp.area_ = cc_areas[i];
        sp.polygon_indices_ = unique_vertices_per_cc[i];

        //TODO: Weighted normal based on area!
        sp.normal_ = Eigen::Vector3f::Zero();
        for(size_t k=0; k < unique_vertices_per_cc[i].size(); k++)
        {
            sp.normal_ += normals[unique_vertices_per_cc[i][k]];
        }

        sp.normal_ = sp.normal_ / (int)(unique_vertices_per_cc[i].size());
        stable_planes.push_back(sp);
    }

    /*pcl::visualization::PCLVisualizer vis ("merging triangles");
    int v1, v2;
    vis.createViewPort (0, 0, 0.5, 1, v1);
    vis.createViewPort (0.5, 0, 1, 1, v2);
    vis.setBackgroundColor(0,0,0);
    vis.addCoordinateSystem(0.1f);
    vis.addPolygonMesh(*mesh_out, "hull", v1);

    for(size_t i=0; i < lines.size(); i++)
    {
        pcl::PointXYZ p1,p2;
        p1.getVector3fMap() = centers[lines[i].first];
        p2.getVector3fMap()  = centers[lines[i].second];

        std::stringstream name;
        name << "line_" << i;
        vis.addLine(p1,p2, name.str(), v2);
    }

    pcl::PointCloud<pcl::PointNormal>::Ptr normal_center_cloud(new pcl::PointCloud<pcl::PointNormal>);
    for(size_t i=0; i < centers.size(); i++)
    {

        if(areas[i] < min_area)
            continue;

        if(checkForGround(centers[i], normals[i]))
            continue;

        pcl::PointNormal p;
        p.getVector3fMap() = centers[i];
        p.getNormalVector3fMap() = normals[i];
        normal_center_cloud->push_back(p);
    }

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal> handler (normal_center_cloud, 255, 0 ,0);
    vis.addPointCloud<pcl::PointNormal> (normal_center_cloud, handler, "center_cloud");
    vis.addPointCloudNormals<pcl::PointNormal,pcl::PointNormal> (normal_center_cloud, normal_center_cloud, 1, 0.01, "normal_center_cloud");
    vis.spin();*/
}

template<typename ModelPointT>
void
faat_pcl::modelling::MergeSequences<ModelPointT>::compute(std::vector<std::pair<float, Eigen::Matrix4f> > & res)
{
    pcl::ConvexHull<ModelPointT> convex_hull;
    convex_hull.setInputCloud (model_cloud_);
    convex_hull.setDimension (3);
    convex_hull.setComputeAreaVolume (false);

    pcl::PolygonMeshPtr mesh_out(new pcl::PolygonMesh);
    convex_hull.reconstruct (*mesh_out);

    std::vector<stablePlane> stable_planes;
    mergeTriangles(mesh_out, model_cloud_, stable_planes);
    std::cout << "Stable planes size:" << stable_planes.size() << std::endl;

    std::stable_sort(stable_planes.begin(), stable_planes.end(),
      boost::bind(&stablePlane::area_, _1) > boost::bind(&stablePlane::area_, _2)
    );

    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > initial_poses;
    float step = angular_step_;

    for(size_t i=0; i < std::min(MAX_PLANES_, (int)stable_planes.size()); i++)
    {
        std::cout << stable_planes[i].area_ << std::endl;
        Eigen::Vector3f normal = stable_planes[i].normal_;
        Eigen::Matrix4f transform;
        transform.setIdentity();

        //ATTENTION: Make sure that the determinant is non-negative (meaning that we have an invertible rotation matrix, otherwise weird flips...)
        transform.block<3,1>(0,2) = normal * -1.f;
        transform.block<3,1>(0,1) = Eigen::Vector3f::UnitZ().cross(transform.block<3,1>(0,2));
        transform.block<3,1>(0,0) = transform.block<3,1>(0,1).cross(transform.block<3,1>(0,2));
        transform.block<3,1>(0,1).normalize();
        transform.block<3,1>(0,0).normalize();

        assert(transform.determinant() > 0);

        transform = transform.inverse().eval();
        typename pcl::PointCloud<ModelPointT>::Ptr model_cloud_trans(new pcl::PointCloud<ModelPointT>);
        pcl::transformPointCloudWithNormals(*model_cloud_, *model_cloud_trans, transform);

        Eigen::Vector4f min_pt, max_pt;
        pcl::getMinMax3D(*model_cloud_trans, min_pt, max_pt);

        Eigen::Matrix4f translation;
        translation.setIdentity();
        translation(2,3) = -min_pt[2];
        pcl::transformPointCloudWithNormals(*model_cloud_trans, *model_cloud_trans, translation);

        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*model_cloud_trans, centroid);

        Eigen::Matrix4f center_transform;
        center_transform.setIdentity();
        center_transform(0,3) = -centroid[0];
        center_transform(1,3) = -centroid[1];
        center_transform(2,3) = 0;
        pcl::transformPointCloudWithNormals(*model_cloud_trans, *model_cloud_trans, center_transform);

        transform = center_transform * translation * transform;

        float rotation = 0.f;
        while(rotation < 360.f)
        {
            Eigen::AngleAxisf rotation_z = Eigen::AngleAxisf (static_cast<float> (pcl::deg2rad(rotation)), Eigen::Vector3f::UnitZ());
            Eigen::Matrix4f pose;
            pose.setIdentity();
            pose.block<3,3>(0,0) = rotation_z.toRotationMatrix();
            pose = pose * transform;
            initial_poses.push_back(pose);
            rotation += step;
        }
    }

    std::cout << "initial poses:" << initial_poses.size() << std::endl;
    //do ICP WITH GC using all initial poses
    faat_pcl::IterativeClosestPointWithGC<ModelPointT, ModelPointT> icp;
    icp.setTransformationEpsilon (0.000001 * 0.000001);
    icp.setMinNumCorrespondences (5);
    icp.setMaxCorrespondenceDistance (0.03);
    icp.setUseCG (true);
    icp.setSurvivalOfTheFittest (false);
    icp.setMaximumIterations(max_iterations_);
    icp.setOverlapPercentage(overlap_);
    icp.setVisFinal(false);
    icp.setDtVxSize(0.002f);
    icp.setInitialPoses(initial_poses);
    icp.setUseRangeImages(false);
    icp.setInliersThreshold(inlier_threshold_);
    icp.setuseColor(use_color_);

    pcl::registration::DefaultConvergenceCriteria<float>::Ptr convergence_criteria;
    convergence_criteria = icp.getConvergeCriteria ();
    convergence_criteria->setAbsoluteMSE (1e-12);
    convergence_criteria->setRelativeMSE(1e-12);
    convergence_criteria->setMaximumIterationsSimilarTransforms (50);
    convergence_criteria->setFailureAfterMaximumIterations (false);
    convergence_criteria->setTranslationThreshold (1e-15);
    convergence_criteria->setRotationThreshold (1.0 - 1e-15);

    icp.setInputTarget (target_model_);
    icp.setInputSource (model_cloud_);

    typename pcl::PointCloud<ModelPointT>::Ptr pp_out(new pcl::PointCloud<ModelPointT>);
    icp.align (*pp_out);
    icp.getResults(res);
}

template class faat_pcl::modelling::MergeSequences<pcl::PointXYZRGBNormal>;

