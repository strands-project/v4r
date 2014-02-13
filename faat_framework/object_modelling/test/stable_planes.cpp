/*
 * GO3D.cpp
 *
 *  Created on: Oct 24, 2013
 *      Author: aitor
 */

#include <faat_pcl/utils/filesystem_utils.h>
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <pcl/common/transforms.h>
#include <fstream>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/common/angles.h>
#include <pcl/registration/icp.h>

#include <boost/graph/connected_components.hpp>
#include <boost/graph/copy.hpp>
#include <boost/graph/biconnected_components.hpp>
#include <boost/graph/prim_minimum_spanning_tree.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_matrix.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/undirected_graph.hpp>

#include <faat_pcl/registration/icp_with_gc.h>

namespace bf = boost::filesystem;

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

template<typename PointInT>
inline void
getIndicesFromCloud(typename pcl::PointCloud<PointInT>::Ptr & processed,
                      typename pcl::PointCloud<PointInT>::Ptr & keypoints_pointcloud,
                      std::vector<int> & indices)
{
  pcl::octree::OctreePointCloudSearch<PointInT> octree (0.005);
  octree.setInputCloud (processed);
  octree.addPointsFromInputCloud ();

  std::vector<int> pointIdxNKNSearch;
  std::vector<float> pointNKNSquaredDistance;

  for(size_t j=0; j < keypoints_pointcloud->points.size(); j++)
  {
   if (octree.nearestKSearch (keypoints_pointcloud->points[j], 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
   {
     indices.push_back(pointIdxNKNSearch[0]);
   }
  }
}

typedef pcl::PointXYZRGBNormal ModelPointT;

class
stablePlane
{
    public:
        Eigen::Vector3f normal_;
        float area_;
        std::vector<int> polygon_indices_;
};

bool checkForGround(Eigen::Vector3f & p, Eigen::Vector3f & n)
{
    if(p[2] < 0.002 && std::abs(n.dot(Eigen::Vector3f::UnitZ())) > 0.95 )
        return true;

    return false;
}

bool checkShareVertex(std::vector<uint32_t> & vert1, std::vector<uint32_t> & vert2)
{
    //check if they share at least a point
    bool share = false;
    for(size_t k=0; k < vert1.size(); k++)
    {
        for(size_t j=0; j < vert2.size(); j++)
        {
            if(vert1[k] == vert2[j])
            {
                share = true;
                break;
            }
        }
    }

    return share;
}

void mergeTriangles(pcl::PolygonMesh::Ptr & mesh_out,
                    pcl::PointCloud<ModelPointT>::Ptr & model_cloud,
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
        pcl::PointCloud<ModelPointT>::Ptr triangle_points(new pcl::PointCloud<ModelPointT>);
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

    pcl::visualization::PCLVisualizer vis ("merging triangles");
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
    vis.spin();
}

int
main (int argc, char ** argv)
{
    std::string model_path = "";
    std::string target_model_path = "";
    float overlap = 0.6f;
    float inliers_threshold = 0.01f;

    pcl::console::parse_argument (argc, argv, "-overlap", overlap);
    pcl::console::parse_argument (argc, argv, "-model_path", model_path);
    pcl::console::parse_argument (argc, argv, "-target_model", target_model_path);
    pcl::console::parse_argument (argc, argv, "-inliers_threshold", inliers_threshold);


    pcl::PointCloud<ModelPointT>::Ptr model_cloud(new pcl::PointCloud<ModelPointT>);
    pcl::io::loadPCDFile(model_path, *model_cloud);

    pcl::PointCloud<ModelPointT>::Ptr target_model(new pcl::PointCloud<ModelPointT>);
    pcl::io::loadPCDFile(target_model_path, *target_model);

    pcl::visualization::PCLVisualizer vis ("registered cloud");
    int v1, v2, v3, v4;
    vis.createViewPort (0, 0, 0.25, 1, v1);
    vis.createViewPort (0.25, 0, 0.5, 1, v4);
    vis.createViewPort (0.5, 0, 0.75, 1, v2);
    vis.createViewPort (0.75, 0, 1, 1, v3);
    vis.setBackgroundColor(1,1,1);
    vis.addCoordinateSystem(0.1f);

    pcl::visualization::PointCloudColorHandlerRGBField<ModelPointT> handler (model_cloud);
    vis.addPointCloud<ModelPointT> (model_cloud, handler, "big", v1);

    pcl::visualization::PointCloudColorHandlerRGBField<ModelPointT> handler_v4 (target_model);
    vis.addPointCloud<ModelPointT> (target_model, handler_v4, "big_target", v4);

    //vis.addPointCloudNormals<ModelPointT,ModelPointT> (model_cloud, model_cloud, 10, 0.01, "normals_big", v1);

    pcl::ConvexHull<ModelPointT> convex_hull;
    convex_hull.setInputCloud (model_cloud);
    convex_hull.setDimension (3);
    convex_hull.setComputeAreaVolume (false);

    pcl::PolygonMeshPtr mesh_out(new pcl::PolygonMesh);
    convex_hull.reconstruct (*mesh_out);

    vis.addPolygonMesh(*mesh_out, "hull", v2);

    {
        Eigen::Vector4f centroid;
        centroid.setZero();
        centroid[2] = -0.01f;
        pcl::demeanPointCloud(*target_model, centroid, *target_model);
        pcl::visualization::PointCloudColorHandlerRGBField<ModelPointT> handler (target_model);
        vis.addPointCloud<ModelPointT> (target_model, handler, "target", v3);
    }

    std::vector<stablePlane> stable_planes;
    mergeTriangles(mesh_out, model_cloud, stable_planes);
    std::cout << "Stable planes size:" << stable_planes.size() << std::endl;

    std::stable_sort(stable_planes.begin(), stable_planes.end(),
      boost::bind(&stablePlane::area_, _1) > boost::bind(&stablePlane::area_, _2)
    );

    int MAX_PLANES_ = 6;
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > initial_poses;
    float step = 30.f;

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
        pcl::PointCloud<ModelPointT>::Ptr model_cloud_trans(new pcl::PointCloud<ModelPointT>);
        pcl::transformPointCloudWithNormals(*model_cloud, *model_cloud_trans, transform);

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

        pcl::visualization::PointCloudColorHandlerRGBField<ModelPointT> handler (model_cloud_trans);
        vis.addPointCloud<ModelPointT> (model_cloud_trans, handler, "model_cloud_trans", v3);
        vis.spin();
        vis.removePointCloud("model_cloud_trans");

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
    icp.setMaximumIterations(50);
    icp.setOverlapPercentage(overlap);
    icp.setVisFinal(false);
    icp.setDtVxSize(0.002f);
    icp.setInitialPoses(initial_poses);
    icp.setUseRangeImages(false);
    icp.setInliersThreshold(inliers_threshold);

    pcl::registration::DefaultConvergenceCriteria<float>::Ptr convergence_criteria;
    convergence_criteria = icp.getConvergeCriteria ();
    convergence_criteria->setAbsoluteMSE (1e-12);
    convergence_criteria->setRelativeMSE(1e-12);
    convergence_criteria->setMaximumIterationsSimilarTransforms (50);
    convergence_criteria->setFailureAfterMaximumIterations (false);
    convergence_criteria->setTranslationThreshold (1e-15);
    convergence_criteria->setRotationThreshold (1.0 - 1e-15);

    icp.setInputTarget (target_model);
    icp.setInputSource (model_cloud);

    typename pcl::PointCloud<ModelPointT>::Ptr pp_out(new pcl::PointCloud<ModelPointT>);
    icp.align (*pp_out);
    std::vector<std::pair<float, Eigen::Matrix4f> > res;
    icp.getResults(res);

    for(size_t k=0; k < std::min((int)res.size(), 8); k++)
    {
        vis.removePointCloud("model_cloud_trans");

        std::cout << k << " " << res[k].first << std::endl;
        pcl::PointCloud<ModelPointT>::Ptr Final(new pcl::PointCloud<ModelPointT>);
        pcl::transformPointCloudWithNormals(*model_cloud, *Final, res[k].second);

        pcl::visualization::PointCloudColorHandlerRGBField<ModelPointT> handler (Final);
        vis.addPointCloud<ModelPointT> (Final, handler, "model_cloud_trans", v3);
        vis.spin();
    }

    pcl::PointCloud<ModelPointT>::Ptr merged_cloud(new pcl::PointCloud<ModelPointT>(*target_model));
    pcl::PointCloud<ModelPointT>::Ptr Final(new pcl::PointCloud<ModelPointT>);
    pcl::transformPointCloudWithNormals(*model_cloud, *Final, res[0].second);
    *merged_cloud += *Final;
    pcl::io::savePCDFileBinary("merged_cloud.pcd", *merged_cloud);

    /*pcl::PointCloud<pcl::PointXYZ> hull_cloud;
    pcl::fromPCLPointCloud2(mesh_out->cloud, hull_cloud);
    //pcl::copyPointCloud(mesh_out->cloud, hull_cloud);

    std::cout << "Number of polygons:" << mesh_out->polygons.size() << std::endl;

    for(size_t i=0; i < mesh_out->polygons.size(); i++)
    {
        Eigen::Vector3f v1, v2;
        v1 = hull_cloud.points[mesh_out->polygons[i].vertices[1]].getVector3fMap() - hull_cloud.points[mesh_out->polygons[i].vertices[0]].getVector3fMap();
        v2 = hull_cloud.points[mesh_out->polygons[i].vertices[2]].getVector3fMap() - hull_cloud.points[mesh_out->polygons[i].vertices[0]].getVector3fMap();
        Eigen::Vector3f cross_v1_v2 = v2.cross(v1);
        float area = 0.5f * cross_v1_v2.norm();
        //std::cout << "area: " << area << std::endl;
        if(area < 1e-4)
            continue;

        Eigen::Vector3f normal = cross_v1_v2;
        normal.normalize();

        //make sure that the normal obtained from the face is pointing in the right direction, otherwise flip it
        pcl::PointCloud<ModelPointT>::Ptr triangle_points(new pcl::PointCloud<ModelPointT>);
        for(size_t k=0; k < mesh_out->polygons[i].vertices.size(); k++)
        {
            ModelPointT p;
            p.getVector4fMap() = hull_cloud.points[mesh_out->polygons[i].vertices[k]].getVector4fMap();
            triangle_points->push_back(p);
        }

        std::vector<int> indices_to_cloud;
        getIndicesFromCloud<ModelPointT>(model_cloud, triangle_points, indices_to_cloud);
        for(size_t k=0; k < 1; k++)
        {
            Eigen::Vector3f normal_p = model_cloud->points[indices_to_cloud[k]].getNormalVector3fMap();
            if(normal_p.dot(normal) < 0)
            {
                //PCL_WARN("flip\n");
                normal = normal * -1.f;
            }
        }

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
        pcl::PointCloud<ModelPointT>::Ptr model_cloud_trans(new pcl::PointCloud<ModelPointT>);
        pcl::transformPointCloudWithNormals(*model_cloud, *model_cloud_trans, transform);

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

        for(size_t k=0; k < mesh_out->polygons[i].vertices.size(); k++)
        {
            std::stringstream name;
            name << "sphere_" << k;
            pcl::PointXYZ p;
            p.getVector4fMap() = hull_cloud.points[mesh_out->polygons[i].vertices[k]].getVector4fMap();
            p.getVector4fMap() = transform * p.getVector4fMap();
            vis.addSphere<pcl::PointXYZ>(p, 0.005f, name.str(), v3);
        }

        pcl::visualization::PointCloudColorHandlerRGBField<ModelPointT> handler (model_cloud_trans);
        vis.addPointCloud<ModelPointT> (model_cloud_trans, handler, "model_cloud_trans", v3);
        vis.spin();
        vis.removePointCloud("model_cloud_trans");

        float rotation = 0.f;
        float step = 30.f;
        pcl::PointCloud<ModelPointT>::Ptr model_cloud_trans_before_rotation(new pcl::PointCloud<ModelPointT>(*model_cloud_trans));

        while(rotation < 360.f)
        {
            Eigen::Affine3f rotation_z (Eigen::AngleAxisf (static_cast<float> (pcl::deg2rad(rotation)), Eigen::Vector3f::UnitZ()));
            pcl::transformPointCloudWithNormals(*model_cloud_trans_before_rotation, *model_cloud_trans, rotation_z);

            pcl::IterativeClosestPoint<ModelPointT,ModelPointT> icp;
            icp.setInputSource(model_cloud_trans);
            icp.setInputTarget(target_model);
            icp.setMaximumIterations(50);
            icp.setMaxCorrespondenceDistance(0.05f);
            icp.setEuclideanFitnessEpsilon(1e-15);
            icp.setTransformationEpsilon(1e-15);
            icp.setUseReciprocalCorrespondences(true);

            pcl::PointCloud<ModelPointT>::Ptr Final(new pcl::PointCloud<ModelPointT>);
            icp.align(*Final);
            std::cout << " converged:" << icp.hasConverged() << std::endl;

            pcl::visualization::PointCloudColorHandlerRGBField<ModelPointT> handler (Final);
            vis.addPointCloud<ModelPointT> (Final, handler, "model_cloud_trans", v3);
            vis.spin();
            vis.removePointCloud("model_cloud_trans");
            //vis.removeAllShapes(v3);
            rotation += step;
        }
    }
    vis.spin ();*/
}
