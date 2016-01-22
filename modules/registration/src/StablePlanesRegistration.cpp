#include "v4r/registration/StablePlanesRegistration.h"
#include <boost/graph/connected_components.hpp>
#include <boost/graph/copy.hpp>
#include <boost/graph/prim_minimum_spanning_tree.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_matrix.hpp>

#include <pcl/registration/icp.h>
#include <pcl/surface/convex_hull.h>
#include <v4r/common/noise_models.h>
#include <v4r/registration/noise_model_based_cloud_integration.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/angles.h>

template<class PointT>
v4r::Registration::StablePlanesRegistration<PointT>::StablePlanesRegistration()
{
    name_ = "StablePlanesRegistration";
}

template<class PointT>
void
v4r::Registration::StablePlanesRegistration<PointT>::mergeTriangles(pcl::PolygonMesh::Ptr & mesh_out,
                                                                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr & model_cloud,
                                                                    std::vector<stablePlane> & stable_planes)
{

    pcl::PointCloud<pcl::PointXYZ> hull_cloud;
    pcl::fromPCLPointCloud2(mesh_out->cloud, hull_cloud);

    std::cout << "Number of polygons:" << mesh_out->polygons.size() << std::endl;

    std::vector<Eigen::Vector3f> normals;
    std::vector<float> areas;
    std::vector<Eigen::Vector3f> centers;

    normals.resize(mesh_out->polygons.size());
    areas.resize(mesh_out->polygons.size());
    centers.resize(mesh_out->polygons.size());

    Eigen::Vector3f centroid;
    int n_points = 0;
    for(size_t i=0; i < mesh_out->polygons.size(); i++, n_points++)
    {
        Eigen::Vector3f center;
        center.setZero();
        for(size_t k=0; k < mesh_out->polygons[i].vertices.size(); k++)
            center = center + hull_cloud.points[mesh_out->polygons[i].vertices[k]].getVector3fMap();

        center = center / 3.f;
        centroid += center;
    }

    centroid /= static_cast<float>(n_points);

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
        //all normals looking outwards from the convex hull centroid
        Eigen::Vector3f direction = center - centroid;
        direction.normalize();
        if(normal.dot(direction) < 0)
            normal = normal * -1.f;

        normals[i] = normal;
    }

    float good_dot = 0.98f;
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

            /*if(checkForGround(centers[j], normals[j]))
                continue;*/

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
        sp.normal_.normalize();
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

    pcl::PointXYZ centroid_sphere;
    centroid_sphere.getVector3fMap() = centroid;
    vis.addSphere<pcl::PointXYZ>(centroid_sphere, 0.01, "sphere");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal> handler (normal_center_cloud, 255, 0 ,0);
    vis.addPointCloud<pcl::PointNormal> (normal_center_cloud, handler, "center_cloud");
    vis.addPointCloudNormals<pcl::PointNormal,pcl::PointNormal> (normal_center_cloud, normal_center_cloud, 1, 0.01, "normal_center_cloud");
    vis.spin();*/

}


template<class PointT>
void
v4r::Registration::StablePlanesRegistration<PointT>::initialize(std::vector<std::pair<int, int> > & session_ranges)
{
    //for each partial model, get a clean model and compute stable planes
    stable_planes_.resize(session_ranges.size());
    partial_models_with_normals_.resize(session_ranges.size());

    for(size_t i=0; i < session_ranges.size(); i++)
    {

        int clouds_session = session_ranges[i].second - session_ranges[i].first + 1;
        std::vector<std::vector<std::vector<float> > > pt_properties (clouds_session);
        std::vector<typename pcl::PointCloud<PointT>::Ptr> clouds(clouds_session);
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > poses(clouds_session);
        std::vector<std::vector<int> > indices(clouds_session);
        std::vector< pcl::PointCloud<pcl::Normal>::Ptr > normals(clouds_session);

        int k=0;
        for(int t=session_ranges[i].first; t <= session_ranges[i].second; t++, k++)
        {
            std::cout << k << " " << t << " " << clouds.size() << std::endl;
            clouds[k] = this->getCloud(t);
            poses[k] = this->getPose(t);
            normals[k] = this->getNormal(t);
            indices[k] = this->getIndices(t);

            v4r::NguyenNoiseModel<PointT> nm;
            nm.setInputCloud(clouds[k]);
            nm.setInputNormals(normals[k]);
            nm.compute();
            pt_properties[k] = nm.getPointProperties();
        }

        typename pcl::PointCloud<PointT>::Ptr octree_cloud(new pcl::PointCloud<PointT>);
        pcl::PointCloud<pcl::Normal>::Ptr big_normals(new pcl::PointCloud<pcl::Normal>);
        v4r::NMBasedCloudIntegration<pcl::PointXYZRGB>::Parameter nmparam;
        nmparam.octree_resolution_ = 0.005f;
        nmparam.min_points_per_voxel_ = 1;
        v4r::NMBasedCloudIntegration<pcl::PointXYZRGB> nmIntegration (nmparam);
        nmIntegration.setInputClouds(clouds);
        nmIntegration.setTransformations(poses);
        nmIntegration.setInputNormals(normals);
        nmIntegration.setIndices(indices);
        nmIntegration.setPointProperties(pt_properties);
        nmIntegration.compute(octree_cloud);
        nmIntegration.getOutputNormals(big_normals);

        partial_models_with_normals_[i].reset (new pcl::PointCloud<pcl::PointXYZRGBNormal>());
        pcl::concatenateFields(*big_normals, *octree_cloud, *partial_models_with_normals_[i]);

        pcl::ConvexHull<pcl::PointXYZRGBNormal> convex_hull;
        convex_hull.setInputCloud (partial_models_with_normals_[i]);
        convex_hull.setDimension (3);
        convex_hull.setComputeAreaVolume (false);

        pcl::PolygonMeshPtr mesh_out(new pcl::PolygonMesh);
        convex_hull.reconstruct (*mesh_out);

        mergeTriangles(mesh_out, partial_models_with_normals_[i], stable_planes_[i]);
        std::cout << "Stable planes size:" << stable_planes_[i].size() << std::endl;

        std::stable_sort(stable_planes_[i].begin(), stable_planes_[i].end(),
          boost::bind(&stablePlane::area_, _1) > boost::bind(&stablePlane::area_, _2)
        );
    }
}

template<class PointT>
void
v4r::Registration::StablePlanesRegistration<PointT>::compute(int s1, int s2)
{
    //test stable planes and do ICP, return as many poses as necessary
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > initial_poses;
    float step = 30;
    typedef pcl::PointXYZRGBNormal ModelPointT;

    int MAX_PLANES_ = 4;

//#define VIS_PLANES

#ifdef VIS_PLANES
    pcl::visualization::PCLVisualizer vis_s2("s2 on stable planes...");
    int v1,v2;
    vis_s2.createViewPort(0,0,0.5,1,v1);
    vis_s2.createViewPort(0.5,0,1,1,v2);
#endif

    //transform partial_models_with_normals_[s1] to be on computed_planes_[s1]
    typename pcl::PointCloud<ModelPointT>::Ptr target_model_on_plane(new pcl::PointCloud<ModelPointT>(*partial_models_with_normals_[s1]));
    Eigen::Matrix4f target_transform = Eigen::Matrix4f::Identity();

#ifdef VIS_PLANES
    pcl::visualization::PointCloudColorHandlerRGBField<ModelPointT> handler(target_model_on_plane);
    vis_s2.addPointCloud(target_model_on_plane, handler, "target", v1);
    vis_s2.addCoordinateSystem(0.1);
#endif

    for(size_t i=0; i < std::min(MAX_PLANES_, (int)stable_planes_[s2].size()); i++)
    {
        std::cout << stable_planes_[s2][i].area_ << std::endl;
        Eigen::Vector3f normal = stable_planes_[s2][i].normal_;
        Eigen::Matrix4f transform;
        transform.setIdentity();

        std::cout << normal << " norm:" << normal.norm() << std::endl;

        //ATTENTION: Make sure that the determinant is non-negative (meaning that we have an invertible rotation matrix, otherwise weird flips...)
        transform.block<3,1>(0,2) = normal * -1.f;

        if(std::abs(Eigen::Vector3f::UnitZ().dot(transform.block<3,1>(0,2))) < 0.9f)
        {
            transform.block<3,1>(0,1) = Eigen::Vector3f::UnitZ().cross(transform.block<3,1>(0,2));
        }
        else
        {
            transform.block<3,1>(0,1) = Eigen::Vector3f::UnitY().cross(transform.block<3,1>(0,2));
        }

        transform.block<3,1>(0,0) = transform.block<3,1>(0,1).cross(transform.block<3,1>(0,2));
        transform.block<3,1>(0,1).normalize();
        transform.block<3,1>(0,0).normalize();

        assert(transform.determinant() > 0);

        transform = transform.inverse().eval();
        typename pcl::PointCloud<ModelPointT>::Ptr model_cloud_trans(new pcl::PointCloud<ModelPointT>);
        pcl::transformPointCloudWithNormals(*partial_models_with_normals_[s2], *model_cloud_trans, transform);

        Eigen::Vector4f min_pt, max_pt;
        pcl::getMinMax3D(*model_cloud_trans, min_pt, max_pt);

        Eigen::Matrix4f translation;
        translation.setIdentity();
        translation(2,3) = -min_pt[2]; //- 0.01f; //maybe, using min point is not good
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

#ifdef VIS_PLANES
        {
            pcl::transformPointCloudWithNormals(*partial_models_with_normals_[s2], *model_cloud_trans, transform);

            pcl::visualization::PointCloudColorHandlerRGBField<ModelPointT> handler(model_cloud_trans);
            vis_s2.addPointCloud(model_cloud_trans, handler, "source_v2", v2);
            vis_s2.spin();
        }
#endif

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

#ifdef VIS_PLANES
            {
                pcl::transformPointCloudWithNormals(*partial_models_with_normals_[s2], *model_cloud_trans, pose);

                pcl::visualization::PointCloudColorHandlerRGBField<ModelPointT> handler(model_cloud_trans);
                vis_s2.addPointCloud(model_cloud_trans, handler, "source", v1);
                vis_s2.spin();
                vis_s2.removePointCloud("source");
            }
#endif

        }

#ifdef VIS_PLANES
        vis_s2.removePointCloud("source_v2");
#endif
    }

    /*{
        Eigen::Vector3f normal = computed_planes_[s1];
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
        pcl::transformPointCloudWithNormals(*partial_models_with_normals_[s1], *model_cloud_trans, transform);

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
        pcl::transformPointCloudWithNormals(*model_cloud_trans, *target_model_on_plane, center_transform);

        transform = center_transform * translation * transform;
        target_transform = transform;
    }*/

    poses_.clear();

    {
        /*pcl::visualization::PCLVisualizer vis("initial alignment on plane");
        pcl::visualization::PointCloudColorHandlerRGBField<ModelPointT> handler(target_model_on_plane);
        vis.addPointCloud(target_model_on_plane, handler, "target");
        vis.addCoordinateSystem(0.1);*/

        int steps_rotation = 360.f / step;
        std::cout << "steps rotation:" << steps_rotation << std::endl;

        std::cout << "Going to do ICP with " << initial_poses.size() << " poses:" << std::endl;
        poses_.resize(initial_poses.size());


#pragma omp parallel for schedule(dynamic, 1) num_threads(4)
        for(size_t i=0; i < initial_poses.size(); i++)
        {
            pcl::IterativeClosestPoint<ModelPointT, ModelPointT> icp;
            icp.setInputSource(partial_models_with_normals_[s2]);
            icp.setInputTarget(target_model_on_plane);
            icp.setMaxCorrespondenceDistance(0.01f);
            icp.setUseReciprocalCorrespondences(false);
            icp.setMaximumIterations(50);

            pcl::PointCloud<ModelPointT> out_cloud;
            icp.align(out_cloud, initial_poses[i]);
            Eigen::Matrix4f output = icp.getFinalTransformation();

            //output transform from s2 to s1, however the coordinate system of both have been changed
            Eigen::Matrix4f total_trans_s2_to_s1 = target_transform.inverse() * output;
            poses_[i] = total_trans_s2_to_s1;

            std::cout << "Done " << i << std::endl;

            if(static_cast<int>(i) % steps_rotation != 0)
                continue;

            /*typename pcl::PointCloud<ModelPointT>::Ptr model_cloud_trans(new pcl::PointCloud<ModelPointT>);
            pcl::transformPointCloudWithNormals(*partial_models_with_normals_[s2], *model_cloud_trans, output);

            pcl::visualization::PointCloudColorHandlerRGBField<ModelPointT> handler(model_cloud_trans);
            vis.addPointCloud(model_cloud_trans, handler, "source");
            vis.spin();
            vis.removePointCloud("source");*/
        }

    }
}

template class V4R_EXPORTS v4r::Registration::StablePlanesRegistration<pcl::PointXYZRGB>;
