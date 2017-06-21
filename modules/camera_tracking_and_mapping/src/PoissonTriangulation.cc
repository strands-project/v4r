/**
 * $Id$
 * 
 * Software License Agreement (GNU General Public License)
 *
 *  Copyright (C) 2015:
 *
 *    Johann Prankl, prankl@acin.tuwien.ac.at
 *    Aitor Aldoma, aldoma@acin.tuwien.ac.at
 *
 *      Automation and Control Institute
 *      Vienna University of Technology
 *      Gusshausstraße 25-29
 *      1170 Vienn, Austria
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * @author Johann Prankl, Aitor Aldoma
 *
 */


/**
  TODO: CLEAN UP!!!!!!!!!!!!!!!!!!!!!!!
  */

#include <v4r/camera_tracking_and_mapping/PoissonTriangulation.hh>


#include <pcl/surface/convex_hull.h>
#include <pcl/filters/crop_hull.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>

namespace v4r
{

PoissonTriangulation::PoissonTriangulation(int depth, int samplesPerNode, bool cropModel) : depth(depth), samplesPerNode(samplesPerNode), cropModel(cropModel)
{
}

void PoissonTriangulation::reconstruct(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud, pcl::PolygonMesh &mesh)
{
   pcl::Poisson<pcl::PointXYZRGBNormal> poisson;
    poisson.setDepth(depth);
    poisson.setSamplesPerNode(samplesPerNode);
    poisson.setInputCloud(cloud);

    poisson.reconstruct(mesh);


    if (cropModel)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_model (new pcl::PointCloud<pcl::PointXYZ>);

        pcl::copyPointCloud(*cloud, *cloud_model);

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull (new pcl::PointCloud<pcl::PointXYZ>);
        std::vector<pcl::Vertices> polygons;
        pcl::ConvexHull<pcl::PointXYZ> chull;
        chull.setInputCloud (cloud_model);
        chull.reconstruct (*cloud_hull, polygons);

        std::cout << "convex hull" << std::endl;


        pcl::PointCloud<pcl::PointXYZ>::Ptr meshCloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromPCLPointCloud2 (mesh.cloud, *meshCloud);

        pcl::CropHull<pcl::PointXYZ> cropHull;
        cropHull.setCropOutside(true);
        cropHull.setDim(chull.getDimension());
        cropHull.setHullCloud(cloud_hull);
        cropHull.setHullIndices(polygons);
        cropHull.setNegative(true);
        cropHull.setInputCloud(meshCloud);

        std::vector<int> outliers;
        cropHull.filter(outliers);

        std::cout << "outlier size: " << outliers.size() << " total size: " << meshCloud->size() << std::endl;

        std::cout << "filter convex hull" << std::endl;

        std::vector<int> inv_outliers;
        for (int i=0;i<(int)meshCloud->size();i++) {
            bool keep = true;

            for (int j=0;j<(int)outliers.size();j++) {

                if (outliers[j] == i) {
                    keep = false;
                }
            }

            if (keep) {
                inv_outliers.push_back(i);
            }
        }

        outliers = inv_outliers;

        std::cout << "outlier size: " << outliers.size() << " total size: " << meshCloud->size() << std::endl;

        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*cloud_hull, centroid);


        // project outliers
        for (int i=0;i<(int)outliers.size();i++) {
            pcl::PointXYZ point = meshCloud->at(outliers[i]);

            Eigen::Vector3f p(point.x, point.y, point.z);
            Eigen::Vector3f ray(centroid[0] - p[0], centroid[1] - p[1], centroid[2] - p[2]);

            float best_r = 10000.0f;

            for (int j=0;j<(int)polygons.size();j++)
            {
                int vertex = polygons.at(j).vertices[0];
                Eigen::Vector3f a(cloud_hull->at(vertex).x, cloud_hull->at(vertex).y, cloud_hull->at(vertex).z);
                vertex = polygons.at(j).vertices[1];
                Eigen::Vector3f b(cloud_hull->at(vertex).x, cloud_hull->at(vertex).y, cloud_hull->at(vertex).z);
                vertex = polygons.at(j).vertices[2];
                Eigen::Vector3f c(cloud_hull->at(vertex).x, cloud_hull->at(vertex).y, cloud_hull->at(vertex).z);

                const Eigen::Vector3f u = b - a;
                const Eigen::Vector3f v = c - a;
                const Eigen::Vector3f n = u.cross (v);
                const float n_dot_ray = n.dot (ray);
                const float r = n.dot (a - p) / n_dot_ray;

                if (r < 0)
                    continue;

                const Eigen::Vector3f w = p + r * ray - a;
                const float denominator = u.dot (v) * u.dot (v) - u.dot (u) * v.dot (v);
                const float s_numerator = u.dot (v) * w.dot (v) - v.dot (v) * w.dot (u);
                const float s = s_numerator / denominator;

                if (s < 0 || s > 1)
                    continue;

                const float t_numerator = u.dot (v) * w.dot (u) - u.dot (u) * w.dot (v);
                const float t = t_numerator / denominator;

                if (t < 0 || s+t > 1)
                    continue;

                //p = w + a;
                if (r < best_r) {
                    best_r = r;
                }
            }

            p = p + best_r * ray;

            point.x = p[0];
            point.y = p[1];
            point.z = p[2];

            meshCloud->at(outliers[i]) = point;
        }

        pcl::toPCLPointCloud2(*meshCloud, mesh.cloud);
    }
}

}
