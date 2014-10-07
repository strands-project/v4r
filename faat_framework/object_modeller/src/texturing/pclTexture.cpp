
#include "texturing/pclTexture.h"

#include <boost/lexical_cast.hpp>

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/conversions.h>
#include <pcl/io/io.h>

#include <pcl/search/kdtree.h>

#include <math.h>

#include <pcl/io/file_io.h>
#include <pcl/io/png_io.h>

#include <opencv2/core/core.hpp>

namespace object_modeller
{
namespace texturing
{

PclTexture::PclTexture(std::string config_name) : InOutModule(config_name)
{
    registerParameter("angleThreshDegree", "Angle Threshold", &angleThreshDegree, 20);
    registerParameter("projectNormals", "Project normals to camera plane", &projectNormals, true);
}

output::TexturedMesh::Ptr PclTexture::process(boost::tuples::tuple<pcl::PolygonMesh::Ptr, std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>, std::vector<Eigen::Matrix4f> > input)
{
    std::cout << "processing pcl texture" << std::endl;

    angleThresh = (angleThreshDegree * M_PI) / 180.0;

    pcl::PolygonMesh::Ptr mesh = boost::tuples::get<0>(input);
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clouds = boost::tuples::get<1>(input);
    std::vector<Eigen::Matrix4f> poses = boost::tuples::get<2>(input);

    std::cout << "got input" << std::endl;

    std::cout << "mesh size " << mesh->polygons.size() << std::endl;

    // contains for each polygon a list of images, that can see this polygon
    std::vector<int> visibleImages[mesh->polygons.size()];

    // contains for each polygon the image that should be used
    std::vector<int> targetImages;
    //std::vector<std::vector<int> > visibleImages;
    //visibleImages.resize(mesh->polygons.size());

    int imageCounter[clouds.size()];
    for (int i=0;i<clouds.size();i++)
    {
        imageCounter[i] = 0;
    }

    std::cout << "created vector" << std::endl;

    pcl::PointCloud<pcl::PointXYZ>::Ptr meshCloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2 (mesh->cloud, *meshCloud);

    std::cout << "poses size " << poses.size() << std::endl;

    for (int triangleIndex=0;triangleIndex<mesh->polygons.size();triangleIndex++)
    {
        //std::cout << "iterate triangles " << triangleIndex << " of " << mesh->polygons.size() << std::endl;
        double bestAngle = 100.0;
        int bestAngleIndex = -1;

        for (int imageIndex=0;imageIndex<clouds.size();imageIndex++)
        {
            //std::cout << "iterate images " << imageIndex << " of " << clouds.size() << std::endl;

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = clouds[imageIndex];

            pcl::Vertices vertices = mesh->polygons[triangleIndex];

            Eigen::Vector3f triangle_normal = calculateNormal(meshCloud, vertices, poses[imageIndex]);
            triangle_normal.normalize();


            //pcl::PointXYZ p1 = meshCloud->points[vertices.vertices[0]];
            //Eigen::Vector3f v1 = Eigen::Vector3f( p1.getArray3fMap() );
            Eigen::Vector3f cam_vector = Eigen::Vector3f(0.0f, 0.0f, -1.0f);//cloud->sensor_orientation_.normalized().vec();
            cam_vector.normalize();

            if (projectNormals)
            {
                Eigen::Vector3f proj_normal(0.0f, 1.0f, 0.0f);
                Eigen::Vector3f proj_origin(0.0f, 0.0f, 0.0f);

                Eigen::Hyperplane<float, 3> plane(proj_normal, proj_origin);

                triangle_normal = plane.projection(triangle_normal);
                triangle_normal.normalize();
            }

            double angle = cam_vector.dot(triangle_normal);

            angle = fabs(acos(angle));


            if (angle < bestAngle)
            {
                bestAngle = angle;
                bestAngleIndex = imageIndex;
            }

            //std::cout << "angle " << angle << " angle thresh " << angleThresh << std::endl;
            if (angle < angleThresh)
            {
                imageCounter[imageIndex]++;
                visibleImages[triangleIndex].push_back(imageIndex);
            }

        }

        //std::cout << "nr visible images " << visibleImages[triangleIndex].size() << "\n";

        //std::cout << "get best image" << std::endl;
        if (visibleImages[triangleIndex].size() == 0)
        {
            if (bestAngleIndex == -1)
            {
                std::cout << "WARNING !!!!!!!!!!!!!" << std::endl;
            }
            imageCounter[bestAngleIndex]++;
            visibleImages[triangleIndex].push_back(bestAngleIndex);
        }
    }

    std::cout << "finished angle calculations " << poses.size() << std::endl;

    //targetImages.resize(mesh->polygons.size());
    for (unsigned int triangleIndex=0;triangleIndex<mesh->polygons.size();triangleIndex++)
    {
        targetImages.push_back(-1);
    }

    //int forcedImage = 0;

    //hack

    for (int j=0;j<clouds.size();j++) // get the one that is used most
    {
        std::cout << "image counter " << imageCounter[j] << std::endl;
    }

    std::cout << "mesh polygon size " << mesh->polygons.size();

    for (int i=0;i<clouds.size();i++) // get best weights
    {
        int best = 0;
        int bestValue = 0;
        for (int j=0;j<clouds.size();j++) // get the one that is used most
        {
            if (imageCounter[j] > bestValue)
            {
                best = j;
                bestValue = imageCounter[j];
            }
        }

        //std::cout << "best image " << best << std::endl;

        if (imageCounter[best] != 0)
        {
            for (unsigned int triangleIndex=0;triangleIndex<mesh->polygons.size();triangleIndex++)
            {
                if(std::find(visibleImages[triangleIndex].begin(), visibleImages[triangleIndex].end(), best) != visibleImages[triangleIndex].end())
                {
                    //std::cout << "adding image for triangle " << triangleIndex << std::endl;
                    targetImages.at(triangleIndex) = best;

                    for (int j=0;j<visibleImages[triangleIndex].size();j++)
                    {
                        int index = visibleImages[triangleIndex][j];
                        imageCounter[index]--;
                    }
                    visibleImages[triangleIndex].clear();

                }
            }
        }
    }

    std::cout << "--------------------" << std::endl;

    for (int j=0;j<clouds.size();j++) // get the one that is used most
    {
        std::cout << "image counter " << imageCounter[j] << std::endl;
    }

    std::cout << "render textures" << std::endl;

    output::TexturedMesh::Ptr result(new output::TexturedMesh());
    result->mesh = mesh;
    result->textureCoordinates.resize(meshCloud->size());
    result->textureIndex.resize(mesh->polygons.size());
    result->textureIndex2.resize(meshCloud->size());

    std::cout << "setup coords" << std::endl;

    for (int i=0;i<meshCloud->size();i++)
    {
        result->textureCoordinates[i] = Eigen::Vector2f(0.0f, 0.0f);
        result->textureIndex2[i] = 0;
    }

    std::cout << "setup indices" << std::endl;

    for (int i=0;i<mesh->polygons.size();i++)
    {
        result->textureIndex[i] = 0;
    }

    std::cout << "create textures" << std::endl;
    std::cout << "cloud size " << clouds.size() << std::endl;
    std::cout << "poses size " << poses.size() << std::endl;

    // create textures
    for (int imageIndex=0;imageIndex<clouds.size() ;imageIndex++)
    {
        addTexture(meshCloud, result, clouds[imageIndex], poses[imageIndex], imageIndex, targetImages);
    }

    int totalWidth = 0;
    int totalHeight = 0;
    for (int imageIndex=0;imageIndex<clouds.size() ;imageIndex++)
    {
        totalWidth += result->textures[imageIndex].cols;
        totalHeight = std::max(totalHeight, result->textures[imageIndex].rows);
    }

    std::cout << "create new texture" << std::endl;
    cv::Mat3b texture(totalHeight, totalWidth, CV_8UC3);
    for (int x=0;x<totalWidth;x++)
    {
        for (int y=0;y<totalHeight;y++)
        {
            texture.at<cv::Vec3b>(y, x).val[0] = 0;
            texture.at<cv::Vec3b>(y, x).val[1] = 0;
            texture.at<cv::Vec3b>(y, x).val[2] = 0;
        }
    }

    int offsetX = 0;
    for (int imageIndex=0;imageIndex<clouds.size() ;imageIndex++)
    {
        std::cout << "copy texture " << imageIndex << std::endl;
        // copy texture
        for (int x=0;x<result->textures[imageIndex].cols;x++)
        {
            for (int y=0;y<result->textures[imageIndex].rows;y++)
            {
                texture.at<cv::Vec3b>(y, x + offsetX).val[0] = result->textures[imageIndex].at<cv::Vec3b>(y, x).val[0];
                texture.at<cv::Vec3b>(y, x + offsetX).val[1] = result->textures[imageIndex].at<cv::Vec3b>(y, x).val[1];
                texture.at<cv::Vec3b>(y, x + offsetX).val[2] = result->textures[imageIndex].at<cv::Vec3b>(y, x).val[2];
            }
        }

        std::cout << "adapt coords " << imageIndex << std::endl;

        for (int j=0;j<meshCloud->points.size();j++)
        {
            int textureIndex = result->textureIndex2[j];
            Eigen::Vector2f coords = result->textureCoordinates[j];

            if (textureIndex < imageIndex)
            {
                int oldWidth = offsetX;
                int newWidth = offsetX + result->textures[imageIndex].cols;

                coords.x() = (coords.x() * oldWidth) / newWidth;
            }

            if (textureIndex == imageIndex)
            {
                int oldWidth = result->textures[imageIndex].cols;
                int newWidth = offsetX + result->textures[imageIndex].cols;

                coords.x() = ((coords.x() * oldWidth) + offsetX) / newWidth;
                coords.y() = (coords.y() * result->textures[imageIndex].rows) / totalHeight;
            }

            result->textureCoordinates[j] = coords;
        }

        offsetX += result->textures[imageIndex].cols;
    }

    result->textures.clear();
    result->textures.push_back(texture);

    std::cout << "complete" << std::endl;

    return result;
}

void PclTexture::addTexture(pcl::PointCloud<pcl::PointXYZ>::Ptr meshCloud, output::TexturedMesh::Ptr result, pcl::PointCloud<pcl::PointXYZRGB>::Ptr image, Eigen::Matrix4f pose, int imageIndex, std::vector<int> bestImages)
{
    std::cout << "create texture" << std::endl;

    pcl::RangeImagePlanar rangeImage = getMeshProjection(meshCloud, image, pose);

    int min_x = image->width;
    int min_y = image->height;
    int max_x = 0;
    int max_y = 0;

    for (int i=0;i<image->width;i++)
    {
        for (int j=0;j<image->height;j++)
        {
            if (!rangeImage.isInImage(i, j) || !rangeImage.isValid(i, j))
            {
                pcl::PointXYZRGB &p = image->at(i, j);
                p.r = 0;
                p.g = 0;
                p.b = 0;
            } else {
                if (i < min_x) min_x = i;
                if (j < min_y) min_y = j;
                if (i > max_x) max_x = i;
                if (j > max_y) max_y = j;
            }
        }
    }

    max_x += (max_x - min_x) % 8;
    max_y += (max_y - min_y) % 8;

    //min_x = 0;
    //max_x = image->width;
    //min_y = 0;
    //max_y = image->height;


    std::cout << "create texture " << min_x << " " << max_x << " " << min_y << " " << max_y << std::endl;

    cv::Mat3b texture(max_y - min_y, max_x - min_x, CV_8UC3);
    cv::Mat3b texture_filtered(max_y - min_y, max_x - min_x, CV_8UC3);

    for (int i=min_x;i<max_x;i++)
    {
        for (int j=min_y;j<max_y;j++)
        {
            texture.at<cv::Vec3b>(j - min_y, i - min_x).val[0] = image->at(i, j).r;
            texture.at<cv::Vec3b>(j - min_y, i - min_x).val[1] = image->at(i, j).g;
            texture.at<cv::Vec3b>(j - min_y, i - min_x).val[2] = image->at(i, j).b;
        }
    }

    cv::bilateralFilter(texture, texture_filtered, -1, 20.0, 20.0);

    result->textures.push_back(texture_filtered);

    std::cout << "calc texture coords" << std::endl;


    // calculate texture coordinates

    int size = result->mesh->polygons.size();
    for (int triangleIndex=0;triangleIndex<size;triangleIndex++)
    {
        pcl::Vertices vertices = result->mesh->polygons[triangleIndex];

        for (int vert_i=0;vert_i<vertices.vertices.size();vert_i++)
        {
            // multiple texture coordinates for one vertex?
            int targetImage = bestImages[triangleIndex];

            //std::cout << "best image: " << targetImage << std::endl;

            if (targetImage == imageIndex)
            {
                //std::cout << "adding coordinates" << std::endl;
                pcl::PointXYZ pclPoint = meshCloud->points[vertices.vertices[vert_i]];
                Eigen::Vector3f p(pclPoint.x, pclPoint.y, pclPoint.z);
                float u,v,r;

                rangeImage.getImagePoint(p, u, v, r);
                u -= min_x;
                v -= min_y;

                if (result->textureCoordinates[vertices.vertices[vert_i]].x() != 0.0f || result->textureCoordinates[vertices.vertices[vert_i]].y() != 0.0f)
                {
                    //std::cout << "clouds size " << meshCloud->size();
                    // duplicate point
                    meshCloud->push_back(pclPoint);
                    result->textureCoordinates.push_back(Eigen::Vector2f(Eigen::Vector2f(u / texture.cols, v / texture.rows)));
                    result->textureIndex2.push_back(targetImage);
                    unsigned int newIndex = meshCloud->size() - 1;
                    //std::cout << "new index " << newIndex << " tex coord size " << result->textureCoordinates.size() << std::endl;
                    result->mesh->polygons[triangleIndex].vertices[vert_i] = newIndex;
                }
                else
                {
                    result->textureCoordinates[vertices.vertices[vert_i]] = Eigen::Vector2f(u / texture.cols, v / texture.rows);

                    result->textureIndex2[vertices.vertices[vert_i]] = targetImage;
                }

                result->textureIndex[triangleIndex] = targetImage;
            }
        }
    }

    pcl::toPCLPointCloud2(*meshCloud, result->mesh->cloud);

    /*
    std::string file_name;
    file_name.append("/home/alex/Arbeitsflaeche/test/texture_");
    file_name.append(boost::lexical_cast<std::string>(imageIndex));
    file_name.append(".png");
    pcl::io::savePNGFile(file_name, *image, "rgb");
    */
}

pcl::RangeImagePlanar PclTexture::getMeshProjection(pcl::PointCloud<pcl::PointXYZ>::Ptr meshCloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, Eigen::Matrix4f pose)
{

    // Image size. Both Kinect and Xtion work at 640x480.
    int imageSizeX = 640;
    int imageSizeY = 480;
    // Center of projection. here, we choose the middle of the image.
    float centerX = 640.0f / 2.0f;
    float centerY = 480.0f / 2.0f;
    // Focal length. The value seen here has been taken from the original depth images.
    // It is safe to use the same value vertically and horizontally.
    float focalLengthX = 525.0f, focalLengthY = focalLengthX;
    // Sensor pose. Thankfully, the cloud includes the data.
    Eigen::Affine3f sensorPose = Eigen::Affine3f(Eigen::Translation3f(cloud->sensor_origin_[0],
                                 cloud->sensor_origin_[1],
                                 cloud->sensor_origin_[2])) *
                                 Eigen::Affine3f(cloud->sensor_orientation_);

    sensorPose = sensorPose * pose;
    // Noise level. If greater than 0, values of neighboring points will be averaged.
    // This would set the search radius (i.e., 0.03 == 3cm).
    float noiseLevel = 0.0f;
    // Minimum range. If set, any point closer to the sensor than this will be ignored.
    float minimumRange = 0.0f;

    // Planar range image object.
    pcl::RangeImagePlanar rangeImagePlanar;
    rangeImagePlanar.createFromPointCloudWithFixedSize(*meshCloud, imageSizeX, imageSizeY,
            centerX, centerY, focalLengthX, focalLengthX,
            sensorPose, pcl::RangeImage::CAMERA_FRAME,
            noiseLevel, minimumRange);

    return rangeImagePlanar;
}

Eigen::Vector3f PclTexture::calculateNormal(pcl::PointCloud<pcl::PointXYZ>::Ptr meshCloud, pcl::Vertices vertices, Eigen::Matrix4f pose)
{
    /*
    pcl::PointXYZRGBNormal p1 = meshCloud->points[vertices.vertices[0]];
    pcl::PointXYZRGBNormal p2 = meshCloud->points[vertices.vertices[1]];
    pcl::PointXYZRGBNormal p3 = meshCloud->points[vertices.vertices[2]];

    Eigen::Vector3f v1 = Eigen::Vector3f( p1.getNormalVector3fMap() );
    Eigen::Vector3f v2 = Eigen::Vector3f( p2.getNormalVector3fMap() );
    Eigen::Vector3f v3 = Eigen::Vector3f( p3.getNormalVector3fMap() );



    return (v1 + v2 + v3).normalized();
    */

    pcl::PointXYZ p1 = meshCloud->points[vertices.vertices[0]];
    pcl::PointXYZ p2 = meshCloud->points[vertices.vertices[1]];
    pcl::PointXYZ p3 = meshCloud->points[vertices.vertices[2]];

    Eigen::Matrix4f inv = pose.inverse();

    p1 = pcl::transformPoint(p1, Eigen::Affine3f(inv));
    p2 = pcl::transformPoint(p2, Eigen::Affine3f(inv));
    p3 = pcl::transformPoint(p3, Eigen::Affine3f(inv));

    Eigen::Vector3f v1 = Eigen::Vector3f( p1.getArray3fMap() );
    Eigen::Vector3f v2 = Eigen::Vector3f( p2.getArray3fMap() );
    Eigen::Vector3f v3 = Eigen::Vector3f( p3.getArray3fMap() );

    v2 = v1 - v2;
    v3 = v1 - v3;
    v2.normalize();
    v3.normalize();

    return v2.cross(v3);
}

}
}





/*

#include "texturing/pclTexture.h"

#include <boost/lexical_cast.hpp>

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/conversions.h>
#include <pcl/io/io.h>

#include <pcl/search/kdtree.h>

#include <math.h>

#include <pcl/io/file_io.h>
#include <pcl/io/png_io.h>

#include <opencv2/core/core.hpp>

namespace object_modeller
{
namespace texturing
{

PclTexture::PclTexture(std::string config_name) : InOutModule(config_name)
{
    angleThresh = M_PI;
}

output::TexturedMesh::Ptr PclTexture::process(boost::tuples::tuple<pcl::PolygonMesh::Ptr, std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>, std::vector<Eigen::Matrix4f> > input)
{
    std::cout << "processing pcl texture" << std::endl;

    pcl::PolygonMesh::Ptr mesh = boost::tuples::get<0>(input);
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clouds = boost::tuples::get<1>(input);
    std::vector<Eigen::Matrix4f> poses = boost::tuples::get<2>(input);

    std::cout << "got input" << std::endl;

    std::cout << "mesh size " << mesh->polygons.size() << std::endl;

    // contains for each polygon a list of images, that can see this polygon
    std::vector<int> visibleImages[mesh->polygons.size()];
    //std::vector<std::vector<int> > visibleImages;
    //visibleImages.resize(mesh->polygons.size());

    std::cout << "created vector" << std::endl;

    pcl::PointCloud<pcl::PointXYZ>::Ptr meshCloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2 (mesh->cloud, *meshCloud);

    std::cout << "copied cloud" << std::endl;

    for (int triangleIndex=0;triangleIndex<mesh->polygons.size();triangleIndex++)
    {
        //std::cout << "iterate triangles " << triangleIndex << " of " << mesh->polygons.size() << std::endl;
        double bestAngle = 100.0;
        int bestAngleIndex;

        for (int imageIndex=0;imageIndex<clouds.size();imageIndex++)
        {
            //std::cout << "iterate images " << imageIndex << " of " << clouds.size() << std::endl;

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = clouds[imageIndex];

            pcl::Vertices vertices = mesh->polygons[triangleIndex];

            Eigen::Vector3f triangle_normal = calculateNormal(meshCloud, vertices);

            Eigen::Vector3f cam_vector = cloud->sensor_orientation_.normalized().vec();

            double angle = cam_vector.dot(triangle_normal);

            if (angle < bestAngle)
            {
                bestAngle = angle;
                bestAngleIndex = imageIndex;
            }

            if (angle < angleThresh)
            {
                // take the best at the moment
                //visibleImages[triangleIndex].push_back(imageIndex);
            }

            //double test = cloud->sensor_orientation_.dot(normal);

        }

        if (visibleImages[triangleIndex].size() == 0)
        {
            visibleImages[triangleIndex].push_back(bestAngleIndex);
        }
    }

    std::cout << "render textures" << std::endl;

    output::TexturedMesh::Ptr result(new output::TexturedMesh());
    result->mesh = mesh;
    result->textureCoordinates.resize(mesh->cloud.data.size());
    result->textureIndex.resize(mesh->polygons.size());

    std::cout << "setup coords" << std::endl;

    for (int i=0;i<mesh->cloud.data.size();i++)
    {
        result->textureCoordinates[i] = Eigen::Vector2f(0.0f, 0.0f);
    }

    std::cout << "setup indices" << std::endl;

    for (int i=0;i<mesh->polygons.size();i++)
    {
        result->textureIndex[i] = 0;
    }

    std::cout << "create textures" << std::endl;
    std::cout << "cloud size " << clouds.size() << std::endl;
    std::cout << "poses size " << poses.size() << std::endl;

    // create textures
    for (int imageIndex=0;imageIndex<1;imageIndex++)
    {
        addTexture(result, clouds[imageIndex], poses[imageIndex], imageIndex, visibleImages);
    }

    std::cout << "complete" << std::endl;

    return result;
}

void PclTexture::addTexture(output::TexturedMesh::Ptr result, pcl::PointCloud<pcl::PointXYZRGB>::Ptr image, Eigen::Matrix4f pose, int imageIndex, std::vector<int> *bestImages)
{
    std::cout << "create texture" << std::endl;

    pcl::PointCloud<pcl::PointXYZ>::Ptr meshCloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2 (result->mesh->cloud, *meshCloud);

    pcl::RangeImagePlanar rangeImage = getMeshProjection(meshCloud, image, pose);

    int min_x = image->width;
    int min_y = image->height;
    int max_x = 0;
    int max_y = 0;

    for (int i=0;i<image->width;i++)
    {
        for (int j=0;j<image->height;j++)
        {
            if (!rangeImage.isInImage(i, j) || !rangeImage.isValid(i, j))
            {
                pcl::PointXYZRGB &p = image->at(i, j);
                p.r = 0;
                p.g = 0;
                p.b = 0;
            } else {
                if (i < min_x) min_x = i;
                if (j < min_y) min_y = j;
                if (i > max_x) max_x = i;
                if (j > max_y) max_y = j;
            }
        }
    }

    max_x += (max_x - min_x) % 8;
    max_y += (max_y - min_y) % 8;

    //min_x = 0;
    //max_x = image->width;
    //min_y = 0;
    //max_y = image->height;


    std::cout << "create texture " << min_x << " " << max_x << " " << min_y << " " << max_y << std::endl;

    cv::Mat3b texture(max_y - min_y, max_x - min_x, CV_8UC3);

    for (int i=min_x;i<max_x;i++)
    {
        for (int j=min_y;j<max_y;j++)
        {
            texture.at<cv::Vec3b>(j - min_y, i - min_x).val[0] = image->at(i, j).r;
            texture.at<cv::Vec3b>(j - min_y, i - min_x).val[1] = image->at(i, j).g;
            texture.at<cv::Vec3b>(j - min_y, i - min_x).val[2] = image->at(i, j).b;
        }
    }


    result->textures.push_back(texture);


    std::cout << "calc texture coords" << std::endl;


    // calculate texture coordinates

    for (int triangleIndex=0;triangleIndex<result->mesh->polygons.size();triangleIndex++)
    {
        pcl::Vertices vertices = result->mesh->polygons[triangleIndex];

        for (int vert_i=0;vert_i<vertices.vertices.size();vert_i++)
        {
            // multiple texture coordinates for one vertex?
            int targetImage = bestImages[triangleIndex][0];

            if (targetImage == imageIndex)
            {
                pcl::PointXYZ pclPoint = meshCloud->points[vertices.vertices[vert_i]];
                Eigen::Vector3f p(pclPoint.x, pclPoint.y, pclPoint.z);
                float u,v,r;

                rangeImage.getImagePoint(p, u, v, r);
                u -= min_x;
                v -= min_y;

                //std::cout << "image point " << u << " - " << v << std::endl;

                result->textureCoordinates[vertices.vertices[vert_i]] = Eigen::Vector2f(u / texture.cols, v / texture.rows);

                //std::cout << "coords " << "\n" << result->textureCoordinates[vertices.vertices[vert_i]] << std::endl;

                result->textureIndex[triangleIndex] = targetImage;
            }
        }
    }
}

pcl::RangeImagePlanar PclTexture::getMeshProjection(pcl::PointCloud<pcl::PointXYZ>::Ptr meshCloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, Eigen::Matrix4f pose)
{

    // Image size. Both Kinect and Xtion work at 640x480.
    int imageSizeX = 640;
    int imageSizeY = 480;
    // Center of projection. here, we choose the middle of the image.
    float centerX = 640.0f / 2.0f;
    float centerY = 480.0f / 2.0f;
    // Focal length. The value seen here has been taken from the original depth images.
    // It is safe to use the same value vertically and horizontally.
    float focalLengthX = 525.0f, focalLengthY = focalLengthX;
    // Sensor pose. Thankfully, the cloud includes the data.
    Eigen::Affine3f sensorPose = Eigen::Affine3f(Eigen::Translation3f(cloud->sensor_origin_[0],
                                 cloud->sensor_origin_[1],
                                 cloud->sensor_origin_[2])) *
                                 Eigen::Affine3f(cloud->sensor_orientation_);

    sensorPose = sensorPose * pose;
    // Noise level. If greater than 0, values of neighboring points will be averaged.
    // This would set the search radius (i.e., 0.03 == 3cm).
    float noiseLevel = 0.0f;
    // Minimum range. If set, any point closer to the sensor than this will be ignored.
    float minimumRange = 0.0f;

    // Planar range image object.
    pcl::RangeImagePlanar rangeImagePlanar;
    rangeImagePlanar.createFromPointCloudWithFixedSize(*meshCloud, imageSizeX, imageSizeY,
            centerX, centerY, focalLengthX, focalLengthX,
            sensorPose, pcl::RangeImage::CAMERA_FRAME,
            noiseLevel, minimumRange);

    return rangeImagePlanar;
}

Eigen::Vector3f PclTexture::calculateNormal(pcl::PointCloud<pcl::PointXYZ>::Ptr meshCloud, pcl::Vertices vertices)
{
    pcl::PointXYZ p1 = meshCloud->points[vertices.vertices[0]];
    pcl::PointXYZ p2 = meshCloud->points[vertices.vertices[1]];
    pcl::PointXYZ p3 = meshCloud->points[vertices.vertices[2]];

    Eigen::Vector3f v1 = Eigen::Vector3f( p1.getArray3fMap() );
    Eigen::Vector3f v2 = Eigen::Vector3f( p2.getArray3fMap() );
    Eigen::Vector3f v3 = Eigen::Vector3f( p3.getArray3fMap() );

    v2 = v2 - v1;
    v3 = v3 - v1;

    return v2.cross(v3);
}

}
}



  */
