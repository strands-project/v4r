
#include "segmentation/checkerboardSegmentation.h"

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>

#include <opencv/cv.h>
#include <opencv2/core/eigen.hpp>
#include <faat_pcl/utils/pcl_opencv.h>

namespace object_modeller
{
namespace segmentation
{

CheckerboardSegmentation::CheckerboardSegmentation(std::vector<cv::Size> boardSizes)
{
    this->boardSizes = boardSizes;
}

std::vector<std::vector<int> > CheckerboardSegmentation::process(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> inputClouds)
{
    std::vector<Eigen::Matrix4f> poses;
    std::vector<std::vector<int> > indices;

    // point clouds -> checkerboard -> list of points
    std::vector<std::vector<std::vector<cv::Point2f> > > imagePoints;
    cv::Size imgSize;

    //find chessboard corners
    for (unsigned i=0; i<inputClouds.size(); i++)
    {
        cv::Mat_ < cv::Vec3b > colorImage;
        PCLOpenCV::ConvertPCLCloud2Image<pcl::PointXYZRGB> (inputClouds[i], colorImage);

        cv::Mat src_gray;
        cv::cvtColor( colorImage, src_gray, CV_RGB2GRAY );

        imgSize = src_gray.size();


        std::vector<std::vector<cv::Point2f> > pointCloudBoardPoints;

        for (unsigned k=0;k<boardSizes.size();k++)
        {
            std::vector<cv::Point2f> corners;
            bool patternFound = cv::findChessboardCorners(src_gray, boardSizes[k], corners);

            //TODO: imrpove corner accuracy

            //cv::drawChessboardCorners(src_gray, boardSizes[k], corners, patternFound);

            //cv::namedWindow("gray");
            //cv::imshow("gray", src_gray);
            //cv::waitKey(0);

            if (patternFound)
            {
                pointCloudBoardPoints.push_back(corners);
            }
            else
            {
                pointCloudBoardPoints.push_back(corners);
            }

            /*
            std::cout << "pattern found: " << patternFound << std::endl;
            for (unsigned i=0; i<corners.size(); i++)
            {
                std::cout << "Corner Point: " << corners[i] << std::endl;
            }
            */
        }

        imagePoints.push_back(pointCloudBoardPoints);








        /////////////segmentation

        /*
        Eigen::Vector4f table_plane = Eigen::Vector4f (model_coefficients[table_plane_selected].values[0], model_coefficients[table_plane_selected].values[1],
                                       model_coefficients[table_plane_selected].values[2], model_coefficients[table_plane_selected].values[3]);

        //cluster..
        typename pcl::EuclideanClusterComparator<PointT, pcl::Normal, pcl::Label>::Ptr
                euclidean_cluster_comparator_ (
                    new pcl::EuclideanClusterComparator<
                    PointT,
                    pcl::Normal,
                    pcl::Label> ());

        //create two labels, 1 one for points belonging to or under the plane, 1 for points above the plane
        label_indices.resize (2);

        for (int j = 0; j < xyz_points->points.size (); j++)
        {
            Eigen::Vector3f xyz_p = xyz_points->points[j].getVector3fMap ();

            if (!pcl_isfinite (xyz_p[0]) || !pcl_isfinite (xyz_p[1]) || !pcl_isfinite (xyz_p[2]))
                continue;

            float val = xyz_p[0] * table_plane[0] + xyz_p[1] * table_plane[1] + xyz_p[2] * table_plane[2] + table_plane[3];

            if (val >= min_height)
            {
                labels->points[j].label = 1;
                label_indices[0].indices.push_back (j);
            }
            else
            {
                labels->points[j].label = 0;
                label_indices[1].indices.push_back (j);
            }
        }

        std::vector<bool> plane_labels;
        plane_labels.resize (label_indices.size (), false);
        plane_labels[0] = true;

        euclidean_cluster_comparator_->setInputCloud (xyz_points);
        euclidean_cluster_comparator_->setLabels (labels);
        euclidean_cluster_comparator_->setExcludeLabels (plane_labels);
        euclidean_cluster_comparator_->setDistanceThreshold (0.035f, true);

        pcl::PointCloud<pcl::Label> euclidean_labels;
        std::vector<pcl::PointIndices> euclidean_label_indices;
        pcl::OrganizedConnectedComponentSegmentation<PointT, pcl::Label> euclidean_segmentation (euclidean_cluster_comparator_);
        euclidean_segmentation.setInputCloud (xyz_points);
        euclidean_segmentation.segment (euclidean_labels, euclidean_label_indices);

        for (size_t i = 0; i < euclidean_label_indices.size (); i++)
        {
            if (euclidean_label_indices[i].indices.size () > 100)
            {
                indices.push_back(euclidean_label_indices[i]);
            }
        }
        */
    }


    // segmentation

    return indices;

}

}
}
