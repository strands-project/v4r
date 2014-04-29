#include "registration/checkerboard.h"


#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/transformation_estimation_svd.h>

#include <opencv/cv.h>
#include <opencv2/core/eigen.hpp>
#include <faat_pcl/utils/pcl_opencv.h>

#include "v4r/KeypointTools/invPose.hpp"

namespace object_modeller
{
namespace registration
{

void CheckerboardRegistration::applyConfig(Config &config)
{
    this->boardSizes = config.getCvSizeList("checkerboardRegistration.boardSizes");
}

std::vector<Eigen::Matrix4f> CheckerboardRegistration::process(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> inputClouds)
{
    std::vector<Eigen::Matrix4f> poses;

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
    }


    // calculate poses

    pcl::registration::TransformationEstimationSVD<pcl::PointXYZRGB, pcl::PointXYZRGB, float> transformation;

    poses.push_back(Eigen::Matrix4f::Identity());

    Eigen::Matrix4f accum = Eigen::Matrix4f::Identity();

    for (unsigned i=1;i<imagePoints.size();i++) // iterate over point clouds
    {
        pcl::PointCloud<pcl::PointXYZRGB> src;
        pcl::PointCloud<pcl::PointXYZRGB> dest;
        Eigen::Matrix4f pose;

        // find checkerboard to use
        int checkerboardIndex = 0;

        for (unsigned k=0;k<imagePoints[i-1].size();k++)
        {
            if (imagePoints[i-1][k].size() > 0 && imagePoints[i-1][k].size() == imagePoints[i][k].size())
            {
                checkerboardIndex = k;
                break;
            }
        }

        std::cout << "Using checkerboard index " << checkerboardIndex << std::endl;


        for (unsigned k=0;k<imagePoints[i-1][checkerboardIndex].size();k++)
        {
            cv::Point2f at_src(round(imagePoints[i-1][checkerboardIndex][k].x), round(imagePoints[i-1][checkerboardIndex][k].y));
            pcl::PointXYZRGB p_src = inputClouds[i-1]->at(at_src.x, at_src.y);

            cv::Point2f at_dest(round(imagePoints[i][checkerboardIndex][k].x), round(imagePoints[i][checkerboardIndex][k].y));
            pcl::PointXYZRGB p_dest = inputClouds[i]->at(at_dest.x, at_dest.y);

            if (pcl::isFinite(p_src) && pcl::isFinite(p_dest))
            {
                src.push_back(p_src);
                dest.push_back(p_dest);
            }
        }

        transformation.estimateRigidTransformation(src, dest, pose);

        Eigen::Matrix4f inv_pose;
        kp::invPose(pose, inv_pose);
        accum = inv_pose * accum;
        poses.push_back(accum);
    }

    return poses;
}

}
}
