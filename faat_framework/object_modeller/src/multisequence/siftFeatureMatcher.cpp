#include "multisequence/siftFeatureMatcher.h"

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>

#include <pcl/features/integral_image_normal.h>
#include <pcl/features/normal_3d_omp.h>

#include <faat_pcl/utils/noise_models.h>
#include <faat_pcl/utils/noise_model_based_cloud_integration.h>

#include <faat_pcl/3d_rec_framework/feature_wrapper/local/image/sift_local_estimator.h>

#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/common/transformation_from_correspondences.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/ia_ransac.h>

#include <limits.h>

#include <pcl/kdtree/impl/kdtree_flann.hpp>

#include <pcl/registration/icp.h>

#include "v4r/KeypointTools/invPose.hpp"


namespace object_modeller
{
namespace multisequence
{

SiftFeatureMatcher::SiftFeatureMatcher(std::string config_name) : InOutModule(config_name)
{
}

Eigen::Matrix4f SiftFeatureMatcher::process(boost::tuples::tuple<std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>, std::vector<Eigen::Matrix4f>, std::vector<std::vector<int> >, std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> > input)
{
    Eigen::Matrix4f result = Eigen::Matrix4f::Identity();

    inputSequences.push_back(boost::tuples::get<0>(input));
    poses.push_back(boost::tuples::get<1>(input));
    indices.push_back(boost::tuples::get<2>(input));
    models.push_back(boost::tuples::get<3>(input)[0]);
    std::cout << "add sequence" << activeSequence << std::endl;

    if (activeSequence == nrInputSequences - 1 && nrInputSequences > 1)
    {
        // SIFT

        faat_pcl::rec_3d_framework::SIFTLocalEstimation<pcl::PointXYZRGB, pcl::PFHSignature125 > estimator;

        std::vector<pcl::PointCloud<pcl::PFHSignature125 >::Ptr> model_features;
        std::vector<pcl::PointCloud<pcl::PointXYZRGB >::Ptr> model_keypoints;

        for (int i=0;i<inputSequences.size();i++)
        {
            pcl::PointCloud<pcl::PFHSignature125 >::Ptr partial_features(new pcl::PointCloud<pcl::PFHSignature125 >);
            pcl::PointCloud<pcl::PointXYZRGB >::Ptr partial_keypoints(new pcl::PointCloud<pcl::PointXYZRGB >);

            for (int j=0;j<inputSequences[i].size();j++)
            {
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints(new pcl::PointCloud<pcl::PointXYZRGB>);
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr processed(new pcl::PointCloud<pcl::PointXYZRGB>);
                pcl::PointCloud<pcl::PFHSignature125 >::Ptr signatures(new pcl::PointCloud<pcl::PFHSignature125 >);

                estimator.setIndices(indices[i][j]);
                bool ret = estimator.estimate(inputSequences[i][j], processed, keypoints, signatures);

                (*partial_features) += *signatures;

                pcl::transformPointCloud(*keypoints, *keypoints, poses[i][j]);

                (*partial_keypoints) += *keypoints;
            }

            model_features.push_back(partial_features);
            model_keypoints.push_back(partial_keypoints);
        }

        pcl::registration::CorrespondenceEstimation<pcl::PFHSignature125, pcl::PFHSignature125 > corEst;
        std::vector<boost::shared_ptr<pcl::Correspondences> > correspondences;

        for (int i=1;i<inputSequences.size();i++)
        {
            pcl::PointCloud<pcl::PFHSignature125 >::Ptr source = model_features[i-1];
            pcl::PointCloud<pcl::PFHSignature125 >::Ptr target = model_features[i];

            corEst.setInputSource(source);
            corEst.setInputTarget(target);

            boost::shared_ptr<pcl::Correspondences> cor (new pcl::Correspondences());
            corEst.determineReciprocalCorrespondences(*cor);

            std::cout << "Correspondences found: " << cor->size() << std::endl;

            correspondences.push_back(cor);
        }
        // TRANSFORMATION

        pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZRGB> crsac;

        //pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);

        //(*cloud) += *models[0];
        for (int i=1;i<models.size();i++)
        {
            crsac.setInputSource(model_keypoints[0]);
            crsac.setInputTarget(model_keypoints[i]);
            crsac.setInlierThreshold(0.01f);
            crsac.setSaveInliers(true);

            crsac.setMaximumIterations(1000);
            //crsac.setInputCorrespondences(correspondences[i-1]);

            boost::shared_ptr<pcl::Correspondences> remaining_cor (new pcl::Correspondences());
            crsac.getRemainingCorrespondences(*(correspondences[i-1]), *remaining_cor);

            Eigen::Matrix4f transform = crsac.getBestTransformation();

            std::vector<int> inliers;
            crsac.getInliersIndices(inliers);

            std::cout << "number of inliers: " << inliers.size() << std::endl;

            std::cout << "transformation found: " << std::endl;
            std::cout << transform << std::endl;

            //std::cout << "fitness: " << sac.getFitnessScore() << std::endl;

            *(correspondences[i-1]) = *remaining_cor;

            pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr temp (new pcl::PointCloud<pcl::PointXYZRGBNormal>());
            pcl::copyPointCloud(*models[i], *temp);
            pcl::transformPointCloud(*temp, *temp, transform);

            //(*cloud) += (*temp);
        }

        //pcl::SampleConsensusInitialAlignment<pcl::PointXYZRGB, pcl::PointXYZRGB, pcl::PFHSignature125> sac;

        //pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);

        //(*cloud) += *models[0];
        for (int i=1;i<models.size();i++)
        {
            /*
            sac.setInputSource(model_keypoints[i-1]);
            sac.setInputTarget(model_keypoints[i]);
            //sac.setInlierThreshold(0.8f);

            sac.setMaxCorrespondenceDistance(1.8f);
            sac.setMinSampleDistance(0.01f);

            sac.setMaximumIterations(1000);
            //sac.setInputCorrespondences(correspondences[i-1]);
            sac.setSourceFeatures(model_features[i-1]);
            sac.setTargetFeatures(model_features[i]);

            pcl::PointCloud<pcl::PointXYZRGB> out;
            sac.align(out);
            Eigen::Matrix4f transform = sac.getFinalTransformation();


            std::cout << "transformation found: " << std::endl;
            std::cout << transform << std::endl;

            std::cout << "fitness: " << sac.getFitnessScore() << std::endl;
            */

            std::cout << "run svd" << std::endl;

            Eigen::Matrix4f svd_pose;
            pcl::registration::TransformationEstimationSVD<pcl::PointXYZRGB, pcl::PointXYZRGB, float> svd;
            svd.estimateRigidTransformation(*(model_keypoints[i-1]), *(model_keypoints[i]), *(correspondences[i-1]), svd_pose);

            std::cout << "svd pose " << std::endl;
            std::cout << svd_pose << std::endl;

            Eigen::Matrix4f inv_transform;
            kp::invPose(svd_pose, inv_transform);
            //return svd_pose;


            pcl::IterativeClosestPoint<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal> icp;
            pcl::PointCloud<pcl::PointXYZRGBNormal> icp_out;

            icp.setTransformationEpsilon(1e-6);
            icp.setMaxCorrespondenceDistance(0.01);
            icp.setInputSource(models[i-1]);
            icp.setInputTarget(models[i]);
            icp.align(icp_out, svd_pose);
            inv_transform = icp.getFinalTransformation();

            std::cout << "icp pose " << std::endl;
            std::cout << inv_transform << std::endl;

            Eigen::Matrix4f inv_transform2;
            kp::invPose(inv_transform, inv_transform2);

            return inv_transform2;

            //pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr temp (new pcl::PointCloud<pcl::PointXYZRGBNormal>());
            //pcl::copyPointCloud(*models[i], *temp);
            //pcl::transformPointCloud(*temp, *temp, inv_transform2);

            //(*cloud) += (*temp);
        }

        //result.push_back(cloud);
        std::cout << "process sequences" << std::endl;
    }

    return result;
}

/*
std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> SiftFeatureMatcher::process(boost::tuples::tuple<std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>, std::vector<Eigen::Matrix4f>, std::vector<std::vector<int> >, std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> > input)
{
    std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> result;

    inputSequences.push_back(boost::tuples::get<0>(input));
    poses.push_back(boost::tuples::get<1>(input));
    indices.push_back(boost::tuples::get<2>(input));
    models.push_back(boost::tuples::get<3>(input)[0]);
    std::cout << "add sequence" << activeSequence << std::endl;

    if (activeSequence == nrInputSequences - 1)
    {
        // SIFT

        faat_pcl::rec_3d_framework::SIFTLocalEstimation<pcl::PointXYZRGB, pcl::Histogram<128> > estimator;

        std::vector<pcl::PointCloud<pcl::Histogram<128> >::Ptr> model_features;
        std::vector<pcl::PointCloud<pcl::PointXYZRGB >::Ptr> model_keypoints;

        for (int i=0;i<inputSequences.size();i++)
        {
            pcl::PointCloud<pcl::Histogram<128> >::Ptr partial_features(new pcl::PointCloud<pcl::Histogram<128> >);
            pcl::PointCloud<pcl::PointXYZRGB >::Ptr partial_keypoints(new pcl::PointCloud<pcl::PointXYZRGB >);

            for (int j=0;j<inputSequences[i].size();j++)
            {
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints(new pcl::PointCloud<pcl::PointXYZRGB>);
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr processed(new pcl::PointCloud<pcl::PointXYZRGB>);
                pcl::PointCloud<pcl::Histogram<128> >::Ptr signatures(new pcl::PointCloud<pcl::Histogram<128> >);

                estimator.setIndices(indices[i][j]);
                bool ret = estimator.estimate(inputSequences[i][j], processed, keypoints, signatures);

                (*partial_features) += *signatures;

                //pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_keypoints(new pcl::PointCloud<pcl::PointXYZRGB>);
                //pcl::transformPointCloud(*keypoints, *transformed_keypoints, poses[i][j]);

                (*partial_keypoints) += *keypoints;
            }

            model_features.push_back(partial_features);
            model_keypoints.push_back(partial_keypoints);
        }

        std::vector<boost::shared_ptr<pcl::Correspondences> > correspondences;

        for (int i=1;i<inputSequences.size();i++)
        {
            pcl::PointCloud<pcl::Histogram<128> >::Ptr source = model_features[i-1];
            pcl::PointCloud<pcl::Histogram<128> >::Ptr target = model_features[i];

            boost::shared_ptr<pcl::Correspondences> cor (new pcl::Correspondences());

            pcl::KdTreeFLANN<pcl::Histogram<128> > kd;
            kd.setInputCloud(target);

            const int k = 1;
            std::vector<int> k_indices(k);
            std::vector<float> k_squared_distances(k);

            for (int j=0;j<source->size();j++)
            {
                int found = kd.nearestKSearch(source->at(j), k, k_indices, k_squared_distances);

                if (found == 1)
                {
                    pcl::Correspondence c(j, k_indices[0], k_squared_distances[0]);

                    //std::cout << "found cor: " << k_indices[0] << " - " << k_squared_distances[0] << std::endl;

                    cor->push_back(c);
                }
            }

            std::cout << "Correspondences found: " << cor->size() << std::endl;

            correspondences.push_back(cor);
        }
        // TRANSFORMATION

        pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZRGB> crsac;

        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);

        (*cloud) += *models[0];
        for (int i=1;i<models.size();i++)
        {
            crsac.setInputSource(model_keypoints[i-1]);
            crsac.setInputTarget(model_keypoints[i]);
            crsac.setInlierThreshold(10.8f);

            crsac.setMaximumIterations(1000000);
            crsac.setInputCorrespondences(correspondences[i-1]);

            Eigen::Matrix4f transform = crsac.getBestTransformation();

            std::vector<int> inliers;
            crsac.getInliersIndices(inliers);

            std::cout << "number of inliers: " << inliers.size() << std::endl;

            std::cout << "transformation found: " << std::endl;
            std::cout << transform << std::endl;

            //std::cout << "fitness: " << sac.getFitnessScore() << std::endl;

            pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr temp (new pcl::PointCloud<pcl::PointXYZRGBNormal>());
            pcl::copyPointCloud(*models[i], *temp);
            pcl::transformPointCloud(*temp, *temp, transform);

            (*cloud) += (*temp);
        }

        pcl::SampleConsensusInitialAlignment<pcl::PointXYZRGB, pcl::PointXYZRGB, pcl::PFHSignature125> sac;

        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);

        (*cloud) += *models[0];
        for (int i=1;i<models.size();i++)
        {
            sac.setInputSource(model_keypoints[i-1]);
            sac.setInputTarget(model_keypoints[i]);
            //sac.setInlierThreshold(0.8f);

            sac.setMaxCorrespondenceDistance(1.8f);
            sac.setMinSampleDistance(0.01f);

            sac.setMaximumIterations(1000);
            //sac.setInputCorrespondences(correspondences[i-1]);
            sac.setSourceFeatures(model_features[i-1]);
            sac.setTargetFeatures(model_features[i]);

            pcl::PointCloud<pcl::PointXYZRGB> out;
            sac.align(out);
            Eigen::Matrix4f transform = sac.getFinalTransformation();

            std::cout << "transformation found: " << std::endl;
            std::cout << transform << std::endl;

            std::cout << "fitness: " << sac.getFitnessScore() << std::endl;

            pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr temp (new pcl::PointCloud<pcl::PointXYZRGBNormal>());
            pcl::copyPointCloud(*models[i], *temp);
            pcl::transformPointCloud(*temp, *temp, transform);

            (*cloud) += (*temp);
        }

        result.push_back(cloud);
        std::cout << "process sequences" << std::endl;
    }

    return result;
}

*/

}
}
