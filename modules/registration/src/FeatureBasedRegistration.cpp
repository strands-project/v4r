#include <v4r_config.h>
#include <v4r/registration/FeatureBasedRegistration.h>
#include <v4r/common/miscellaneous.h>
#include <v4r/common/impl/geometric_consistency.hpp>
#include <v4r/common/graph_geometric_consistency.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/impl/correspondence_estimation.hpp>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/search/impl/kdtree.hpp>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/octree/octree_impl.h>
#include <pcl/octree/octree_pointcloud_occupancy.h>
#include <pcl/octree/impl/octree_base.hpp>
#include <v4r/common/miscellaneous.h>

#ifdef HAVE_SIFTGPU
#include <v4r/features/sift_local_estimator.h>
#else
#include <v4r/features/opencv_sift_local_estimator.h>
#endif

namespace v4r
{
namespace Registration
{

template<class PointT>
FeatureBasedRegistration<PointT>::FeatureBasedRegistration()
{
    name_ = "FeatureBasedRegistration";
    do_cg_ = false;
    gc_threshold_ = 9;
    inlier_threshold_ = 0.015f;
    kdtree_splits_ = 512;
}

template<class PointT> void
FeatureBasedRegistration<PointT>::initialize(std::vector<std::pair<int, int> > & session_ranges)
{

#ifdef HAVE_SIFTGPU
    typename v4r::SIFTLocalEstimation<PointT> estimator;
#else
    typename v4r::OpenCVSIFTLocalEstimation<PointT> estimator;
#endif

    //computes features and keypoints for the views of all sessions using appropiate object indices
    size_t total_views = this->getTotalNumberOfClouds();
    std::cout << "total views in initialize:" << total_views << std::endl;

    sift_keypoints_.resize(total_views);
    sift_features_.resize(total_views);
    sift_normals_.resize(total_views);

    std::vector<int> cloud_idx_to_session(total_views);
    model_features_.resize(session_ranges.size());
    flann_data_.resize(session_ranges.size());
    flann_index_.resize(session_ranges.size());

    std::vector<boost::shared_ptr<pcl::octree::OctreePointCloudOccupancy<PointT> > > octree_sessions(session_ranges.size());

    for(size_t i=0; i < session_ranges.size(); i++)
    {
        for(int t=session_ranges[i].first; t <= session_ranges[i].second; t++)
            cloud_idx_to_session[t] = static_cast<int>(i);

        octree_sessions[i].reset(new pcl::octree::OctreePointCloudOccupancy<PointT>(0.003f));
    }

    for(size_t i=0; i < total_views; i++)
    {
        typename pcl::PointCloud<PointT>::Ptr cloud = this->getCloud(i);
        pcl::PointCloud<pcl::Normal>::Ptr normals = this->getNormal(i);
        std::vector<int> & indices = this->getIndices(i);
        Eigen::Matrix4f pose = this->getPose(i);

        sift_keypoints_[i].reset(new pcl::PointCloud<PointT>);
        sift_normals_[i].reset(new pcl::PointCloud< pcl::Normal >);

        std::vector<std::vector<float> > sift_descs;
        estimator.setInputCloud(cloud);
        estimator.setIndices(indices);
        estimator.compute(sift_descs);
        typename pcl::PointCloud< PointT >::Ptr  sift_keys = estimator.getKeypointCloud();

        pcl::PointIndices original_indices;
        original_indices.indices = estimator.getKeypointIndices();

        //check if there exist already a feature at sift_keys[k] position, add points to octree that were not found before
        std::vector<int> non_occupied;
        non_occupied.reserve(sift_keys->points.size());

        std::vector<int> original_indices_non_occupied;

        for(size_t k=0; k < sift_keys->points.size(); k++)
        {
            PointT p;
            p.getVector4fMap() = pose * sift_keys->points[k].getVector4fMap();
            if(!(octree_sessions[cloud_idx_to_session[i]]->isVoxelOccupiedAtPoint(p)))
            {
                non_occupied.push_back(static_cast<int>(k));
                original_indices_non_occupied.push_back(original_indices.indices[k]);
            }
        }

        std::cout << non_occupied.size() << " " << sift_keys->points.size() << std::endl;

        pcl::copyPointCloud(*sift_keys, non_occupied, *sift_keypoints_[i]);

        sift_features_[i] = filterVector(sift_descs, non_occupied);

        pcl::copyPointCloud(*normals, original_indices_non_occupied, *sift_normals_[i]);

        /*typename pcl::PointCloud< PointT >::Ptr sift_keys_trans(new pcl::PointCloud< PointT >);
        pcl::transformPointCloud(*sift_keypoints_[i], *sift_keys_trans, pose);

        octree_sessions[cloud_idx_to_session[i]]->setInputCloud(sift_keys_trans);
        octree_sessions[cloud_idx_to_session[i]]->addPointsFromInputCloud();*/

        model_features_[cloud_idx_to_session[i]].insert(model_features_[cloud_idx_to_session[i]].end(),
                sift_features_[i].begin(),
                sift_features_[i].end());
    }

    for(size_t i=0; i < session_ranges.size(); i++)
    {
        flann::Matrix<float> flann_data (new float[model_features_[i].size () * 128], model_features_[i].size (), 128);

        for (size_t r = 0; r < model_features_[i].size (); ++r)
            for (size_t j = 0; j < 128; ++j)
                flann_data.ptr ()[r * 128 + j] = model_features_[i][r][j];

        flann_data_[i] = flann_data;

        flann_index_[i] = new flann::Index< DistT > (flann_data_[i], flann::KDTreeIndexParams (4));
        flann_index_[i]->buildIndex ();
    }
}

template<class PointT> void
FeatureBasedRegistration<PointT>::compute(int s1, int s2)
{

    //compute sift features for views in partial_1 and partial_2 (already computed in initialize in fact)
    //match features

    boost::shared_ptr<pcl::Correspondences> cor (new pcl::Correspondences());

    int knn=1;

    for (size_t i = 0; i < model_features_[s2].size (); ++i)
    {
        int size_feat = 128;
        flann::Matrix<int> indices;
        flann::Matrix<float> distances;
        distances = flann::Matrix<float> (new float[knn], 1, knn);
        indices = flann::Matrix<int> (new int[knn], 1, knn);

        flann::Matrix<float> p = flann::Matrix<float> (new float[size_feat], 1, size_feat);
        memcpy (&p.ptr ()[0], &model_features_[s2][i][0], size_feat * sizeof(float));
        flann_index_[s1]->knnSearch (p, indices, distances, knn, flann::SearchParams (kdtree_splits_));

        for(int n=0; n < knn; n++)
        {
            pcl::Correspondence corr (static_cast<int> (i), indices[0][n], distances[0][n]);
            cor->push_back (corr);
        }
    }

    std::cout << "Correspondences found: " << cor->size() << std::endl;

    //transform all view-based keypoints to common reference frame
    //GC (not implemented now) + RANSAC + SVD
    typename pcl::PointCloud<PointT>::Ptr kps_s1(new pcl::PointCloud<PointT>);
    typename pcl::PointCloud<PointT>::Ptr kps_s2(new pcl::PointCloud<PointT>);
    typename pcl::PointCloud<pcl::Normal>::Ptr normals_s1(new pcl::PointCloud<pcl::Normal>);
    typename pcl::PointCloud<pcl::Normal>::Ptr normals_s2(new pcl::PointCloud<pcl::Normal>);

    for(int t=partial_1.first; t <= partial_1.second; t++)
    {
        typename pcl::PointCloud<PointT>::Ptr transformed(new pcl::PointCloud<PointT>);
        Eigen::Matrix4f pose_inv = this->getPose(static_cast<size_t>(t));
        pcl::transformPointCloud(*sift_keypoints_[t], *transformed, pose_inv);
        *kps_s1 += *transformed;

        typename pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        transformNormals(*sift_normals_[t], *normals, pose_inv);

        *normals_s1 += *normals;
    }

    for(int t=partial_2.first; t <= partial_2.second; t++)
    {
        typename pcl::PointCloud<PointT>::Ptr transformed(new pcl::PointCloud<PointT>);
        Eigen::Matrix4f pose_inv = this->getPose(static_cast<size_t>(t));
        pcl::transformPointCloud(*sift_keypoints_[t], *transformed, pose_inv);
        *kps_s2 += *transformed;

        typename pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        transformNormals(*sift_normals_[t], *normals, pose_inv);
        *normals_s2 += *normals;
    }

    std::cout << kps_s1->points.size() << " " << model_features_[s1].size() << " " << normals_s1->points.size() << std::endl;
    std::cout << kps_s2->points.size() << " " << model_features_[s2].size() << " " << normals_s2->points.size() << std::endl;

    /*{
        pcl::visualization::PCLVisualizer vis("keypoints with normals");
        pcl::visualization::PointCloudColorHandlerCustom<PointT> handler (kps_s1, 255, 0 ,0);
        vis.addPointCloud<PointT> (kps_s1, handler, "center_cloud");
        vis.addPointCloudNormals<PointT,pcl::Normal> (kps_s1, normals_s1, 1, 0.01, "normal_center_cloud");
        vis.spin();
    }

    {
        pcl::visualization::PCLVisualizer vis("keypoints with normals");
        pcl::visualization::PointCloudColorHandlerCustom<PointT> handler (kps_s2, 255, 0 ,0);
        vis.addPointCloud<PointT> (kps_s2, handler, "center_cloud");
        vis.addPointCloudNormals<PointT,pcl::Normal> (kps_s2, normals_s2, 1, 0.01, "normal_center_cloud");
        vis.spin();
    }*/

    if(do_cg_)
    {

        std::vector<pcl::Correspondences> clustered_corrs;
        bool graph_based = true;
        if(!graph_based)
        {
            GeometricConsistencyGrouping<PointT, PointT> gc_clusterer;
            gc_clusterer.setGCSize (inlier_threshold_);
            gc_clusterer.setGCThreshold (gc_threshold_);

            gc_clusterer.setInputCloud (kps_s2);
            gc_clusterer.setSceneCloud (kps_s1);
            gc_clusterer.setModelSceneCorrespondences (*cor);

            gc_clusterer.cluster (clustered_corrs);
        }
        else
        {
            typename GraphGeometricConsistencyGrouping<PointT, PointT>::Parameter param;
            param.gc_size_ = inlier_threshold_;
            param.gc_threshold_ = gc_threshold_;
            param.ransac_threshold_ = inlier_threshold_;
            param.dist_for_cluster_factor_ = 1.f;
            param.thres_dot_distance_ = 0.25f;
            param.max_taken_correspondence_ = 2;
            param.max_time_allowed_cliques_comptutation_ = 50;
            param.check_normals_orientation_ = true;
            GraphGeometricConsistencyGrouping<PointT, PointT> gc_clusterer;

            gc_clusterer.setInputCloud (kps_s2);
            gc_clusterer.setSceneCloud (kps_s1);
            gc_clusterer.setInputAndSceneNormals(normals_s2, normals_s1);
            gc_clusterer.setModelSceneCorrespondences (*cor);
            gc_clusterer.cluster (clustered_corrs);
        }

        std::cout << "clustered_corrs size:" << clustered_corrs.size() << " " << cor->size() << std::endl;
        poses_.resize(clustered_corrs.size());

        size_t max_cluster = 0;

        for(size_t jj=0; jj < clustered_corrs.size(); jj++)
        {
            if(clustered_corrs[jj].size() > max_cluster)
            {
                max_cluster = clustered_corrs[jj].size();
            }
        }

        //#define VIS_SVD

#ifdef VIS_SVD
        pcl::visualization::PCLVisualizer vis("pairwise alignment (svd)");
#endif

        int used = 0;
        for(size_t jj=0; jj < clustered_corrs.size(); jj++)
        {
            float sizee = static_cast<float>(clustered_corrs[jj].size());
            /*if(sizee < static_cast<float>(max_cluster) * 0.5f )
                continue;*/

            std::cout << sizee << " " << max_cluster << std::endl;

            Eigen::Matrix4f svd_pose;
            pcl::registration::TransformationEstimationSVD<PointT, PointT, float> svd;
            svd.estimateRigidTransformation(*kps_s2, *kps_s1, clustered_corrs[jj], svd_pose);
            poses_[used] = svd_pose;
            used++;

#ifdef VIS_SVD
            typename pcl::PointCloud<PointT>::Ptr kps_s2_after_ransac(new pcl::PointCloud<PointT>);
            pcl::transformPointCloud(*kps_s2, *kps_s2_after_ransac, svd_pose);

            {
                vis.addPointCloud(kps_s1, "cloud_1");
                vis.addPointCloud(kps_s2_after_ransac, "cloud_2");
                vis.spin();
                vis.removeAllPointClouds();
            }
#endif

        }

        poses_.resize(used);
    }
    else
    {
        pcl::registration::CorrespondenceRejectorSampleConsensus<PointT> crsac;
        crsac.setInputSource(kps_s2);
        crsac.setInputTarget(kps_s1);
        crsac.setInlierThreshold(inlier_threshold_);
        crsac.setMaximumIterations (50000);

        boost::shared_ptr<pcl::Correspondences> remaining (new pcl::Correspondences());
        crsac.getRemainingCorrespondences(*cor, *remaining);

        std::cout << "Correspondences after filtering: " << remaining->size() << std::endl;

        Eigen::Matrix4f svd_pose;
        pcl::registration::TransformationEstimationSVD<PointT, PointT, float> svd;
        svd.estimateRigidTransformation(*kps_s2, *kps_s1, *remaining, svd_pose);

        std::cout << svd_pose << std::endl;

        /// debug
        /*pcl::visualization::PCLVisualizer vis("correspondences");
        int v1,v2;
        vis.createViewPort(0,0,0.5,1,v1);
        vis.createViewPort(0.5,0,1,1,v2);

        vis.addPointCloud(kps_s1, "cloud_1", v1);
        vis.addPointCloud(kps_s2, "cloud_2", v2);

        vis.spin();*/

        /*Eigen::Matrix4f ransac_pose = crsac.getBestTransformation();

        typename pcl::PointCloud<PointT>::Ptr kps_s2_after_ransac(new pcl::PointCloud<PointT>);
        pcl::transformPointCloud(*kps_s2, *kps_s2_after_ransac, ransac_pose);

        {
            pcl::visualization::PCLVisualizer vis("pairwise alignment");
            vis.addPointCloud(kps_s1, "cloud_1");
            vis.addPointCloud(kps_s2_after_ransac, "cloud_2");
            vis.spin();
        }*/

        /*
        {

            typename pcl::PointCloud<PointT>::Ptr kps_s2_after_ransac(new pcl::PointCloud<PointT>);
            pcl::transformPointCloud(*kps_s2, *kps_s2_after_ransac, svd_pose);

            {
                pcl::visualization::PCLVisualizer vis("pairwise alignment (svd)");
                vis.addPointCloud(kps_s1, "cloud_1");
                vis.addPointCloud(kps_s2_after_ransac, "cloud_2");
                vis.spin();
            }
        }*/

        //write poses aligning partial_2 to partial_1 (right now, only 1)
        poses_.resize(1);
        poses_[0] = svd_pose;
    }
}


template<class PointT>
std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >
FeatureBasedRegistration<PointT>::estimateViewTransformationBySIFT(const pcl::PointCloud<PointT> &src_cloud,
                                                                   const pcl::PointCloud<PointT> &dst_cloud,
                                                                   const std::vector<int> &src_sift_keypoint_indices,
                                                                   const std::vector<int> &dst_sift_keypoint_indices,
                                                                   const std::vector<std::vector<float> > &src_sift_signatures,
                                                                   const std::vector<std::vector<float> > &dst_sift_signatures,
                                                                   bool use_gc )
 {
     std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transformations;
     const int K = 1;
     flann::Matrix<int> indices = flann::Matrix<int> ( new int[K], 1, K );
     flann::Matrix<float> distances = flann::Matrix<float> ( new float[K], 1, K );

     boost::shared_ptr< flann::Index<DistT> > flann_index;
     convertToFLANN( dst_sift_signatures, flann_index );

     boost::shared_ptr< pcl::PointCloud<PointT> > pSiftKeypointsSrc (new pcl::PointCloud<PointT>);
     boost::shared_ptr< pcl::PointCloud<PointT> > pSiftKeypointsDst (new pcl::PointCloud<PointT>);
     pcl::copyPointCloud(src_cloud, src_sift_keypoint_indices, *pSiftKeypointsSrc );
     pcl::copyPointCloud(dst_cloud, dst_sift_keypoint_indices, *pSiftKeypointsDst);

     pcl::CorrespondencesPtr temp_correspondences ( new pcl::Correspondences );
     temp_correspondences->resize(pSiftKeypointsSrc->size ());

     for ( size_t keypointId = 0; keypointId < pSiftKeypointsSrc->points.size (); keypointId++)
     {
         nearestKSearch ( flann_index, src_sift_signatures[keypointId], K, indices, distances );

         pcl::Correspondence corr;
         corr.distance = distances[0][0];
         corr.index_query = keypointId;
         corr.index_match = indices[0][0];
         temp_correspondences->at(keypointId) = corr;
     }

     if(!use_gc)
     {
         typename pcl::registration::CorrespondenceRejectorSampleConsensus<PointT>::Ptr rej;
         rej.reset (new pcl::registration::CorrespondenceRejectorSampleConsensus<PointT> ());
         pcl::CorrespondencesPtr after_rej_correspondences (new pcl::Correspondences ());

         rej->setMaximumIterations (50000);
         rej->setInlierThreshold (0.02);
         rej->setInputTarget (pSiftKeypointsDst);
         rej->setInputSource (pSiftKeypointsSrc);
         rej->setInputCorrespondences (temp_correspondences);
         rej->getCorrespondences (*after_rej_correspondences);

         Eigen::Matrix4f refined_pose;
         transformations.push_back( rej->getBestTransformation () );
         pcl::registration::TransformationEstimationSVD<PointT, PointT> t_est;
         t_est.estimateRigidTransformation (*pSiftKeypointsSrc, *pSiftKeypointsDst, *after_rej_correspondences, refined_pose);
         transformations.back() = refined_pose;
     }
     else
     {
         std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > new_transforms;
         pcl::GeometricConsistencyGrouping<PointT, PointT> gcg_alg;

         gcg_alg.setGCThreshold (15);
         gcg_alg.setGCSize (0.01);
         gcg_alg.setInputCloud(pSiftKeypointsSrc);
         gcg_alg.setSceneCloud(pSiftKeypointsDst);
         gcg_alg.setModelSceneCorrespondences(temp_correspondences);

         std::vector<pcl::Correspondences> clustered_corrs;
         gcg_alg.recognize(new_transforms, clustered_corrs);
         transformations.insert(transformations.end(), new_transforms.begin(), new_transforms.end());
     }
     return transformations;
 }

}


template class V4R_EXPORTS Registration::FeatureBasedRegistration<pcl::PointXYZRGB>;

}
