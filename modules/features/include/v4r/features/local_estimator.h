/*
 * estimator.h
 *
 *  Created on: Mar 22, 2012
 *      Author: Aitor Aldoma
 *      Maintainer: Thomas Faeulhammer
 */

#ifndef REC_FRAMEWORK_LOCAL_ESTIMATOR_H_
#define REC_FRAMEWORK_LOCAL_ESTIMATOR_H_

#include <v4r/common/normal_estimator.h>
#include "pcl/keypoints/uniform_sampling.h"
#include <pcl/surface/mls.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/keypoints/susan.h>
#include <pcl/keypoints/iss_3d.h>
#include <v4r/core/macros.h>
#include <v4r/common/faat_3d_rec_framework_defines.h>

namespace v4r
{

template<typename PointInT>
class V4R_EXPORTS KeypointExtractor
{
protected:
    typedef typename pcl::PointCloud<PointInT>::Ptr PointInTPtr;
    typedef typename pcl::PointCloud<PointInT>::Ptr PointOutTPtr;
    typename pcl::PointCloud<PointInT>::Ptr input_;
    float radius_;
    pcl::PointIndicesConstPtr keypoint_indices_;

public:

    void
    setInputCloud (const PointInTPtr & input)
    {
        input_ = input;
    }

    void
    setSupportRadius (float f)
    {
        radius_ = f;
    }

    virtual void
    compute (PointOutTPtr & keypoints) = 0;

    virtual void
    setNormals (const pcl::PointCloud<pcl::Normal>::Ptr & normals)
    {
        (void)normals;
        std::cerr << "setNormals is not implemented for this object." << std::endl;
    }

    virtual bool
    needNormals ()
    {
        return false;
    }

    void getKeypointsIndices (pcl::PointIndices::Ptr &keypoint_indices) const
    {
        *keypoint_indices = *keypoint_indices_;
    }
};

template<typename PointInT>
class V4R_EXPORTS UniformSamplingExtractor : public KeypointExtractor<PointInT>
{
private:

private:
    typedef typename pcl::PointCloud<PointInT>::Ptr PointInTPtr;
    bool filter_planar_;
    using v4r::KeypointExtractor<PointInT>::input_;
    using v4r::KeypointExtractor<PointInT>::radius_;
    using KeypointExtractor<PointInT>::keypoint_indices_;
    float sampling_density_;
    boost::shared_ptr<std::vector<std::vector<int> > > neighborhood_indices_;
    boost::shared_ptr<std::vector<std::vector<float> > > neighborhood_dist_;
    float max_distance_;
    float threshold_planar_;
    bool z_adaptative_;
    bool force_unorganized_;

    void
    filterPlanar (PointInTPtr & input, pcl::PointCloud<int> & keypoints_cloud)
    {
        pcl::PointCloud<int> filtered_keypoints;
        //create a search object
        typename pcl::search::Search<PointInT>::Ptr tree;
        if (input->isOrganized () && !force_unorganized_)
            tree.reset (new pcl::search::OrganizedNeighbor<PointInT> ());
        else
            tree.reset (new pcl::search::KdTree<PointInT> (false));
        tree->setInputCloud (input);

        neighborhood_indices_.reset (new std::vector<std::vector<int> >);
        neighborhood_indices_->resize (keypoints_cloud.points.size ());
        neighborhood_dist_.reset (new std::vector<std::vector<float> >);
        neighborhood_dist_->resize (keypoints_cloud.points.size ());

        filtered_keypoints.points.resize (keypoints_cloud.points.size ());
        int good = 0;

        for (size_t i = 0; i < keypoints_cloud.points.size (); i++)
        {

            if (tree->radiusSearch (keypoints_cloud[i], radius_, (*neighborhood_indices_)[good], (*neighborhood_dist_)[good]))
            {

                EIGEN_ALIGN16 Eigen::Matrix3f covariance_matrix;
                Eigen::Vector4f xyz_centroid;
                EIGEN_ALIGN16 Eigen::Vector3f eigenValues;
                EIGEN_ALIGN16 Eigen::Matrix3f eigenVectors;

                //compute planarity of the region
                computeMeanAndCovarianceMatrix (*input, (*neighborhood_indices_)[good], covariance_matrix, xyz_centroid);
                pcl::eigen33 (covariance_matrix, eigenVectors, eigenValues);

                float eigsum = eigenValues.sum ();
                if (!pcl_isfinite(eigsum))
                {
                    PCL_ERROR("Eigen sum is not finite\n");
                }

                float t_planar = threshold_planar_;
                if(z_adaptative_)
                {
                    t_planar *= (1 + (std::max(input->points[keypoints_cloud.points[i]].z,1.f) - 1.f));
                }

                //if ((fabs (eigenValues[0] - eigenValues[1]) < 1.5e-4) || (eigsum != 0 && fabs (eigenValues[0] / eigsum) > 1.e-2))
                if ((fabs (eigenValues[0] - eigenValues[1]) < 1.5e-4) || (eigsum != 0 && fabs (eigenValues[0] / eigsum) > t_planar))
                {
                    //region is not planar, add to filtered keypoint
                    keypoints_cloud.points[good] = keypoints_cloud.points[i];
                    good++;
                }
            }
        }

        neighborhood_indices_->resize (good);
        neighborhood_dist_->resize (good);
        keypoints_cloud.points.resize (good);

        neighborhood_indices_->clear ();
        neighborhood_dist_->clear ();

    }

public:

    UniformSamplingExtractor()
    {
        max_distance_ = std::numeric_limits<float>::infinity();
        threshold_planar_ = 1.e-2;
        z_adaptative_ = false;
        force_unorganized_ = false;
    }

    void setForceUnorganized(bool b)
    {
        force_unorganized_ = b;
    }

    void zAdaptative(bool b)
    {
        z_adaptative_ = b;
    }

    void setThresholdPlanar(float t)
    {
        threshold_planar_ = t;
    }

    void setMaxDistance(float d)
    {
        max_distance_ = d;
    }

    void
    setFilterPlanar (bool b)
    {
        filter_planar_ = b;
    }

    void
    setSamplingDensity (float f)
    {
        sampling_density_ = f;
    }

    void
    compute (PointInTPtr & keypoints)
    {
        keypoints.reset (new pcl::PointCloud<PointInT>);

        pcl::UniformSampling<PointInT> keypoint_extractor;
        keypoint_extractor.setRadiusSearch (sampling_density_);
        keypoint_extractor.setInputCloud (input_);

        pcl::PointCloud<int> keypoints_idxes;
        keypoint_extractor.compute (keypoints_idxes);

        pcl::PointIndicesPtr pIndicesPcl;
        pIndicesPcl.reset(new pcl::PointIndices);
        for (size_t i=0; i < keypoints_idxes.size(); i++)
        {
            pIndicesPcl->indices.push_back(keypoints_idxes.at(i));
        }

        if(pcl_isfinite(max_distance_))
        {
            int valid = 0;
            int original_size = (int)keypoints_idxes.size();
            for(size_t i=0; i < keypoints_idxes.size(); i++)
            {
                if(input_->points[keypoints_idxes.points[i]].z < max_distance_)
                {
                    keypoints_idxes.points[valid] = keypoints_idxes.points[i];
                    valid++;
                }
            }

            keypoints_idxes.points.resize(valid);
            keypoints_idxes.width = valid;
            PCL_WARN("filtered %d keypoints based on z-distance %f\n", (original_size - valid), max_distance_);
        }

        if (filter_planar_)
            filterPlanar (input_, keypoints_idxes);

        std::vector<int> indices;
        indices.resize (keypoints_idxes.points.size ());
        for (size_t i = 0; i < indices.size (); i++)
            indices[i] = keypoints_idxes.points[i];

        pcl::copyPointCloud (*input_, indices, *keypoints);
        pIndicesPcl->indices = indices;
        keypoint_indices_ = pIndicesPcl;
    }

    void
    compute (std::vector<int> & indices)
    {
        pcl::UniformSampling<PointInT> keypoint_extractor;
        keypoint_extractor.setRadiusSearch (sampling_density_);
        keypoint_extractor.setInputCloud (input_);

        pcl::PointCloud<int> keypoints_idxes;
        keypoint_extractor.compute (keypoints_idxes);

        if (filter_planar_)
            filterPlanar (input_, keypoints_idxes);

        indices.resize (keypoints_idxes.points.size ());
        for (size_t i = 0; i < indices.size (); i++)
            indices[i] = keypoints_idxes.points[i];
    }
};

template<typename PointInT, typename FeatureT>
class V4R_EXPORTS LocalEstimator
{
protected:
    typedef typename pcl::PointCloud<PointInT>::Ptr PointInTPtr;
    typedef typename pcl::PointCloud<FeatureT>::Ptr FeatureTPtr;

    typename boost::shared_ptr<PreProcessorAndNormalEstimator<PointInT, pcl::Normal> > normal_estimator_;
    pcl::PointCloud<pcl::Normal>::Ptr normals_;
    std::vector<typename boost::shared_ptr<KeypointExtractor<PointInT> > > keypoint_extractor_; //this should be a vector
    pcl::PointIndices keypoint_indices_;


    boost::shared_ptr<std::vector<std::vector<int> > > neighborhood_indices_;
    boost::shared_ptr<std::vector<std::vector<float> > > neighborhood_dist_;

    void
    computeKeypoints (const PointInTPtr & cloud, PointInTPtr & keypoints, const pcl::PointCloud<pcl::Normal>::Ptr & normals)
    {
        keypoint_indices_.indices.clear();
        keypoints.reset (new pcl::PointCloud<PointInT>);
        for (size_t i = 0; i < keypoint_extractor_.size (); i++)
        {
            keypoint_extractor_[i]->setInputCloud (cloud);
            if (keypoint_extractor_[i]->needNormals ())
                keypoint_extractor_[i]->setNormals (normals);

            keypoint_extractor_[i]->setSupportRadius (param_.support_radius_);

            PointInTPtr detected_keypoints;
            //std::vector<int> keypoint_indices;
            keypoint_extractor_[i]->compute (detected_keypoints);

            pcl::PointIndicesPtr pKeypointPclIndices (new pcl::PointIndices);
            keypoint_extractor_[i]->getKeypointsIndices(pKeypointPclIndices);
            keypoint_indices_.indices.insert(keypoint_indices_.indices.end(), pKeypointPclIndices->indices.begin(), pKeypointPclIndices->indices.end());
            *keypoints += *detected_keypoints;
        }
    }

public:

    class V4R_EXPORTS Parameter
    {
    public:
        int normal_computation_method_;
        float support_radius_;
        bool adaptative_MLS_;

        Parameter(
                int normal_computation_method = 2,
                float support_radius = 0.04f,
                bool adaptive_MLS = false)
            :
              normal_computation_method_ (normal_computation_method),
              support_radius_ (support_radius),
              adaptative_MLS_ (adaptive_MLS)
        {}
    }param_;

    LocalEstimator (const Parameter &p = Parameter())
    {
        param_ = p;
        keypoint_extractor_.clear ();
    }


    virtual size_t getFeatureType() const
    {
        return 0;
    }

    void
    setAdaptativeMLS (const bool b)
    {
        param_.adaptative_MLS_ = b;
    }

    virtual bool acceptsIndices() const
    {
        return false;
    }

    void getKeypointIndices(pcl::PointIndices & indices) const
    {
        indices = keypoint_indices_;
    }

//    void getKeypointIndices(std::vector<int> &keypoint_indices) const
//    {
//        keypoint_indices = &keypoint_indices_;
//    }

    virtual void
    setIndices(const pcl::PointIndices & p_indices)
    {
        (void) p_indices;
        std::cerr << "This function is not implemented!" << std::endl;
    }

    virtual void
    setIndices(const std::vector<int> & p_indices)
    {
        (void) p_indices;
        std::cerr << "This function is not implemented!" << std::endl;
    }

    virtual bool
    estimate (const PointInTPtr & in, PointInTPtr & processed, PointInTPtr & keypoints, FeatureTPtr & signatures)=0;

    /**
         * \brief Right now only uniformSampling keypoint extractor is allowed
         */
    void
    addKeypointExtractor (boost::shared_ptr<KeypointExtractor<PointInT> > & ke)
    {
        keypoint_extractor_.push_back (ke);
    }

    void
    setKeypointExtractors (std::vector<typename boost::shared_ptr<KeypointExtractor<PointInT> > > & ke)
    {
        keypoint_extractor_ = ke;
    }

    void
    setSupportRadius (const float r)
    {
        param_.support_radius_ = r;
    }

    virtual bool
    needNormals ()
    {
        return false;
    }

    void getNormals(pcl::PointCloud<pcl::Normal>::Ptr & normals) const
    {
        normals = normals_;
    }

    /**
     * @brief sets the normals point cloud of the scene
     * @param normals
     */
    void setNormals(const pcl::PointCloud<pcl::Normal>::Ptr & normals)
    {
        normals_ = normals;
    }

    virtual
    std::string getFeatureDescriptorName() const = 0;
};
}

#endif /* REC_FRAMEWORK_LOCAL_ESTIMATOR_H_ */
