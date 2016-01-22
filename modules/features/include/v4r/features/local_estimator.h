/******************************************************************************
 * Copyright (c) 2013 Aitor Aldoma, Thomas Faeulhammer
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 ******************************************************************************/

#ifndef V4R_LOCAL_ESTIMATOR_H__
#define V4R_LOCAL_ESTIMATOR_H__

#include <v4r/common/normal_estimator.h>
#include <pcl/surface/mls.h>
#include <v4r/features/uniform_sampling_extractor.h>
#include <v4r/core/macros.h>

namespace v4r
{

template<typename PointT>
class V4R_EXPORTS LocalEstimator
{
protected:
    typedef typename pcl::PointCloud<PointT>::Ptr PointInTPtr;

    pcl::PointCloud<pcl::Normal>::Ptr normals_;
    std::vector<typename boost::shared_ptr<KeypointExtractor<PointT> > > keypoint_extractor_;
    std::vector<int> keypoint_indices_;
    std::string descr_name_;

    void
    computeKeypoints (const pcl::PointCloud<PointT> & cloud, pcl::PointCloud<PointT> & keypoints, const pcl::PointCloud<pcl::Normal>::Ptr & normals)
    {
        keypoint_indices_.clear();
        for (size_t i = 0; i < keypoint_extractor_.size (); i++)
        {
            keypoint_extractor_[i]->setInputCloud (cloud.makeShared());
            if (keypoint_extractor_[i]->needNormals ())
                keypoint_extractor_[i]->setNormals (normals);

            keypoint_extractor_[i]->setSupportRadius (param_.support_radius_);

            pcl::PointCloud<PointT> detected_keypoints;
            keypoint_extractor_[i]->compute (detected_keypoints);

            std::vector<int> kp_indices;
            keypoint_extractor_[i]->getKeypointsIndices(kp_indices);
            keypoint_indices_.insert(keypoint_indices_.end(), kp_indices.begin(), kp_indices.end());
            keypoints += detected_keypoints;
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

    virtual bool acceptsIndices() const
    {
        return false;
    }

    void getKeypointIndices(std::vector<int> & indices) const
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

    /**
    * \brief Right now only uniformSampling keypoint extractor is allowed
    */
    void
    addKeypointExtractor (boost::shared_ptr<KeypointExtractor<PointT> > & ke)
    {
        keypoint_extractor_.push_back (ke);
    }

    void
    setKeypointExtractors (std::vector<typename boost::shared_ptr<KeypointExtractor<PointT> > > & ke)
    {
        keypoint_extractor_ = ke;
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

    std::string getFeatureDescriptorName() const
    {
        return descr_name_;
    }

    virtual bool
    estimate (const pcl::PointCloud<PointT> & in, pcl::PointCloud<PointT> & processed, pcl::PointCloud<PointT> & keypoints, std::vector<std::vector<float> > & signatures)=0;

};
}

#endif
