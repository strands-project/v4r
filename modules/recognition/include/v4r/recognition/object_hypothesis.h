/******************************************************************************
 * Copyright (c) 2016 Thomas Faeulhammer
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


#pragma once

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/vector.hpp>
#include <Eigen/StdVector>
#include <Eigen/Sparse>
#include <pcl/common/common.h>

#include <v4r/core/macros.h>
#include <v4r/common/pcl_serialization.h>
#include <v4r/recognition/model.h>
#include <v4r/recognition/source.h>

namespace v4r
{
template<typename PointT>
class V4R_EXPORTS ObjectHypothesis
{
private:
    friend class boost::serialization::access;

private:
    friend class boost::serialization::access;
    template<class Archive> V4R_EXPORTS void serialize(Archive & ar, const unsigned int version)
    {
        (void) version;
        ar & BOOST_SERIALIZATION_NVP(class_id_)
           & BOOST_SERIALIZATION_NVP(model_id_)
           & BOOST_SERIALIZATION_NVP(transform_)
           & BOOST_SERIALIZATION_NVP(confidence_)
         ;
    }

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef boost::shared_ptr< ObjectHypothesis<PointT> > Ptr;
    typedef boost::shared_ptr< ObjectHypothesis<PointT> const> ConstPtr;

    pcl::Correspondences corr_; ///< local feature matches / keypoint correspondences between model and scene (only for visualization purposes)

    ObjectHypothesis() : class_id_(""), model_id_ ("") {}

    std::string class_id_;  ///< category
    std::string model_id_;  ///< instance
    Eigen::Matrix4f transform_; ///< 4x4 homogenous transformation to project model into camera coordinate system.
    float confidence_; ///< confidence score (coming from feature matching stage)

    virtual ~ObjectHypothesis(){}
};

class V4R_EXPORTS ModelSceneCorrespondence
{
private:
    friend class boost::serialization::access;
    template<class Archive>
    void serialize( Archive & ar, const unsigned int file_version)
    {
        ar & scene_id_ & model_id_ & dist_3D_ & color_distance_ & angle_surface_normals_rad_ & fitness_;
    }

public:
    int scene_id_; /// Index of scene point.
    int model_id_; /// Index of matching model point.
    float dist_3D_; /// Squared distance between the corresponding points in Euclidean 3D space
    float color_distance_; /// Distance between the corresponding points in color
    float angle_surface_normals_rad_; /// Angle in degree between surface normals
    float fitness_; /// model fitness score

    bool operator < (const ModelSceneCorrespondence& other) const
    {
        return this->fitness_ > other.fitness_;
    }

    /** \brief Constructor. */
    ModelSceneCorrespondence () :
        scene_id_ (-1), model_id_ (-1), dist_3D_ (std::numeric_limits<float>::quiet_NaN()),
        color_distance_ (std::numeric_limits<float>::quiet_NaN()), angle_surface_normals_rad_(M_PI/2),
        fitness_ (0.f)
    {}
};


template<typename PointT>
class V4R_EXPORTS HVRecognitionModel : public ObjectHypothesis<PointT>
{
private:
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & boost::serialization::base_object<ObjectHypothesis>(*this);
    }

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef boost::shared_ptr< HVRecognitionModel> Ptr;
    typedef boost::shared_ptr< HVRecognitionModel const> ConstPtr;

    typename pcl::PointCloud<PointT>::Ptr complete_cloud_;
    typename pcl::PointCloud<PointT>::Ptr visible_cloud_;
    std::vector<boost::dynamic_bitset<> > image_mask_; ///< image mask per view (in single-view case, there will be only one element in outer vector). Used to compute pairwise intersection
    pcl::PointCloud<pcl::Normal>::Ptr visible_cloud_normals_;
    pcl::PointCloud<pcl::Normal>::Ptr complete_cloud_normals_;
    std::vector<int> visible_indices_;  ///< visible indices computed by z-Buffering (for model self-occlusion) and occlusion reasoning with scene cloud
    std::vector<int> visible_indices_by_octree_; ///< visible indices computed by creating an octree for the model and checking which leaf nodes are occupied by a visible point computed from the z-buffering approach
    std::vector<ModelSceneCorrespondence> model_scene_c_; ///< correspondences between visible model points and scene
    float model_fit_; ///< the fitness score of the visible cloud to the model scene (sum of model_scene_c correspondenes weight divided by the number of visible points)
    boost::dynamic_bitset<> visible_pt_is_outlier_; ///< indicates for each visible point if it is considered an outlier

    Eigen::MatrixXf pt_color_;  ///< color values for each point of the (complete) model (row_id). Width is equal to the number of color channels
    float mean_brigthness_;   ///< average value of the L channel for all visible model points
    float mean_brigthness_scene_;   ///< average value of the L channel for all scene points close to the visible model points
    std::vector<int> scene_indices_in_crop_box_; ///< indices of the scene that are occupied from the bounding box of the (complete) hypothesis
    float L_value_offset_; ///< the offset being added to the computed L color values to compensate for different lighting conditions

    Eigen::Matrix4f refined_pose_;  ///< refined pose after ICP (do be multiplied by the initial transform)
    Eigen::SparseVector<float> scene_explained_weight_;   ///< stores for each scene point how well it is explained by the visible model points

    bool rejected_due_to_low_visibility_;   ///< true if the object model rendered in the view is not visible enough
    bool is_outlier_;    ///< true if the object model is not able to explain the scene well enough
    bool rejected_due_to_better_hypothesis_in_group_; ///< true if there is any other object model in the same hypotheses group which explains the scene better
    bool rejected_globally_;

    HVRecognitionModel() :
        L_value_offset_ (0.f),
        refined_pose_ ( Eigen::Matrix4f::Identity() ),
        rejected_due_to_low_visibility_ (false),
        is_outlier_ (false),
        rejected_due_to_better_hypothesis_in_group_ (false),
        rejected_globally_ (false)
    {}

    HVRecognitionModel(const ObjectHypothesis<PointT> &oh) :
        ObjectHypothesis<PointT>(oh),
        L_value_offset_ (0.f),
        refined_pose_ ( Eigen::Matrix4f::Identity() ),
        rejected_due_to_low_visibility_ (false),
        is_outlier_ (false),
        rejected_due_to_better_hypothesis_in_group_ (false),
        rejected_globally_ (false)
    {}

    void
    freeSpace()
    {
        complete_cloud_.reset();
        visible_cloud_.reset();
        visible_cloud_normals_.reset();
        complete_cloud_normals_.reset();
        visible_indices_.clear();
        image_mask_.clear();
        model_scene_c_.clear();
        pt_color_.resize(0,0);
        scene_indices_in_crop_box_.clear();
    }

    bool
    isRejected() const
    {
        return is_outlier_ || rejected_due_to_low_visibility_  || rejected_globally_ || rejected_due_to_better_hypothesis_in_group_;
    }

    /**
         * @brief does dilation and erosion on the occupancy image of the rendered point cloud
         * @param do_smoothing
         * @param smoothing_radius
         * @param do_erosion
         * @param erosion_radius
         * @param img_width
         */
    void
    processSilhouette(bool do_smoothing=true, int smoothing_radius=2, bool do_erosion=true, int erosion_radius=4, int img_width=640);

    static
    bool modelFitCompare(typename HVRecognitionModel<PointT>::Ptr const & a, typename HVRecognitionModel<PointT>::Ptr const & b)
    {
        return a->model_fit_ > b->model_fit_;
    }
};


template<typename PointT>
class V4R_EXPORTS ObjectHypothesesGroup
{
private:
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        (void)version;
        ar & global_hypotheses_ & ohs_;
//        size_t nItems = ohs_.size();
//        ar & nItems;
//        for (const auto &oh : ohs_) {  ar & *oh; }
    }

public:
    typedef boost::shared_ptr< ObjectHypothesesGroup > Ptr;
    typedef boost::shared_ptr< ObjectHypothesesGroup const> ConstPtr;

    ObjectHypothesesGroup() {}
    ObjectHypothesesGroup(const std::string &filename, const Source<PointT> &src);
    void save(const std::string &filename) const;

    std::vector<typename ObjectHypothesis<PointT>::Ptr > ohs_; ///< Each hypothesis can have several object model (e.g. global recognizer tries to macht several object instances for a clustered point cloud segment).
    bool global_hypotheses_; ///< if true, hypothesis was generated by global recognition pipeline. Otherwise, from local feature matches-
};
}
