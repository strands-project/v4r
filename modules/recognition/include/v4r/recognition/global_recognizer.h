/******************************************************************************
 * Copyright (c) 2015 Thomas Faeulhammer
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
#ifndef V4R_GLOBAL_RECOGNIZER_H__
#define V4R_GLOBAL_RECOGNIZER_H__

#include <pcl/common/common.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <v4r/core/macros.h>
#include <v4r/features/global_estimator.h>
#include <v4r/ml/classifier.h>
#include <v4r/segmentation/segmenter.h>
#include <v4r/recognition/recognizer.h>

namespace v4r
{

/**
 *      @brief class for object classification based on global descriptors (and segmentation)
 *      @date Nov., 2015
 *      @author Thomas Faeulhammer
 */
template<typename PointT>
class V4R_EXPORTS GlobalRecognizer : public Recognizer<PointT> {

public:
    class V4R_EXPORTS Parameter : public Recognizer<PointT>::Parameter
    {
    public:
        using Recognizer<PointT>::Parameter::normal_computation_method_;

        int kdtree_splits_;
        size_t knn_;  /// @brief nearest neighbors to search for when checking feature descriptions of the scene
        int distance_metric_; /// @brief defines the norm used for feature matching (1... L1 norm, 2... L2 norm)

        bool filter_border_pts_; /// @brief Filter keypoints at the boundary
        int boundary_width_; /// @brief Width in pixel of the depth discontinuity
        bool visualize_clusters_;
        bool check_elongations_; /// @brief if true, checks if the elongation of the segmented cluster fits approximately the elongation of the matched object hypothesis
        float max_elongation_ratio_; /// @brief if the elongation of the segment w.r.t. to the matched hypotheses is above this threshold, it will be rejected (used only if check_elongations_ is true.
        float min_elongation_ratio_; /// @brief if the elongation of the segment w.r.t. to the matched hypotheses is below this threshold, it will be rejected (used only if check_elongations_ is true).
        bool use_table_plane_for_alignment_; /// @brief if true, aligns the matched object model such that the centroid corresponds to the centroid of the segmented cluster downprojected onto the found table plane. The z-axis corresponds to the normal axis of the table plane and the remaining axis build a orthornmal system. Rotation is then sampled in equidistant angles around the z-axis.
        float z_angle_sampling_density_degree_; /// @brief if use_table_plane_for_alignment_, this value will generate object hypotheses at each multiple of this value.
        int icp_iterations_; /// @brief number of ICP iterations to align the estimated object pose to the scene. If 0, no pose refinement.

        Parameter(
                int kdtree_splits = 512,
                size_t knn = 1,
                int distance_metric = 2,
                bool filter_border_pts = false,
                int boundary_width = 5,
                bool visualize_clusters = false,
                bool check_elongations = true,
                float max_elongation_ratio = 1.2f,
                float min_elongation_ratio = 0.5f,  // lower because of possible occlusion
                bool use_table_plane_for_alignment = false,
                float z_angle_sampling_density_degree = 60.f,
                int icp_iterations = 5
                )
            : Recognizer<PointT>::Parameter(),
              kdtree_splits_ (kdtree_splits),
              knn_ ( knn ),
              distance_metric_ (distance_metric),
              filter_border_pts_ (filter_border_pts),
              boundary_width_ (boundary_width),
              visualize_clusters_ ( visualize_clusters ),
              check_elongations_ ( check_elongations ),
              max_elongation_ratio_ ( max_elongation_ratio ),
              min_elongation_ratio_ ( min_elongation_ratio ),
              use_table_plane_for_alignment_ ( use_table_plane_for_alignment ),
              z_angle_sampling_density_degree_ ( z_angle_sampling_density_degree ),
              icp_iterations_ (icp_iterations)
        {}
    }param_;

private:
    typedef Model<PointT> ModelT;
    typedef boost::shared_ptr<ModelT> ModelTPtr;
    typedef typename pcl::PointCloud<PointT>::Ptr PointTPtr;

    mutable pcl::visualization::PCLVisualizer::Ptr vis_;
    mutable int vp1_, vp2_, vp3_, vp4_, vp5_;
    mutable std::vector<std::string> coordinate_axis_ids_global_;
    void visualize();

    std::vector<ObjectHypothesesGroup<PointT> > obj_hypotheses_wo_elongation_check_; /// @brief just for visualization (to see effect of elongation check)

protected:

    using Recognizer<PointT>::scene_;
    using Recognizer<PointT>::scene_normals_;
    using Recognizer<PointT>::models_dir_;
    using Recognizer<PointT>::obj_hypotheses_;
    using Recognizer<PointT>::indices_;
    using Recognizer<PointT>::source_;
    using Recognizer<PointT>::requires_segmentation_;

//    std::string training_dir_;  /// @brief directory containing training data
    std::vector<std::string> categories_;   /// @brief classification results
    std::vector<float> confidences_;   /// @brief confidences associated to the classification results (normalized to 0...1)
    typename GlobalEstimator<PointT>::Ptr estimator_; /// @brief estimator used for describing the object
    typename Segmenter<PointT>::Ptr seg_;
    Classifier::Ptr classifier_;
    std::vector<pcl::PointIndices> clusters_;
    std::vector<Eigen::Vector4f,  Eigen::aligned_allocator<Eigen::Vector4f> > centroids_; /// @brief centroids of each cluster
    std::vector<Eigen::Vector3f,  Eigen::aligned_allocator<Eigen::Vector3f> > elongations_; /// @brief stores elongations along the principal components of each cluster
    std::vector<Eigen::Matrix3f,  Eigen::aligned_allocator<Eigen::Matrix3f> > eigen_basis_; /// @brief the eigen basis matrix for each cluster

    Eigen::MatrixXf signatures_; /// @brief signatures for each cluster (clusters are equal to columns, dimension of signature equal rows)
    Eigen::Vector4f table_plane_; /// @brief extracted table plane

    void computeEigenBasis();
    bool featureEncoding(Eigen::MatrixXf &signatures);
    void featureMatching(const Eigen::MatrixXf &query_sig, int cluster_id, ObjectHypothesesGroup<PointT> &oh);
    void loadFeaturesFromDisk();
    void poseRefinement();

    // FLANN stuff for nearest neighbor search
    class flann_model
    {
    public:
      ModelTPtr model;
      size_t view_id;
    };

    std::vector<flann_model> flann_models_;
    boost::shared_ptr<flann::Matrix<float> > flann_data_;
    Eigen::MatrixXf all_model_signatures_;
    Eigen::VectorXi all_trained_model_label_;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    GlobalRecognizer(const Parameter &p = Parameter()) : Recognizer<PointT>(), param_(p)
    {
        requires_segmentation_ = true;
    }

    virtual ~GlobalRecognizer(){}

//    void
//    setTrainingDir (const std::string & dir)
//    {
//        training_dir_ = dir;
//    }

    void
    getCategory (std::vector<std::string> & categories) const
    {
        categories = categories_;
    }

    void
    getConfidence (std::vector<float> & conf) const
    {
        conf = confidences_;
    }

    void
    setFeatureEstimator (const typename GlobalEstimator<PointT>::Ptr & feat)
    {
        estimator_ = feat;
    }

    void
    setSegmentationAlgorithm(const typename Segmenter<PointT>::Ptr &seg)
    {
        seg_ = seg;
    }

    void
    setClassifier(const Classifier::Ptr &classifier)
    {
        classifier_ = classifier;
    }


    /**
     * @brief trains the object models
     * @param force_retrain (if true, re-trains the object model even if the folder already exists)
     * @return success
     */
    virtual bool
    initialize(bool force_retrain = false);

    bool
    needNormals() const
    {
        return estimator_->needNormals();
    }

    size_t
    getFeatureType() const
    {
        return estimator_->getFeatureType();
    }

    void
    recognize ();


    typedef boost::shared_ptr< GlobalRecognizer<PointT> > Ptr;
    typedef boost::shared_ptr< GlobalRecognizer<PointT> const> ConstPtr;
};
}

#endif
