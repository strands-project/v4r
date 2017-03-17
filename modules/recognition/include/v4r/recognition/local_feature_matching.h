/******************************************************************************
 * Copyright (c) 2012 Aitor Aldoma, Thomas Faeulhammer
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

#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/serialization/serialization.hpp>
#include <flann/flann.h>
#include <glog/logging.h>
#include <pcl/common/common.h>

#include <v4r/common/normals.h>
#include <v4r/common/pcl_visualization_utils.h>
#include <v4r/features/local_estimator.h>
#include <v4r/features/types.h>
#include <v4r/keypoints/keypoint_extractor.h>
#include <v4r/io/filesystem.h>
#include <v4r/recognition/local_rec_object_hypotheses.h>
#include <v4r/recognition/source.h>

namespace v4r
{
class V4R_EXPORTS LocalRecognizerParameter
{
public:
    // parameters for feature matching
    int kdtree_splits_; ///< kdtree splits
    int kdtree_num_trees_; ///< number of trees for FLANN approximate nearest neighbor search
    size_t knn_;  ///< nearest neighbors to search for when checking feature descriptions of the scene
    float max_descriptor_distance_; ///< maximum distance of the descriptor in the respective norm (L1 or L2) to create a correspondence
    float correspondence_distance_weight_; ///< weight factor for correspondences distances. This is done to favour correspondences from different pipelines that are more reliable than other (SIFT and SHOT corr. simultaneously fed into CG)
    int distance_metric_; ///< defines the norm used for feature matching (1... L1 norm, 2... L2 norm, 3... ChiSquare, 4... Hellinger)
    float max_keypoint_distance_z_; ///< maxiumum distance of an extracted keypoint to be accepted

    // parameters for plane filter
    bool filter_planar_; ///< Filter keypoints with a planar surface
    int min_plane_size_; ///< Minimum number of points for a plane to be checked if filter only points above table plane
    int planar_computation_method_; ///< defines the method used to check for planar points. 0... based on curvate value after normalestimationomp, 1... with eigenvalue check of scatter matrix
    float planar_support_radius_; ///< Radius used to check keypoints for planarity.
    float threshold_planar_; ///< threshold ratio used for deciding if patch is planar. Ratio defined as largest eigenvalue to all others.

    // parameters for depth-discontinuity filter
    int filter_border_pts_; ///< Filter keypoints at the boundary (value according to the edge types defined in pcl::OrganizedEdgeBase (EDGELABEL_OCCLUDING  | EDGELABEL_OCCLUDED | EDGELABEL_NAN_BOUNDARY))
    int boundary_width_; ///< Width in pixel of the depth discontinuity

    float required_viewpoint_change_deg_; ///< required viewpoint change in degree for a new training view to be used for feature extraction. Training views will be sorted incrementally by their filename and if the camera pose of a training view is close to the camera pose of an already existing training view, it will be discarded for training.

    bool train_on_individual_views_; ///< if true, extracts features from each view of the object model. Otherwise will use the full 3d cloud

    LocalRecognizerParameter( ) :
          kdtree_splits_ (512),
          kdtree_num_trees_ (4),
          knn_ ( 1 ),
          max_descriptor_distance_ ( std::numeric_limits<float>::max() ),
          correspondence_distance_weight_ ( 1.f ),
          distance_metric_ (1),
          max_keypoint_distance_z_ ( std::numeric_limits<float>::max() ),
          filter_planar_ (false),
          min_plane_size_ (1000),
          planar_support_radius_ (0.04f),
          threshold_planar_ (0.02f),
          filter_border_pts_ (7),
          boundary_width_ (5),
          required_viewpoint_change_deg_ (10.f),
          train_on_individual_views_(true)
    {}

    void
    save(const std::string &filename) const
    {
        std::ofstream ofs(filename);
        boost::archive::xml_oarchive oa(ofs);
        oa << boost::serialization::make_nvp("LocalRecognizerParameter", *this );
        ofs.close();
    }

    LocalRecognizerParameter(const std::string &filename)
    {
        if( !v4r::io::existsFile(filename) )
            throw std::runtime_error("Given config file " + filename + " does not exist! Current working directory is " + boost::filesystem::current_path().string() + ".");

        std::ifstream ifs(filename);
        boost::archive::xml_iarchive ia(ifs);
        ia >> boost::serialization::make_nvp("LocalRecognizerParameter", *this );
        ifs.close();
    }


private:
    friend class boost::serialization::access;
    template<class Archive> V4R_EXPORTS void serialize(Archive & ar, const unsigned int version)
    {
        (void) version;
        ar & BOOST_SERIALIZATION_NVP(kdtree_splits_)
                & BOOST_SERIALIZATION_NVP(kdtree_num_trees_)
                & BOOST_SERIALIZATION_NVP(knn_)
                & BOOST_SERIALIZATION_NVP(max_descriptor_distance_)
                & BOOST_SERIALIZATION_NVP(correspondence_distance_weight_)
                & BOOST_SERIALIZATION_NVP(distance_metric_)
                & BOOST_SERIALIZATION_NVP(max_keypoint_distance_z_)
                & BOOST_SERIALIZATION_NVP(filter_planar_)
                & BOOST_SERIALIZATION_NVP(min_plane_size_)
                & BOOST_SERIALIZATION_NVP(planar_support_radius_)
                & BOOST_SERIALIZATION_NVP(threshold_planar_)
                & BOOST_SERIALIZATION_NVP(filter_border_pts_)
                & BOOST_SERIALIZATION_NVP(boundary_width_)
                & BOOST_SERIALIZATION_NVP(required_viewpoint_change_deg_)
                & BOOST_SERIALIZATION_NVP(train_on_individual_views_)
                ;
    }
};


/**
 * @brief The LocalObjectModel class stores information about the object model related to local feature extraction
 */
class V4R_EXPORTS LocalObjectModel
{
public:
    pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_; ///< all extracted keypoints of the object model
    pcl::PointCloud<pcl::Normal>::Ptr kp_normals_; ///< normals associated to each extracted keypoints of the object model

    LocalObjectModel()
    {
        keypoints_.reset(new pcl::PointCloud<pcl::PointXYZ>);
        kp_normals_.reset(new pcl::PointCloud<pcl::Normal>);
    }

    typedef boost::shared_ptr< LocalObjectModel > Ptr;
    typedef boost::shared_ptr< LocalObjectModel const> ConstPtr;
};

class V4R_EXPORTS LocalObjectModelDatabase
{
public:
    std::map<std::string, typename LocalObjectModel::ConstPtr>  l_obj_models_; ///< information about object models for each model id

    typedef boost::shared_ptr< LocalObjectModelDatabase > Ptr;
    typedef boost::shared_ptr< LocalObjectModelDatabase const> ConstPtr;

    boost::shared_ptr<flann::Index<flann::L1<float> > > flann_index_l1_;
    boost::shared_ptr<flann::Index<flann::L2<float> > > flann_index_l2_;
    boost::shared_ptr<flann::Index<flann::ChiSquareDistance<float> > > flann_index_chisquare_;
    boost::shared_ptr<flann::Index<flann::HellingerDistance<float> > > flann_index_hellinger_;
    boost::shared_ptr<flann::Matrix<float> > flann_data_;

    /**
     * @brief The flann_model class stores for each signature to which model and which keypoint it belongs to
     */
    struct flann_model
    {
        std::string model_id_;
        size_t keypoint_id_;
    };

    std::vector<flann_model> flann_models_;
};

/**
     * \brief Object recognition + 6DOF pose based on local features, GC and HV
     * Contains keypoints/local features computation, matching using FLANN,
     * point-to-point correspondence grouping, pose refinement and hypotheses verification
     * Available features: SHOT, FPFH
     * See apps/3d_rec_framework/tools/apps for usage
     * \author Aitor Aldoma, Federico Tombari, Thomas Faeulhammer
     */

template<typename PointT>
class V4R_EXPORTS LocalFeatureMatcher
{
public:
    LocalRecognizerParameter param_; ///< parameters

private:
    typedef std::vector<float> FeatureDescriptor;
    typedef int KeypointIndex;

    typename pcl::PointCloud<PointT>::ConstPtr scene_; ///< Point cloud to be classified
    std::vector<int> indices_; ///< segmented cloud to be recognized (if empty, all points will be processed)
    pcl::PointCloud<pcl::Normal>::ConstPtr scene_normals_; ///< Point cloud to be classified
    typename Source<PointT>::ConstPtr m_db_;  ///< model data base
    typename NormalEstimator<PointT>::Ptr normal_estimator_;    ///< normal estimator used for computing surface normals (currently only used at training)

    bool have_sift_estimator_; ///< indicates if one of the feature estimator contains SIFT. This is required as this is a special case. For SIFT, keypoint detection and feature description happens at the same time. Therefore it is not allowed to mix with other.

    std::string descr_name_; ///< descriptor name

    std::vector<FeatureDescriptor > scene_signatures_;   ///< signatures extracted from the scene
    std::vector<KeypointIndex> keypoint_indices_;   ///< scene point indices extracted as keypoints
    std::vector<KeypointIndex> keypoint_indices_unfiltered_;    ///< only for visualization

    std::vector<LocalObjectModelDatabase::Ptr > lomdbs_; ///< object model database used for local recognition for each feature estiamtor
    std::map<std::string, LocalObjectHypothesis<PointT> > corrs_; ///< correspondences for each object model (model id, correspondences)

    std::vector<typename LocalEstimator<PointT>::Ptr > estimators_; ///< estimators to compute features/signatures
    std::vector<typename KeypointExtractor<PointT>::Ptr > keypoint_extractor_; ///< set of keypoint extractors

    void mergeKeypointsFromMultipleEstimators(); ///< this puts the model keypoints extracted from multiple feature estimators into a common database

    std::map<std::string, typename LocalObjectModel::ConstPtr> model_keypoints_;

    PCLVisualizationParams::ConstPtr vis_param_;

    void
    validate()
    {
        for (const auto &est : estimators_)
        {
            if(est->getFeatureType() == FeatureType::SIFT_GPU || est->getFeatureType() == FeatureType::SIFT_OPENCV) // for SIFT we do not need to extract keypoints explicitly
            {
                have_sift_estimator_ = true;
                break;
            }
        }
        CHECK( estimators_.size() <= 1 || !have_sift_estimator_) << "SIFT is not allowed to be mixed with other feature descriptors.";

        CHECK( !have_sift_estimator_ || param_.train_on_individual_views_ ) << "SIFT needs organized point clouds. Therefore training from a full model is not supported! " << std::endl;
    }

    bool visualize_keypoints_; ///< if true, visualizes the extracted keypoints

    /**
     * @brief extractKeypoints extracts keypoints from the scene
     * @param[in] region_of_interest object indices (if empty, keypoints will be extracted over whole cloud)
     * @return keypoint indices
     */
    std::vector<KeypointIndex>
    extractKeypoints(const std::vector<int> &region_of_interest = std::vector<int>());

    /**
     * @brief featureMatching matches all scene keypoints with model signatures
     * @param kp_indices query keypoint indices
     * @param signatures query feature descriptors
     * @param lomdb search space
     */
    void
    featureMatching (const std::vector<KeypointIndex> &kp_indices,
                     const std::vector<FeatureDescriptor> &signatures,
                     const LocalObjectModelDatabase::ConstPtr &model_keypoints_,
                     size_t model_keypoint_offset = 0);

    /**
     * @brief featureEncoding describes each keypoint with corresponding feature descriptor
     * @param est feature estimator
     * @param keypoint_indices given keypoint indices
     * @param filtered_keypoint_indices extracted keypoint indices after removing nan points for instance
     * @param signatures feature descriptors
     */
    void
    featureEncoding (LocalEstimator<PointT> &est,
                     const std::vector<KeypointIndex> &keypoint_indices,
                     std::vector<KeypointIndex> &filtered_keypoint_indices,
                     std::vector<FeatureDescriptor > &signatures);


    /**
     * @brief filterKeypoints filters keypoints based on planarity and closeness to depth discontinuity (if according parameters are set)
     * @param[in] input_keypoints keypoints to be filtered
     * @param[inout] input_signatures optional can also filter associated signatures (e.g. for SIFT, keypoint detection and feature description happens in one step - therefore we can only filter after feature description), this variable will update the signatures
     * @return filtered keypoints
     */
    std::vector<int>
    getInlier(const std::vector<KeypointIndex> &input_keypoints) const;

    /**
     * @brief computeFeatures
     * @param est local feature descriptor
     * @return
     */
    bool
    computeFeatures(LocalEstimator<PointT> &est);

    void
    visualizeKeypoints(const std::vector<KeypointIndex> &kp_indices, const std::vector<KeypointIndex> &unfiltered_kp_indices = std::vector<KeypointIndex>()) const;

    std::vector< std::map<std::string, size_t > > model_kp_idx_range_start_; ///< since keypoints are coming from multiple local recognizer, we need to store which range belongs to which recognizer. This variable is the starting parting t

public:

    LocalFeatureMatcher (const LocalRecognizerParameter &p = LocalRecognizerParameter()) :
        param_(p),
        have_sift_estimator_ (false),
        visualize_keypoints_(false)
    { }

    /**
    * @brief getFeatureType
    * @return unique feature type id of estimator
    */
    size_t
    getFeatureType() const
    {
        size_t type=0;
        for( const auto &est : estimators_ )
            type |= est->getFeatureType();

        return type;
    }

//    /**
//    * @brief getFeatureName
//    * @return feature name of estimator
//    */
//    std::string
//    getFeatureName() const
//    {
//        return estimator_->getFeatureDescriptorName();
//    }

    /**
     * @brief getKeypointIndices
     * @param indices of the point clouds that are keypoints
     */
    void
    getKeypointIndices(std::vector<KeypointIndex> &indices) const
    {
        indices = keypoint_indices_;
    }

    /**
    * \brief Sets the local feature estimator
    * \param estimator feature estimator
    */
    void
    addFeatureEstimator (const typename LocalEstimator<PointT>::Ptr & feat)
    {
        estimators_.push_back(feat);
    }

    /**
    * \brief Initializes the FLANN structure from the provided source
    * It does training for the models that havent been trained yet
    * @param training directory
    * @param retrain if set to true, re-trains the object no matter if the data already exists in the given training directory
    */
    void
    initialize(const std::string &trained_dir, bool retrain = false);

    /**
    * @brief adds a keypoint extractor
    * @param keypoint extractor object
    */
    void
    addKeypointExtractor (typename KeypointExtractor<PointT>::Ptr & ke)
    {
        keypoint_extractor_.push_back (ke);
    }

    /**
    * @brief needNormals
    * @return boolean indicating if normals need to be set
    */
    virtual bool
    needNormals() const
    {
        for( const auto &est : estimators_ )
        {
            if (est && est->needNormals())
                return true;
        }


        if (!keypoint_extractor_.empty())
        {
            for (size_t i=0; i<keypoint_extractor_.size(); i++)
            {
                if ( keypoint_extractor_[i]->needNormals() )
                    return true;
            }
        }
        return false;
    }

    /**
    * \brief Performs recognition and pose estimation on the input cloud
    */
    void
    recognize ();

    virtual bool requiresSegmentation() const { return false; }

    /**
    * @brief getLocalObjectModelDatabase
    * @return local object model database
    */
    typename
    std::map<std::string, typename LocalObjectModel::ConstPtr>
    getModelKeypoints() const
    {
        return model_keypoints_;
    }

    /**
    * @brief getCorrespondences
    * @return all extracted correspondences between keypoints of the local object models and the input cloud
    */
    std::map<std::string, LocalObjectHypothesis<PointT>>
    getCorrespondences() const
    {
        return corrs_;
    }


    /**
    * @brief setInputCloud
    * @param cloud to be recognized
    */
    void
    setInputCloud (const typename pcl::PointCloud<PointT>::ConstPtr cloud)
    {
        scene_ = cloud;
    }

    /**
    * @brief setSceneNormals
    * @param normals normals of the input cloud
    */
    void
    setSceneNormals(const pcl::PointCloud<pcl::Normal>::ConstPtr &normals)
    {
        scene_normals_ = normals;
    }

    /**
    * @brief setModelDatabase
    * @param m_db model database
    */
    void
    setModelDatabase(const typename Source<PointT>::ConstPtr &m_db)
    {
        m_db_ = m_db;
    }

    /**
     * @brief setNormalEstimator sets the normal estimator used for computing surface normals (currently only used at training)
     * @param normal_estimator
     */
    void
    setNormalEstimator(const typename NormalEstimator<PointT>::Ptr &normal_estimator)
    {
        normal_estimator_ = normal_estimator;
    }

    void
    setVisualizeKeypoints(bool vis=true)
    {
        visualize_keypoints_ = vis;
    }

    /**
     * @brief setVisualizationParameter sets the PCL visualization parameter (only used if some visualization is enabled)
     * @param vis_param
     */
    void
    setVisualizationParameter(const PCLVisualizationParams::ConstPtr &vis_param)
    {
        vis_param_ = vis_param;
    }


    typedef boost::shared_ptr< LocalFeatureMatcher<PointT> > Ptr;
    typedef boost::shared_ptr< LocalFeatureMatcher<PointT> const> ConstPtr;
};
}
