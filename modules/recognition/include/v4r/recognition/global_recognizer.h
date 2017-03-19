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
#pragma once


#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/serialization/serialization.hpp>

#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <v4r/core/macros.h>
#include <v4r/common/normal_estimator.h>
#include <v4r/common/pcl_serialization.h>
#include <v4r/common/pcl_visualization_utils.h>
#include <v4r/features/global_estimator.h>
#include <v4r/io/filesystem.h>
#include <v4r/ml/classifier.h>
#include <v4r/recognition/object_hypothesis.h>
#include <v4r/recognition/source.h>

namespace v4r
{

class V4R_EXPORTS GlobalRecognizerParameter
{
public:
    bool check_elongations_; ///< if true, checks if the elongation of the segmented cluster fits approximately the elongation of the matched object hypothesis
    float max_elongation_ratio_; ///< if the elongation of the segment w.r.t. to the matched hypotheses is above this threshold, it will be rejected (used only if check_elongations_ is true.
    float min_elongation_ratio_; ///< if the elongation of the segment w.r.t. to the matched hypotheses is below this threshold, it will be rejected (used only if check_elongations_ is true).
    bool use_table_plane_for_alignment_; ///< if true, aligns the matched object model such that the centroid corresponds to the centroid of the segmented cluster downprojected onto the found table plane. The z-axis corresponds to the normal axis of the table plane and the remaining axis build a orthornmal system. Rotation is then sampled in equidistant angles around the z-axis. ATTENTION: This assumes the models are in a coordinate system with the z-axis alinging with the typical upright position of the object.
    float z_angle_sampling_density_degree_; ///< if use_table_plane_for_alignment_, this value will generate object hypotheses at each multiple of this value.
    float required_viewpoint_change_deg_; ///< required viewpoint change in degree for a new training view to be used for feature extraction. Training views will be sorted incrementally by their filename and if the camera pose of a training view is close to the camera pose of an already existing training view, it will be discarded for training.
    bool estimate_pose_; ///< if true, tries to estimate a coarse pose of the object based on the other parameters
    bool classify_instances_; ///< if true, classifier learns to distinguish between model instances instead of categories

    GlobalRecognizerParameter(
            bool check_elongations = true,
            float max_elongation_ratio = 1.2f,
            float min_elongation_ratio = 0.5f,  // lower because of possible occlusion
            bool use_table_plane_for_alignment = false,
            float z_angle_sampling_density_degree = 60.f,
            float required_viewpoint_change_deg = 0.f,
            bool estimate_pose = false,
            bool classify_instances = false
            )
        : check_elongations_ ( check_elongations ),
          max_elongation_ratio_ ( max_elongation_ratio ),
          min_elongation_ratio_ ( min_elongation_ratio ),
          use_table_plane_for_alignment_ ( use_table_plane_for_alignment ),
          z_angle_sampling_density_degree_ ( z_angle_sampling_density_degree ),
          required_viewpoint_change_deg_ (required_viewpoint_change_deg),
          estimate_pose_ (estimate_pose),
          classify_instances_ (classify_instances)
    {}

    friend class boost::serialization::access;
    template<class Archive> V4R_EXPORTS void serialize(Archive & ar, const unsigned int version)
    {
      ar & BOOST_SERIALIZATION_NVP(check_elongations_)
         & BOOST_SERIALIZATION_NVP(max_elongation_ratio_)
         & BOOST_SERIALIZATION_NVP(min_elongation_ratio_)
         & BOOST_SERIALIZATION_NVP(use_table_plane_for_alignment_)
         & BOOST_SERIALIZATION_NVP(z_angle_sampling_density_degree_)
         & BOOST_SERIALIZATION_NVP(required_viewpoint_change_deg_)
         & BOOST_SERIALIZATION_NVP(estimate_pose_)
         & BOOST_SERIALIZATION_NVP(classify_instances_)
        ;
        (void) version;
    }

    void
    save(const std::string &filename) const
    {
        std::ofstream ofs(filename);
        boost::archive::xml_oarchive oa(ofs);
        oa << boost::serialization::make_nvp("GlobalRecognizerParameter", *this );
        ofs.close();
    }

    GlobalRecognizerParameter(const std::string &filename)
    {
        if( !v4r::io::existsFile(filename) )
            throw std::runtime_error("Given config file " + filename + " does not exist! Current working directory is " + boost::filesystem::current_path().string() + ".");

        std::ifstream ifs(filename);
        boost::archive::xml_iarchive ia(ifs);
        ia >> boost::serialization::make_nvp("GlobalRecognizerParameter", *this );
        ifs.close();
    }
};

/**
 * @brief The GlobalObjectModel class stores information about the object model related to global feature extraction
 * @author Thomas Faeulhammer
 */
class V4R_EXPORTS GlobalObjectModel
{
public:
    Eigen::MatrixXf model_signatures_; ///< signatures (from all views) of the object model
    Eigen::MatrixX3f model_elongations_; ///< spatial elongations in X,Y and Z direction (from all views)
    Eigen::MatrixX4f model_centroids_;  ///< model centroids (from all views)
    float mean_distance_view_centroid_to_3d_model_centroid_; ///< average distance of the centroids computed on the training views to the centroid computed on the whole 3D object
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > model_poses_; ///< model poses (from all views)
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > eigen_based_pose_; ///< poses (from all views) that transform view such that principial axis correspond to x,y and z axis

    // for OURCVFH
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > descriptor_transforms_;

    GlobalObjectModel() :
        mean_distance_view_centroid_to_3d_model_centroid_(0.f)
    {}

    friend class boost::serialization::access;
    template<class Archive> V4R_EXPORTS void serialize(Archive & ar, const unsigned int version)
    {
        ar & model_signatures_
           & model_elongations_
           & model_centroids_
           & model_poses_
           & eigen_based_pose_
           & mean_distance_view_centroid_to_3d_model_centroid_
           & descriptor_transforms_
         ;
        (void) version;
    }

    typedef boost::shared_ptr< GlobalObjectModel > Ptr;
    typedef boost::shared_ptr< GlobalObjectModel const> ConstPtr;
};

class V4R_EXPORTS GlobalObjectModelDatabase
{
public:
    typedef boost::shared_ptr< GlobalObjectModelDatabase > Ptr;
    typedef boost::shared_ptr< GlobalObjectModelDatabase const> ConstPtr;

    Eigen::MatrixXf all_model_signatures_;
    Eigen::VectorXi all_trained_model_label_;

    std::map<std::string, GlobalObjectModel::ConstPtr> global_models_;

    /**
     * @brief The flann_model struct is neccessary to know which signature id belongs to which view
     */
    struct flann_model
    {
        std::string instance_name_;
        std::string class_name_;
        size_t view_id_;
    };

    std::vector<flann_model> flann_models_;
};

/**
 *      @brief class for object classification based on global descriptors (and segmentation)
 *      @date Nov., 2015
 *      @author Thomas Faeulhammer
 */
template<typename PointT>
class V4R_EXPORTS GlobalRecognizer
{
public:
    GlobalRecognizerParameter param_;

    /**
     * @brief The Cluster class represents the properties of a point cloud cluster that needs to be segmented
     */
    class Cluster
    {
        bool table_plane_set_;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        std::vector<int> indices_; ///< segmented cloud to be recognized (if empty, all points will be processed)
        Eigen::Vector4f table_plane_; ///< extracted table plane of input cloud (if used for pose estimation)
        Eigen::Vector4f centroid_; ///< centroid of cluster
        Eigen::Vector3f elongation_; ///< elongations along the principal component of cluster
        Eigen::Matrix4f eigen_pose_alignment_; ///< basis matrix of cluster
        Eigen::Matrix3f eigen_basis_;

        /**
         * @brief Cluster represents a segment from a point cloud
         * @param cloud full point cloud
         * @param indices indices that belong to the segment
         * @param compute_properties if true, computes centroid, principal components and transformation to align with principal axis
         */
        Cluster(const pcl::PointCloud<PointT> &cloud, const std::vector<int> &indices = std::vector<int>(), bool compute_properties = true)
            : table_plane_set_(false), indices_(indices)
        {
            if(compute_properties)
            {
                EIGEN_ALIGN16 Eigen::Matrix3f covariance_matrix;
                EIGEN_ALIGN16 Eigen::Vector3f eigenValues;
                EIGEN_ALIGN16 Eigen::Matrix3f eigenVectors;

                if(indices.empty())
                    computeMeanAndCovarianceMatrix ( cloud, covariance_matrix, centroid_ );
                else
                    computeMeanAndCovarianceMatrix ( cloud, indices, covariance_matrix, centroid_ );

                pcl::eigen33 (covariance_matrix, eigenVectors, eigenValues);

                // create orthonormal rotation matrix from eigenvectors
                eigen_basis_.col(0) = eigenVectors.col(0).normalized();
                float dotp12 = eigenVectors.col(1).dot(eigen_basis_.col(0));
                Eigen::Vector3f eig2 = eigenVectors.col(1) - dotp12 * eigen_basis_.col(0);
                eigen_basis_.col(1) = eig2.normalized();
                Eigen::Vector3f eig3 = eigen_basis_.col(0).cross ( eigen_basis_.col(1) );
                eigen_basis_.col(2) = eig3.normalized();

                // transform cluster into origin and align with eigenvectors
                Eigen::Matrix4f tf_rot = Eigen::Matrix4f::Identity();
                tf_rot.block<3,3>(0,0) = eigen_basis_.transpose();
                Eigen::Matrix4f tf_trans = Eigen::Matrix4f::Identity();
                tf_trans.block<3,1>(0,3) = -centroid_.topRows(3);
                eigen_pose_alignment_ = tf_rot * tf_trans;

                // compute max elongations
                pcl::PointCloud<PointT> eigenvec_aligned;

                if(indices.empty())
                    pcl::copyPointCloud( cloud, eigenvec_aligned);
                else
                    pcl::copyPointCloud( cloud, indices, eigenvec_aligned);

                pcl::transformPointCloud(eigenvec_aligned, eigenvec_aligned, eigen_pose_alignment_);

                float xmin,ymin,xmax,ymax,zmin,zmax;
                xmin = ymin = xmax = ymax = zmin = zmax = 0.f;
                for(size_t pt=0; pt<eigenvec_aligned.points.size(); pt++)
                {
                    const PointT &p = eigenvec_aligned.points[pt];
                    if(p.x < xmin)
                        xmin = p.x;
                    if(p.x > xmax)
                        xmax = p.x;
                    if(p.y < ymin)
                        ymin = p.y;
                    if(p.y > ymax)
                        ymax = p.y;
                    if(p.z < zmin)
                        zmin = p.z;
                    if(p.z > zmax)
                        zmax = p.z;
                }

                elongation_(0) = xmax - xmin;
                elongation_(1) = ymax - ymin;
                elongation_(2) = zmax - zmin;
            }
        }

        /**
         * @brief setTablePlane
         * @param table_plane
         */
        void
        setTablePlane(const Eigen::Vector4f &table_plane)
        {
            table_plane_ = table_plane;
            table_plane_set_ = true;
        }

        bool
        isTablePlaneSet() const { return table_plane_set_; }

        typedef boost::shared_ptr< Cluster > Ptr;
        typedef boost::shared_ptr< Cluster const> ConstPtr;
    };

private:
    typedef Model<PointT> ModelT;

    typename pcl::PointCloud<PointT>::ConstPtr scene_; ///< Point cloud to be classified
    pcl::PointCloud<pcl::Normal>::ConstPtr scene_normals_; ///< Point cloud to be classified
    typename Source<PointT>::ConstPtr m_db_;  ///< model data base

    typename Cluster::Ptr cluster_;  ///< cluster to be classified

    GlobalObjectModelDatabase gomdb_;   ///< database used for global recognition

    typename NormalEstimator<PointT>::Ptr normal_estimator_;    ///< normal estimator used for computing surface normals (currently only used at training)

    std::vector<std::string> id_to_model_name_; ///< which target label (target id = vector element id) of the classifier corresponds to which object model

    std::vector<typename ObjectHypothesis<PointT>::Ptr> obj_hyps_filtered_; ///<  extracted object hypotheses after running through (potential) filter
    std::vector<typename ObjectHypothesis<PointT>::Ptr> all_obj_hyps_; ///< all extracted object hypotheses

    std::vector<std::string> categories_;   ///< classification results
    std::vector<float> confidences_;   ///< confidences associated to the classification results (normalized to 0...1)
    typename GlobalEstimator<PointT>::Ptr estimator_; ///< estimator used for describing the object
    Classifier::Ptr classifier_; ///< classifier object

    void featureEncodingAndMatching();

    bool keep_all_hypotheses_;

    PCLVisualizationParams::ConstPtr vis_param_;

    void
    validate() const;


public:
    GlobalRecognizer(const GlobalRecognizerParameter &p = GlobalRecognizerParameter()) :
        param_(p),
        keep_all_hypotheses_(true)
    {}

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

    std::string
    getFeatureName() const
    {
        return estimator_->getFeatureDescriptorName();
    }

    /**
     * @brief setClassifier
     * @param classifier
     */
    void
    setClassifier(const Classifier::Ptr &classifier)
    {
        classifier_ = classifier;
    }

    /**
     * @brief getHypotheses
     * @return generated object hypotheses
     */
    std::vector<typename ObjectHypothesis<PointT>::Ptr >
    getHypotheses()
    {
        return obj_hyps_filtered_;
    }


    virtual void
    initialize(const std::string &trained_dir, bool retrain);

    /**
     * @brief needNormals
     * @return
     */
    bool
    needNormals() const
    {
        return estimator_->needNormals();
    }

    /**
     * @brief getFeatureType
     * @return
     */
    size_t
    getFeatureType() const
    {
        return estimator_->getFeatureType();
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
     * @brief getObjectHypothesis
     * @return generated (potentiallly filtered) object hypothesis
     */
    std::vector<typename ObjectHypothesis<PointT>::Ptr >
    getFilteredHypotheses() const
    {
        return obj_hyps_filtered_;
    }

    /**
     * @brief getObjectHypothesis
     * @return generated object hypothesis
     */
    std::vector<typename ObjectHypothesis<PointT>::Ptr >
    getAllHypotheses() const
    {
        return all_obj_hyps_;
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
     * @brief setIndices
     * @param indices indices of the input cloud that are to be described and matched
     */
    void
    setCluster( const typename Cluster::Ptr &cluster)
    {
        cluster_ = cluster;
    }

    /**
     * @brief recognize
     */
    void
    recognize ();

    /**
     * @brief setVisualizationParameter
     * @param vis_param
     */
    void
    setVisualizationParameter(const PCLVisualizationParams::ConstPtr &vis_param)
    {
        vis_param_ = vis_param;
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


    typedef boost::shared_ptr< GlobalRecognizer<PointT> > Ptr;
    typedef boost::shared_ptr< GlobalRecognizer<PointT> const> ConstPtr;
};
}
