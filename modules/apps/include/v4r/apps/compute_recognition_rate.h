#pragma once

#include <v4r/core/macros.h>
#include <v4r/common/pcl_visualization_utils.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <iostream>
#include <sstream>
#include <fstream>

#define BOOST_NO_CXX11_SCOPED_ENUMS
#include <boost/filesystem.hpp>
#undef BOOST_NO_CXX11_SCOPED_ENUMS

namespace bf = boost::filesystem;

namespace v4r
{
namespace apps
{


struct Hypothesis
{
    Eigen::Matrix4f pose;
    float occlusion;
};


class V4R_EXPORTS RecognitionEvaluator
{
private:
    typedef pcl::PointXYZRGB PointT;

    struct Model
    {
        pcl::PointCloud<PointT>::Ptr cloud;
        Eigen::Vector4f centroid;
        bool is_rotation_invariant_ = false; // in-plane (x/y)
        bool is_rotational_symmetric_ = false;
    };

    float rotation_error_threshold_deg;
    float translation_error_threshold_m;
    float occlusion_threshold;
    std::vector<std::string> coordinate_system_ids_;
    std::string out_dir;
    std::string gt_dir;
    std::string or_dir;
    std::string models_dir;
    std::string test_dir;
    bool visualize_;
    bool use_generated_hypotheses;

    pcl::visualization::PCLVisualizer::Ptr vis_;
    int vp1_, vp2_, vp3_;

    std::map<std::string, Model> models;

    PCLVisualizationParams::Ptr vis_params_;

    void loadModels();

    std::vector<std::string> rotational_symmetric_objects_; ///< name of objects which are rotational symmetric with respect to xy plane
    std::vector<std::string> rotational_invariant_objects_; ///< name of objects which are rotational invariant with respect to xy plane

public:
    RecognitionEvaluator()
        :
          rotation_error_threshold_deg(30.f),
          translation_error_threshold_m(0.05f),
          occlusion_threshold(0.95f),
          out_dir("/tmp/recognition_rates/"),
          visualize_(false),
          use_generated_hypotheses(false)
    {
        rotational_invariant_objects_ = {
            "toilet_paper", "red_mug_white_spots", "muller_milch_shoko", "muller_milch_banana", "coffee_container",
            "object_11", "object_19", "object_29", "object_32",
            "object_18", "object_22", "object_23", "object_25", "object_27", "object_28" //last row are debatable objects - not completely invariant but very hard to distinguish
        };

        rotational_symmetric_objects_ = {
            "jasmine_green_tea", "fruchtmolke", "asus_box",
            "object_10", "object_26", "object_35",
            "object_01", "object_02", "object_03", "object_09", "object_08", "object_30", "object_31", "object_33" //last row are debatable objects - not 100% symmetric but very hard to distinguish

        };

        vis_params_.reset(new PCLVisualizationParams());
        vis_params_->bg_color_ = Eigen::Vector3i(255,255,255);
        vis_params_->coordinate_axis_scale_ = 0.04f;
    }

    /**
     * @brief init set directories and stuff from (boost) console arguments
     * @param params parameters (boost program options)
     * @param unused parameters
     */
    std::vector<std::string>
    init(const std::vector<std::string> &params);

    // =======  DECLARATIONS ===================
    /**
     * @brief computeError compute translation error for a given pose
     * @param[in] pose_a
     * @param[in] pose_b
     * @param[in] centroid of object model in model coordinate system
     * @param[out] trans_error translation error
     * @param[out] rot_error rotation error
     * @param[in] true if model is invariant for rotation around z
     * @param[in] true if model is symmetric for rotation around z (180deg periodic)
     * @return boolean indicitating if error is outside given threshold
     */
    bool computeError(const Eigen::Matrix4f &pose_a, const Eigen::Matrix4f &pose_b, const Eigen::Vector4f &centroid_model,
                      float &trans_error, float &rot_error, bool is_rotation_invariant , bool is_rotational_symmetric );



    /**
     * @brief checkMatchvector check a certain match configuration
     * @param[in] rec2gt matches indicated by first index corresponding
     * to model id and second index to ground-truth id
     * @param[in] rec_hyps recognition hypotheses
     * @param[in] gt_hyps ground-truth hypotheses
     * @param[in] centroid of object model in model coordinate system
     * @param[out] sum_translation_error
     * @param[out] sum_rotational_error
     * @param[out] tp true positives
     * @param[out] fp flase positives
     * @param[out] fn false negatives
     * @param[in] true if model is invariant for rotation around z
     * @param[in] true if model is symmetric for rotation around z (180deg periodic)
     */
    void checkMatchvector(const std::vector< std::pair<int, int> > &rec2gt,
                          const std::vector<Hypothesis> &rec_hyps,
                          const std::vector<Hypothesis> &gt_hyps,
                          const Eigen::Vector4f &model_centroid,
                          double &sum_translation_error,
                          double &sum_rotational_error,
                          size_t &tp, size_t &fp, size_t &fn,
                          bool is_rotation_invariant , bool is_rotational_symmetric );

    /**
     * @brief selectBestMatch computes the best matches for a set of hypotheses. This
     * tackles the problem when multiple object instances are present in the scene and
     * the question is which ground-truth object belongs to which recognized object. By
     * permuating through all possible solutions, it finds the match which gives the
     * best f-score taking into account to be neglected hypotheses due to occlusion.
     * @param rec_hyps recognized hypotheses
     * @param gt_hyps ground-truth hypotheses
     * @param centroid of object model in model coordinate system
     * @param tp true positives for best match
     * @param fp false positives for best match
     * @param fn false negatives for best match
     * @param sum_translation_error accumulated translation error for best match
     * @param sum_rotational_error accumulated rotational error for best match
     * @param[in] true if model is invariant for rotation around z
     * @param[in] true if model is symmetric for rotation around z (180deg periodic)
     * @return best match for the given hypotheses. First index corresponds to element in
     * given recognition hypothesis, second index to ground-truth hypothesis
     */
    std::vector<std::pair<int, int> > selectBestMatch(const std::vector<Hypothesis> &rec_hyps,
                                                      const std::vector<Hypothesis> &gt_hyps,
                                                      const Eigen::Vector4f &model_centroid,
                                                      size_t &tp, size_t &fp, size_t &fn,
                                                      double &sum_translation_error,
                                                      double &sum_rotational_error,
                                                      bool is_rotation_invariant , bool is_rotational_symmetric );

    std::map<std::string, std::vector<Hypothesis> >
    readHypothesesFromFile( const std::string &filename );



    /**
     * @brief setRotationalInvariantObjects
     * @param rot_invariant_objects name of objects which are rotational invariant with respect to xy plane
     */
    void
    setRotationalInvariantObjects(const std::vector<std::string> &rot_invariant_objects)
    {
        rotational_invariant_objects_ = rot_invariant_objects;
    }

    /**
     * @brief setRotationalSymmetricObjects
     * @param rotational_symmetric_objects name of objects which are rotational symmetric with respect to xy plane
     */
    void
    setRotationalSymmetricObjects(const std::vector<std::string> &rotational_symmetric_objects)
    {
        rotational_symmetric_objects_ = rotational_symmetric_objects;
    }

    void compute_recognition_rate (size_t &total_tp, size_t &total_fp, size_t &total_fn);
    float compute_recognition_rate_over_occlusion (); ///< basically checks for each ground-truth object if there exists a corresponding recognized object
    void checkIndividualHypotheses(); ///< check for each recognized object if there is a corresponding ground-truth object<w

    std::string getModels_dir() const;
    void setModels_dir(const std::string &value);
    std::string getTest_dir() const;
    void setTest_dir(const std::string &value);
    std::string getOr_dir() const;
    void setOr_dir(const std::string &value);
    std::string getGt_dir() const;
    void setGt_dir(const std::string &value);
    bool getUse_generated_hypotheses() const;
    void setUse_generated_hypotheses(bool value = true);
    bool getVisualize() const;
    void setVisualize(bool value);
    std::string getOut_dir() const;
    void setOut_dir(const std::string &value);

    /**
     * @brief visualize results for an input cloud with ground-truth and recognized object models
     * @param input_cloud cloud of the input scene
     * @param gt_path file path to the ground-truth annotation file
     * @param recognition_results_path file path to the results stored in the format of the annotation file
     */
    void
    visualizeResults(const typename pcl::PointCloud<PointT>::Ptr &input_cloud, const bf::path & gt_path, const bf::path &recognition_results_path);

    Eigen::MatrixXi compute_confusion_matrix();
};


}
}
