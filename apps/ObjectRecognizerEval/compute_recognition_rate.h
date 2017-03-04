#pragma once

#include <pcl/visualization/pcl_visualizer.h>

#include <iostream>
#include <sstream>
#include <fstream>


// -m /media/Data/datasets/TUW/models/ -t /media/Data/datasets/TUW/validation_set/ -g /media/Data/datasets/TUW/annotations/ -r /home/thomas/recognition_results_eval/

typedef pcl::PointXYZRGB PointT;

struct Model
{
    pcl::PointCloud<PointT>::Ptr cloud;
    Eigen::Vector4f centroid;
};

struct Hypothesis
{
    Eigen::Matrix4f pose;
    float occlusion;
};


class RecognitionEvaluator
{
private:
    float rotation_error_threshold_deg;
    float translation_error_threshold_m;
    float occlusion_threshold;
    std::vector<std::string> coordinate_system_ids_;
    std::string out_dir;
    std::string gt_dir;
    std::string or_dir;
    std::string models_dir;
    std::string test_dir;
    bool visualize;
    bool use_generated_hypotheses;

    pcl::visualization::PCLVisualizer::Ptr vis;
    int vp1, vp2, vp3;

    std::map<std::string, Model> models;

    int init(const std::vector<std::string> &params);


public:
    RecognitionEvaluator(const std::vector<std::string> &params = std::vector<std::string>())
        :
          rotation_error_threshold_deg(30.f),
          translation_error_threshold_m(0.05f),
          occlusion_threshold(0.95f),
          out_dir("/tmp/recognition_rates/"),
          visualize(false),
          use_generated_hypotheses(false)
    {
        init(params);
    }

    // =======  DECLARATIONS ===================
    /**
     * @brief computeError compute translation error for a given pose
     * @param[in] pose_a
     * @param[in] pose_b
     * @param[in] centroid of object model in model coordinate system
     * @param[out] trans_error translation error
     * @param[out] rot_error rotation error
     * @return boolean indicitating if error is outside given threshold
     */
    bool computeError(const Eigen::Matrix4f &pose_a, const Eigen::Matrix4f &pose_b, const Eigen::Vector4f &centroid_model,
                      float &trans_error, float &rot_error);



    /**
     * @brief checkMatchvector check a certain match configuration
     * @param rec2gt matches indicated by first index corresponding
     * to model id and second index to ground-truth id
     * @param rec_hyps recognition hypotheses
     * @param gt_hyps ground-truth hypotheses
     * @param centroid of object model in model coordinate system
     * @param sum_translation_error
     * @param tp true positives
     * @param fp flase positives
     * @param fn false negatives
     */
    void checkMatchvector(const std::vector< std::pair<int, int> > &rec2gt,
                          const std::vector<Hypothesis> &rec_hyps,
                          const std::vector<Hypothesis> &gt_hyps,
                          const Eigen::Vector4f &model_centroid,
                          double &sum_translation_error,
                          size_t &tp, size_t &fp, size_t &fn);

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
     * @return best match for the given hypotheses. First index corresponds to element in
     * given recognition hypothesis, second index to ground-truth hypothesis
     */
    std::vector<std::pair<int, int> > selectBestMatch(const std::vector<Hypothesis> &rec_hyps,
                                                      const std::vector<Hypothesis> &gt_hyps,
                                                      const Eigen::Vector4f &model_centroid,
                                                      size_t &tp, size_t &fp, size_t &fn,
                                                      double &sum_translation_error);

    std::map<std::string, std::vector<Hypothesis> >
    readHypothesesFromFile( const std::string &filename );

    void compute_recognition_rate (size_t &total_tp, size_t &total_fp, size_t &total_fn);
    float compute_recognition_rate_over_occlusion ();

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
};

