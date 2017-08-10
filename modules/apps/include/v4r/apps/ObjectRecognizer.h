/******************************************************************************
 * Copyright (c) 2017 Thomas Faeulhammer
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

#include <boost/serialization/vector.hpp>
#include <v4r/apps/CloudSegmenter.h>
#include <v4r/apps/ObjectRecognizerParameter.h>
#include <v4r/apps/visualization.h>
#include <v4r/common/normals.h>
#include <v4r/core/macros.h>
#include <v4r/io/filesystem.h>
#include <v4r/recognition/local_recognition_pipeline.h>
#include <v4r/recognition/multi_pipeline_recognizer.h>
#include <v4r/recognition/hypotheses_verification.h>
#pragma once

namespace v4r
{

namespace apps
{

template<typename PointT>
class V4R_EXPORTS ObjectRecognizer
{
private:
    typename v4r::RecognitionPipeline<PointT>::Ptr mrec_; ///< multi-pipeline recognizer
    typename v4r::LocalRecognitionPipeline<PointT>::Ptr local_recognition_pipeline_; ///< local recognition pipeline (member variable just because of visualization of keypoints)
    typename v4r::HypothesisVerification<PointT, PointT>::Ptr hv_; ///< hypothesis verification object
    typename v4r::NormalEstimator<PointT>::Ptr normal_estimator_;    ///< normal estimator used for computing surface normals (currently only used at training)

    typename v4r::ObjectRecognitionVisualizer<PointT>::Ptr rec_vis_; ///< visualization object

    typename v4r::apps::CloudSegmenter<PointT>::Ptr cloud_segmenter_; ///< cloud segmenter for plane removal (if enabled)

    bool visualize_; ///< if true, visualizes objects
    bool skip_verification_; ///< if true, will only generate hypotheses but not verify them
    std::string models_dir_;

    ObjectRecognizerParameter param_;

    Camera::Ptr camera_;

    typename Source<PointT>::Ptr model_database_;

    // MULTI-VIEW STUFF
    class View
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typename pcl::PointCloud<PointT>::ConstPtr cloud_;
        typename pcl::PointCloud<PointT>::Ptr processed_cloud_;
        typename pcl::PointCloud<PointT>::Ptr removed_points_;
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals_;
        std::vector<std::vector<float> > pt_properties_;
        Eigen::Matrix4f camera_pose_;
    };
    std::vector<View> views_; ///< all views in sequence

    /**
     * @brief detectChanges detect changes in multi-view sequence (e.g. objects removed or added to the scene within observation period)
     * @param v current view
     */
    void
    detectChanges(View &v);

    typename pcl::PointCloud<PointT>::Ptr registered_scene_cloud_;  ///< registered point cloud of all processed input clouds in common camera reference frame

    std::vector<std::pair<std::string, float> > elapsed_time_; ///< measurements of computation times for various components


public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ObjectRecognizer() :
        visualize_ (false),
        skip_verification_(false)
    {}

    /**
     * @brief initialize initialize Object recognizer (sets up model database, recognition pipeline and hypotheses verification)
     * @param argc
     * @param argv
     */
    void initialize(int argc, char ** argv)
    {
        std::vector<std::string> arguments(argv + 1, argv + argc);
        initialize(arguments);
    }

    /**
     * @brief initialize initialize Object recognizer (sets up model database, recognition pipeline and hypotheses verification)
     * @param arguments
     */
    void initialize(std::vector<std::string> &command_line_arguments, const boost::filesystem::path &config_folder = bf::path("cfg"));

    /**
     * @brief recognize recognize objects in point cloud
     * @param cloud (organized) point cloud
     * @return
     */
    std::vector<ObjectHypothesesGroup > recognize(const typename pcl::PointCloud<PointT>::ConstPtr &cloud);

    typename pcl::PointCloud<PointT>::ConstPtr
    getModel( const std::string &model_name, int resolution_mm ) const
    {
        bool found;
        typename Source<PointT>::ConstPtr mdb = mrec_->getModelDatabase();
        typename Model<PointT>::ConstPtr model = mdb->getModelById("", model_name, found);
        if(!found)
        {
            std::cerr << "Could not find model with name " << model_name << std::endl;
            typename pcl::PointCloud<PointT>::ConstPtr foo;
            return foo;
        }

        return model->getAssembled( resolution_mm );
    }

    std::string
    getModelsDir() const
    {
        return models_dir_;
    }

    void
    setModelsDir(const std::string &dir)
    {
        models_dir_ = dir;
    }

    /**
     * @brief getElapsedTimes
     * @return compuation time measurements for various components
     */
    std::vector<std::pair<std::string, float> >
    getElapsedTimes() const
    {
        return elapsed_time_;
    }

    /**
     * @brief getParam get recognition parameter
     * @return parameter
     */
    ObjectRecognizerParameter
    getParam() const
    {
        return param_;
    }

    /**
     * @brief resetMultiView resets all state variables of the multi-view and initializes a new multi-view sequence
     */
    void
    resetMultiView();
};

}

}
