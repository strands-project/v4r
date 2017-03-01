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

#include <v4r/core/macros.h>
#include <v4r/recognition/local_recognition_pipeline.h>
#include <v4r/recognition/multi_pipeline_recognizer.h>
#include <v4r/recognition/hypotheses_verification.h>
#include <v4r/apps/visualization.h>

#pragma once

namespace v4r
{

namespace apps
{

template<typename PointT>
class V4R_EXPORTS ObjectRecognizer
{
private:
    typename v4r::MultiRecognitionPipeline<PointT>::Ptr mrec_; ///< multi-pipeline recognizer
    typename v4r::LocalRecognitionPipeline<PointT>::Ptr local_recognition_pipeline_; ///< local recognition pipeline (member variable just because of visualization of keypoints)
    typename v4r::HypothesisVerification<PointT, PointT>::Ptr hv_; ///< hypothesis verification object

    typename v4r::ObjectRecognitionVisualizer<PointT>::Ptr rec_vis_; ///< visualization object

    std::vector<ObjectHypothesesGroup<PointT> > generated_object_hypotheses_;
    std::vector<typename ObjectHypothesis<PointT>::Ptr > verified_hypotheses_;

    double chop_z_; ///< Cut-off distance in meter
    bool remove_planes_ = false;
    size_t min_plane_points_ = 200;
    bool visualize_; ///< if true, visualizes objects
    bool skip_verification_; ///< if true, will only generate hypotheses but not verify them

public:
    ObjectRecognizer() :
        chop_z_ ( 5.f ),
        remove_planes_ (false),
        min_plane_points_ (200),
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
    void initialize(const std::vector<std::string> &command_line_arguments);

    /**
     * @brief recognize recognize objects in point cloud
     * @param cloud (organized) point cloud
     * @return
     */
    std::vector<typename ObjectHypothesis<PointT>::Ptr >
    recognize(typename pcl::PointCloud<PointT>::Ptr &cloud);

    /**
     * @brief getObjectHypothesis
     * @return generated object hypothesis
     */
    std::vector<ObjectHypothesesGroup<PointT> >
    getGeneratedObjectHypothesis() const
    {
        return generated_object_hypotheses_;
    }

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
};

}

}
