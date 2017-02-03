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

#pragma once

#include <v4r/core/macros.h>
#include <v4r/common/pcl_visualization_utils.h>
#include <v4r/recognition/hypotheses_verification.h>
#include <v4r/recognition/object_hypothesis.h>

#include <pcl/visualization/pcl_visualizer.h>

namespace v4r
{

template<typename ModelT, typename SceneT> class V4R_EXPORTS HypothesisVerification;

template<typename ModelT, typename SceneT>
class V4R_EXPORTS HV_ModelVisualizer
{
private:
    friend class HypothesisVerification<ModelT, SceneT>;

    int
    vp_model_scene_, vp_model_, vp_model_scene_overlay_, vp_model_outliers_, vp_model_scene_fit_,
    vp_model_visible_, vp_model_total_fit_, vp_model_3d_fit_, vp_model_color_fit_, vp_model_normals_fit_;

    pcl::visualization::PCLVisualizer::Ptr vis_model_;

    PCLVisualizationParams::ConstPtr vis_param_;

public:
    HV_ModelVisualizer(
            const PCLVisualizationParams::ConstPtr &vis_params = boost::make_shared<PCLVisualizationParams>()
            )
        : vis_param_(vis_params)
    { }

    void visualize(const HypothesisVerification<ModelT, SceneT> *hv, const HVRecognitionModel<ModelT> &rm);

    typedef boost::shared_ptr< HV_ModelVisualizer<ModelT, SceneT> > Ptr;
    typedef boost::shared_ptr< HV_ModelVisualizer<ModelT, SceneT> const> ConstPtr;
};


template<typename ModelT, typename SceneT>
class V4R_EXPORTS HV_CuesVisualizer
{
private:
    int vp_scene_scene_, vp_scene_active_hypotheses_, vp_model_fitness_, vp_model_scene_color_dist_,
    vp_scene_fitness_, vp_scene_duplicity_, vp_scene_smooth_regions_;

    pcl::visualization::PCLVisualizer::Ptr vis_go_cues_;

    PCLVisualizationParams::ConstPtr vis_param_;

public:
    HV_CuesVisualizer(
            const PCLVisualizationParams::ConstPtr &vis_params = boost::make_shared<PCLVisualizationParams>()
            )
        : vis_param_(vis_params)
    { }

    void visualize(const HypothesisVerification<ModelT, SceneT> *hv, const boost::dynamic_bitset<> & active_solution, float cost, int times_evaluated);

    typedef boost::shared_ptr< HV_CuesVisualizer<ModelT, SceneT> > Ptr;
    typedef boost::shared_ptr< HV_CuesVisualizer<ModelT, SceneT> const> ConstPtr;
};

template<typename ModelT, typename SceneT>
class V4R_EXPORTS HV_PairwiseVisualizer
{
private:
    pcl::visualization::PCLVisualizer::Ptr vis_pairwise_;
    int vp_pair_1_, vp_pair_2_, vp_pair_3_;

    PCLVisualizationParams::ConstPtr vis_param_;

public:
    HV_PairwiseVisualizer(
            const PCLVisualizationParams::ConstPtr &vis_params = boost::make_shared<PCLVisualizationParams>()
            )
        : vis_param_(vis_params)
    { }

    void visualize( const HypothesisVerification<ModelT, SceneT> *hv );

    typedef boost::shared_ptr< HV_PairwiseVisualizer<ModelT, SceneT> > Ptr;
    typedef boost::shared_ptr< HV_PairwiseVisualizer<ModelT, SceneT> const> ConstPtr;
};


}
