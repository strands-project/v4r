/******************************************************************************
 * Copyright (c) 2015 Martin Velas
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

/**
*
*      @author Martin Velas (ivelas@fit.vutbr.cz)
*      @date Jan, 2016
*      @brief multiview object instance recognizer with change detection filtering
*/

#ifndef V4R_MULTIVIEW_OBJECT_RECOGNIZER_CHANGE_DETECTION_H__
#define V4R_MULTIVIEW_OBJECT_RECOGNIZER_CHANGE_DETECTION_H__

#include <vector>
#include <iostream>
#include <string>
#include <sstream>

#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <v4r_config.h>
#include <v4r/core/macros.h>
#include <v4r/recognition/multiview_object_recognizer.h>
#include <v4r/recognition/multiview_representation.h>
#include <v4r/recognition/multi_pipeline_recognizer.h>

namespace v4r
{

template<typename PointT>
class V4R_EXPORTS MultiviewRecognizerWithChangeDetection : public MultiviewRecognizer<PointT>
{
public:
    typedef Model<PointT> ModelT;
    typedef boost::shared_ptr<ModelT> ModelTPtr;
    typedef pcl::PointCloud<PointT> Cloud;
    typedef typename Cloud::Ptr CloudPtr;

protected:
    using MultiviewRecognizer<PointT>::views_;
    using MultiviewRecognizer<PointT>::id_;

    std::map < size_t, typename pcl::PointCloud<PointT>::Ptr > removed_points_history_; /// @brief changes detected in previous observations
    std::map < size_t, typename pcl::PointCloud<PointT>::Ptr > removed_points_cumulated_history_; /// @brief changes detected in previous observations (cumulated for reconstruction filtering)

    CloudPtr changing_scene; /// @brief current status of dynamic scene

public:
    class Parameter: public MultiviewRecognizer<PointT>::Parameter {
    public:
        int min_points_for_hyp_removal_; /// @brief how many removed points must overlap hypothesis to be also considered removed
        float tolerance_for_cloud_diff_; /// @brief tolerance for point cloud difference [2cm]

        Parameter(
                int min_points_for_hyp_removal = 50,
                float tolerance_for_cloud_diff = 0.02) :
                    min_points_for_hyp_removal_(min_points_for_hyp_removal),
                    tolerance_for_cloud_diff_(tolerance_for_cloud_diff) {
        }
    } param_;

    MultiviewRecognizerWithChangeDetection(const Parameter &p = Parameter()) : MultiviewRecognizer<PointT>(p)
    {}

    MultiviewRecognizerWithChangeDetection(int argc, char ** argv);

protected:
    virtual void initHVFilters();

    virtual void cleanupHVFilters();

    void findChangedPoints(Cloud observation_unposed, Eigen::Affine3f pose,
            pcl::PointCloud<PointT> &removed_points, pcl::PointCloud<PointT> &added_points);

    virtual void reconstructionFiltering(CloudPtr observation,
            pcl::PointCloud<pcl::Normal>::Ptr observation_normals, const Eigen::Matrix4f &absolute_pose, size_t view_id);

    void setNan(pcl::Normal &normal);

    void setNan(PointT &pt);

    virtual std::vector<bool> getHypothesisInViewsMask(ModelTPtr model, const Eigen::Matrix4f &pose, size_t origin_id);

    template<typename K, typename V>
    std::vector<K> getMapKeys(const std::map<K, V> &container);
};

}

#endif
