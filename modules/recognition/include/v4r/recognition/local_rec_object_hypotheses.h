/******************************************************************************
 * Copyright (c) 2013 Aitor Aldoma, Thomas Faeulhammer
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

#ifndef LOCAL_REC_OBJECT_HYPOTHESES_H_
#define LOCAL_REC_OBJECT_HYPOTHESES_H_

#include <v4r/core/macros.h>
#include <v4r/recognition/model.h>
#include <pcl/correspondence.h>
#include <pcl/visualization/pcl_visualizer.h>

namespace v4r{

/**
 * @brief This class represents object hypotheses coming from local feature correspondences
 * @author Aitor Aldoma, Thomas Faeulhammer
 */
template<typename PointT>
class V4R_EXPORTS LocalObjectHypothesis
{
  typedef Model<PointT> ModelT;
  typedef boost::shared_ptr<ModelT> ModelTPtr;

  private:
    mutable boost::shared_ptr<pcl::visualization::PCLVisualizer> vis_;

  public:
    ModelTPtr model_;

    LocalObjectHypothesis() { }

    pcl::Correspondences model_scene_corresp_; //indices between model keypoints (index query) and scene cloud (index match)
    std::vector<int> indices_to_flann_models_;

    void visualize(const pcl::PointCloud<pcl::PointXYZRGB> &scene, const pcl::PointCloud<pcl::PointXYZRGB> &scene_kp) const;

    LocalObjectHypothesis & operator=(const LocalObjectHypothesis &rhs)
    {
        this->model_scene_corresp_ = rhs.model_scene_corresp_;
        this->indices_to_flann_models_ = rhs.indices_to_flann_models_;
        this->model_ = rhs.model_;
        return *this;
    }

    static bool
    gcGraphCorrespSorter (pcl::Correspondence i, pcl::Correspondence j) { return i.distance < j.distance; }

    typedef boost::shared_ptr<LocalObjectHypothesis<PointT> > Ptr;
    typedef boost::shared_ptr<const LocalObjectHypothesis<PointT> > ConstPtr;
};

}

#endif
