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

#include <pcl/common/common.h>
#include <v4r/core/macros.h>
#include <v4r/features/global_estimator.h>

namespace v4r
{

/**
 *      @brief Abstract class for object classification based on point cloud data
 *      @date Nov., 2015
 *      @author Thomas Faeulhammer
 */
template<typename PointInT>
class V4R_EXPORTS GlobalClassifier {

protected:
    typedef typename pcl::PointCloud<PointInT>::Ptr PointInTPtr;

    std::string training_dir_;  /// @brief directory containing training data

    PointInTPtr input_; /// @brief Point cloud to be classified

    std::vector<int> indices_; /// @brief indices of the object to be classified

    std::vector<std::string> categories_;   /// @brief classification results

    std::vector<float> confidences_;   /// @brief confidences associated to the classification results (normalized to 0...1)

    typename boost::shared_ptr<GlobalEstimator<PointInT> > estimator_; /// @brief estimator used for describing the object


public:

    /** @brief sets the indices of the object to be classified */
    void
    setIndices (const std::vector<int> & indices)
    {
        indices_ = indices;
    }

    /** \brief Sets the input cloud to be classified */
    void
    setInputCloud (const PointInTPtr & cloud)
    {
        input_ = cloud;
    }

    void
    setTrainingDir (const std::string & dir)
    {
        training_dir_ = dir;
    }

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
    setFeatureEstimator (const typename boost::shared_ptr<GlobalEstimator<PointInT> > & feat)
    {
        estimator_ = feat;
    }

    virtual void
    classify () = 0;
};
}
