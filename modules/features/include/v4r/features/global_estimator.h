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


#ifndef V4R_GLOBAL_ESTIMATOR_H_
#define V4R_GLOBAL_ESTIMATOR_H_

#include <v4r/core/macros.h>
#include <v4r/common/faat_3d_rec_framework_defines.h>
#include <v4r/common/normal_estimator.h>

namespace v4r
{
    template <typename PointT>
    class V4R_EXPORTS GlobalEstimator {
      protected:
        bool computed_normals_;
        typedef typename pcl::PointCloud<PointT>::Ptr PointInTPtr;
        typename boost::shared_ptr<PreProcessorAndNormalEstimator<PointT, pcl::Normal> > normal_estimator_;
        pcl::PointCloud<pcl::Normal>::Ptr normals_;
        PointInTPtr input_cloud_; /// @brief point cloud containing the object
        std::vector<int> indices_; /// @brief indices of the point cloud belonging to the object


      public:

        /**
         * @brief global feature description
         * @return signatures describing the point cloud
         */
        virtual std::vector<float>
        estimate ()=0;

        /** @brief sets the normals of the point cloud belonging to the object (optional) */
        void
        setNormals(const pcl::PointCloud<pcl::Normal>::Ptr & normals)
        {
          normals_ = normals;
        }


        /** @brief sets the indices of the point cloud belonging to the object */
        void
        setIndices(const std::vector<int> & p_indices)
        {
            indices_ = p_indices;
        }


        /** @brief sets the input cloud containing the object to be classified */
        void
        setInput(const PointInTPtr & in)
        {
            input_cloud_ = in;
        }

    };
}


#endif /* REC_FRAMEWORK_ESTIMATOR_H_ */
