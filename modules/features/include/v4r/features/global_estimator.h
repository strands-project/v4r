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


#pragma once

#include <v4r/core/macros.h>
#include <v4r/common/faat_3d_rec_framework_defines.h>
#include <v4r/features/types.h>

namespace v4r
{
    template <typename PointT>
    class V4R_EXPORTS GlobalEstimator {
      protected:
        pcl::PointCloud<pcl::Normal>::ConstPtr normals_;
        typename pcl::PointCloud<PointT>::ConstPtr cloud_; ///< point cloud containing the object
        std::vector<int> indices_; ///< indices of the point cloud belonging to the object
        std::string descr_name_;
        size_t descr_type_;
        size_t feature_dimensions_;

        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transforms_;   ///< transforms (e.g. from OUR-CVFH) aligning the input cloud with the descriptors's local referance frame

      public:
        GlobalEstimator(const std::string &descr_name = "", size_t descr_type = 0, size_t feature_dimensions = 0)
            : descr_name_ (descr_name),
              descr_type_ (descr_type),
              feature_dimensions_ (feature_dimensions)
        { }

        virtual ~GlobalEstimator() { }

        /**
         * @brief global feature description
         * @return signatures describing the point cloud
         */
        virtual bool
        compute(Eigen::MatrixXf &signature) = 0;

        /** @brief sets the normals of the point cloud belonging to the object (optional) */
        void
        setNormals(const pcl::PointCloud<pcl::Normal>::ConstPtr & normals)
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
        setInputCloud(const typename pcl::PointCloud<PointT>::ConstPtr & in)
        {
            cloud_ = in;
        }

        std::string
        getFeatureDescriptorName() const
        {
            return descr_name_;
        }

        size_t
        getFeatureType() const
        {
            return descr_type_;
        }

        size_t
        getFeatureDimensions() const
        {
            return feature_dimensions_;
        }

        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >
        getTransforms() const
        {
            return transforms_;
        }

        virtual bool
        needNormals() const = 0;

        typedef boost::shared_ptr< GlobalEstimator<PointT> > Ptr;
        typedef boost::shared_ptr< GlobalEstimator<PointT> const> ConstPtr;
    };
}

