/******************************************************************************
 * Copyright (c) 2012 Aitor Aldoma
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

#include <v4r/recognition/source.h>
#include <v4r/core/macros.h>

namespace v4r
{
/**
     * \brief Data source class based on partial views from sensor.
     * In this case, the training data is obtained directly from a depth sensor.
     * The filesystem should contain pcd files (representing a view of an object in
     * camera coordinates) and each view needs to be associated with a txt file
     * containing a 4x4 matrix representing the transformation from camera coordinates
     * to a global object coordinates frame.
     * \author Aitor Aldoma
     */

template<typename PointT = pcl::PointXYZRGB>
class V4R_EXPORTS RegisteredViewsSource : public Source<PointT>
{
private:
    using Source<PointT>::path_;
    using Source<PointT>::models_;
    using Source<PointT>::load_into_memory_;
    using Source<PointT>::view_prefix_;
    using Source<PointT>::indices_prefix_;
    using Source<PointT>::pose_prefix_;
    using Source<PointT>::entropy_prefix_;

public:
    RegisteredViewsSource ()
    {
        load_into_memory_ = false;
    }

    void
    loadInMemorySpecificModel(Model<PointT> &model);

    void
    loadModel (Model<PointT> & model);

    void
    generate (); ///< Creates the model representation of the training set, generating views if needed

    typedef boost::shared_ptr< RegisteredViewsSource<PointT> > Ptr;
    typedef boost::shared_ptr< RegisteredViewsSource<PointT> const> ConstPtr;
};
}
