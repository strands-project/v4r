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

/**
*
*      @author Thomas Faeulhammer (faeulhammer@acin.tuwien.ac.at)
*      @date July, 2015
*      @brief FLANN helper functions for conversion from Eigen (useful for e.g. fast nearest neighbor search in high-dimensional space)
*/

#ifndef V4R_FLANN_H__
#define V4R_FLANN_H__

#include <pcl/kdtree/flann.h>
#include <v4r/core/macros.h>
#include <boost/shared_ptr.hpp>
#include <Eigen/Core>

namespace v4r
{

class V4R_EXPORTS EigenFLANN
{
public:
    typedef boost::shared_ptr< EigenFLANN > Ptr;
    typedef boost::shared_ptr< EigenFLANN const> ConstPtr;

    class V4R_EXPORTS Parameter
    {
    public:
        int kdtree_splits_;
        int distance_metric_; ///< defines the norm used for feature matching (1... L1 norm, 2... L2 norm)
        int knn_;

        Parameter(
                int kdtree_splits = 128,
                int distance_metric = 2,
                int knn = 1
                )
            : kdtree_splits_ (kdtree_splits),
              distance_metric_ (distance_metric),
              knn_ (knn)
        {}
    }param_;

private:
    boost::shared_ptr< typename flann::Index<flann::L1<float> > > flann_index_l1_;
    boost::shared_ptr< typename flann::Index<flann::L2<float> > > flann_index_l2_;
    boost::shared_ptr<flann::Matrix<float> > flann_data_;

public:
    EigenFLANN(const Parameter &p = Parameter()) : param_(p) { }

    /**
     * @brief creates a FLANN index
     * @param signatures matrix with size num_features x dimensionality
     * @return
     */
    bool
    createFLANN ( const Eigen::MatrixXf &data);


    /**
     * @brief nearestKSearch perform nearest neighbor search for the rows queries in query_signature
     * @param query_signature (rows = num queries; cols = feature dimension)
     * @param indices (rows = num queries; cols = nearest neighbor indices)
     * @param distances (rows = num queries; cols = nearest neighbor distances)
     * @return
     */
    bool
    nearestKSearch (const Eigen::MatrixXf &query_signature, Eigen::MatrixXi &indices, Eigen::MatrixXf &distances) const;
};
}

#endif
