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
    class V4R_EXPORTS Parameter
    {
    public:
        int kdtree_splits_;
        int distance_metric_; /// @brief defines the norm used for feature matching (1... L1 norm, 2... L2 norm)
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
    EigenFLANN(const Parameter &p = Parameter()) : param_(p)
    {

    }

    /**
     * @brief creates a FLANN index
     * @param signatures matrix with size num_features x dimensionality
     * @return
     */
    bool
    createFLANN ( const Eigen::MatrixXf &data)
    {
        if(data.rows() == 0 || data.cols() == 0) {
            std::cerr << "No data provided for building Flann index!" << std::endl;
            return false;
        }

        flann_data_.reset ( new flann::Matrix<float>( new float[data.rows() * data.cols()], data.rows(), data.cols() ) );

        for ( int row_id = 0; row_id < data.rows(); row_id++ )
        {
            for ( int col_id = 0; col_id < data.cols(); col_id++ )
                flann_data_->ptr() [row_id * data.cols() + col_id] = data(row_id, col_id);
        }

        if(param_.distance_metric_==2)
        {
            flann_index_l2_.reset( new flann::Index<flann::L2<float> > (*flann_data_, flann::KDTreeIndexParams ( 4 )));
            flann_index_l2_->buildIndex();
        }
        else
        {
            flann_index_l1_.reset( new flann::Index<flann::L1<float> > (*flann_data_, flann::KDTreeIndexParams ( 4 )));
            flann_index_l1_->buildIndex();
        }
        return true;
    }

    /**
     * @brief nearestKSearch perform nearest neighbor search for the rows queries in query_signature
     * @param query_signature (rows = num queries; cols = feature dimension)
     * @param indices (rows = num queries; cols = nearest neighbor indices)
     * @param distances (rows = num queries; cols = nearest neighbor distances)
     * @return
     */
    bool
    nearestKSearch (const Eigen::MatrixXf &query_signature, Eigen::MatrixXi &indices, Eigen::MatrixXf &distances)
    {
        flann::Matrix<int> flann_indices (new int[query_signature.rows() * param_.knn_], query_signature.rows(), param_.knn_);
        flann::Matrix<float> flann_distances (new float[query_signature.rows() * param_.knn_], query_signature.rows(), param_.knn_);

        indices.resize( query_signature.rows(), param_.knn_ );
        distances.resize( query_signature.rows(), param_.knn_ );

        flann::Matrix<float> query_desc (new float[ query_signature.rows() *  query_signature.cols()],  query_signature.rows(), query_signature.cols());
        for ( int row_id = 0; row_id < query_signature.rows(); row_id++ )
        {
            for ( int col_id = 0; col_id < query_signature.cols(); col_id++ )
            {
                query_desc.ptr() [row_id * query_signature.cols() + col_id] = query_signature(row_id, col_id);
            }
        }

        if(param_.distance_metric_==2)
        {
            if(!flann_index_l2_->knnSearch ( query_desc, flann_indices, flann_distances, param_.knn_, flann::SearchParams ( param_.kdtree_splits_ ) ))
                return false;
        }
        else
        {
            if(!flann_index_l1_->knnSearch ( query_desc, flann_indices, flann_distances, param_.knn_, flann::SearchParams ( param_.kdtree_splits_ ) ))
                return false;
        }

        for ( int row_id = 0; row_id < query_signature.rows(); row_id++ )
        {
            for(int i=0; i<param_.knn_; i++)
            {
                indices(row_id, i) = flann_indices[row_id][i];
                distances(row_id, i) = flann_distances[row_id][i];
            }
        }

        delete[] flann_indices.ptr ();
        delete[] flann_distances.ptr ();

        return true;
    }


    typedef boost::shared_ptr< EigenFLANN > Ptr;
    typedef boost::shared_ptr< EigenFLANN const> ConstPtr;
};
}

#endif
