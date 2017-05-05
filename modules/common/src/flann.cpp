#include <v4r/common/flann.h>

namespace v4r
{

bool
EigenFLANN::nearestKSearch (const Eigen::MatrixXf &query_signature, Eigen::MatrixXi &indices, Eigen::MatrixXf &distances) const
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

bool
EigenFLANN::createFLANN ( const Eigen::MatrixXf &data)
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

}
