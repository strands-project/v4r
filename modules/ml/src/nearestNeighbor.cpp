#include <v4r/ml/nearestNeighbor.h>
#include <glog/logging.h>

namespace v4r
{

void
NearestNeighborClassifier::train(const Eigen::MatrixXf &training_data, const Eigen::VectorXi &training_label)
{
    CHECK ( training_data.rows() == training_label.rows() );

    flann_.reset ( new EigenFLANN);
    flann_->param_.knn_ = param_.knn_;
    flann_->param_.distance_metric_ = param_.distance_metric_;
    flann_->param_.kdtree_splits_ = param_.kdtree_splits_;
    flann_->createFLANN(training_data);

    training_label_ = training_label;
}

void
NearestNeighborClassifier::predict(const Eigen::MatrixXf &query_data, Eigen::MatrixXi &predicted_label)
{
    flann_->nearestKSearch(query_data, knn_indices_, knn_distances_);

    predicted_label.resize( knn_indices_.rows() ,knn_indices_.cols() );

    for(int row_id=0; row_id<knn_indices_.rows(); row_id++)
    {
        for (int col_id = 0; col_id < knn_indices_.cols(); col_id++)
        {
            predicted_label(row_id, col_id ) = training_label_( knn_indices_(row_id, col_id) );
        }
    }
}

}
