#include <v4r/ml/ml_utils.h>
#include <glog/logging.h>

namespace v4r
{

struct ML_Data
{
    Eigen::VectorXf x;
    int y;
};

bool sortOp (ML_Data i, ML_Data j) ;
bool sortOp (ML_Data i, ML_Data j) { return (i.y<j.y); }

void
sortTrainingData( Eigen::MatrixXf &data_train, Eigen::VectorXi &target_train)
{
    CHECK (data_train.rows() == target_train.rows() );

    std::vector<ML_Data> d( data_train.rows() );
    for(int i=0; i<data_train.rows(); i++)
    {
        d[i].x = data_train.row(i);
        d[i].y = target_train(i);
    }
    std::sort(d.begin(),d.end(), sortOp);

    for(int i=0; i<data_train.rows(); i++)
    {
        data_train.row(i) = d[i].x;
        target_train(i) = d[i].y;
    }
}


void
shuffleTrainingData(Eigen::MatrixXf &data, Eigen::VectorXi &target)
{
    CHECK (data.rows() == target.rows() );

    std::vector<size_t> vector_indices;
    vector_indices.reserve(data.size());
    for(int i=0; i<data.rows(); i++)
        vector_indices.push_back(i);
    std::random_shuffle(vector_indices.begin(), vector_indices.end());

    for(int i=0; i<data.rows(); i++)
    {
        data.  row(i).swap( data.  row( vector_indices[i] ) );
        target.row(i).swap( target.row( vector_indices[i] ) );
    }
}


Eigen::MatrixXi
computeConfusionMatrix(const Eigen::VectorXi &actual_label, const Eigen::VectorXi &predicted_label, size_t num_classes)
{
    CHECK (actual_label.rows() == predicted_label.rows() );

    size_t num_falsely_classified=0, num_correctly_classified=0;

    for(int i=0; i<actual_label.rows(); i++)
    {
        if(predicted_label(i) == actual_label(i))
            num_correctly_classified++;
        else
            num_falsely_classified++;
    }

    Eigen::MatrixXi confusion_matrix = Eigen::MatrixXi::Zero(num_classes, num_classes);

    for(int i=0; i < actual_label.rows(); i++)
        confusion_matrix( actual_label(i), predicted_label(i) ) ++;

    return confusion_matrix;
}

}
