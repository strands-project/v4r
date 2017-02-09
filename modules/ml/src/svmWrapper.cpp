#include <v4r/common/miscellaneous.h>
#include <v4r/io/filesystem.h>
#include <v4r/ml/ml_utils.h>
#include <v4r/ml/svmWrapper.h>
#include <iostream>
#include <fstream>
#include <set>
#include <glog/logging.h>

namespace v4r
{

void svmClassifier::predict(const Eigen::MatrixXf &query_data, Eigen::MatrixXi &predicted_label) const
{
    if(param_.svm_.probability)
        predicted_label.resize(query_data.rows(), param_.knn_);
    else
        predicted_label.resize(query_data.rows(), 1);

    for(int i=0; i<query_data.rows(); i++)
    {
        ::svm_node *svm_n_test = new ::svm_node[ query_data.cols()+1 ];

        for(int kk=0; kk<query_data.cols(); kk++)
        {
            svm_n_test[kk].value = query_data(i, kk);
            svm_n_test[kk].index = kk+1;
        }
        svm_n_test[ query_data.cols() ].index = -1;

        if(param_.svm_.probability)
        {
            double *prob_estimates;
            try
            {
                prob_estimates = new double[ svm_mod_->nr_class ];
            }
            catch (std::bad_alloc&)
            {
                std::cerr << "Error allocating memory " << std::endl;
            }

            double bla = svm_predict_probability(svm_mod_, svm_n_test, prob_estimates);
            (void) bla;

            std::vector<double> probs ( svm_mod_->nr_class );
            for(int label_id=0; label_id<svm_mod_->nr_class; label_id++)
                probs[label_id] = prob_estimates[label_id];

            std::vector<size_t> indices = sort_indexes(probs);  //NOTE sorted in ascending order. We want highest values!

            for(int k=0; k<param_.knn_; k++)
                predicted_label(i, k) = indices[ indices.size() - 1 - k ];

            delete [] prob_estimates;
        }
        else
        {
            predicted_label(i, 0) = (int)::svm_predict(svm_mod_, svm_n_test);
        }

        delete [] svm_n_test;
    }
}

void svmClassifier::dokFoldCrossValidation(
        const Eigen::MatrixXf &data_train,
        const Eigen::VectorXi &target_train,
        size_t k,
        double model_para_C_min,
        double model_para_C_max,
        double step_multiplicator_C,
        double model_para_gamma_min,
        double model_para_gamma_max,
        double step_multiplicator_gamma)
{
    CHECK(data_train.rows() == target_train.rows() );

    double bestC = model_para_C_min, bestGamma = model_para_gamma_min, bestTestPerformanceValue=0;
    std::vector<Eigen::MatrixXi> best_confusion_matrices_v(k);
    std::vector<Eigen::MatrixXi> confusion_matrices_v(k);

    std::set<int> labels; // to know how many different labels there are
    for(int i=0; i<target_train.rows(); i++)
    {
        int label = target_train(i);
        if( label > (int)labels.size()+1 )
            std::cerr << "Training labels are not sorted. Take care with unsorted training labels when using probabilities. The order will then correspond to the time of occurence in the training labels." << std::endl;
        labels.insert(label);
    }

    size_t num_classes = labels.size();

    Eigen::MatrixXf data_train_shuffled = data_train;
    Eigen::VectorXi target_train_shuffled = target_train;
    shuffleTrainingData(data_train_shuffled, target_train_shuffled);

    for(double C = model_para_C_min; C <= model_para_C_max; C *= step_multiplicator_C)
    {
        for(double gamma = model_para_gamma_min; gamma <= model_para_gamma_max; gamma *= step_multiplicator_gamma)
        {
            double avg_performance;
            param_.svm_.C = C;
            param_.svm_.gamma = gamma;
            std::cout << "Computing svm for C=" << C << " and gamma=" << gamma << std::endl;

            for(size_t current_val_set_id = 0; current_val_set_id < k; current_val_set_id++)
            {
                Eigen::MatrixXf data_train_sub;
                Eigen::MatrixXf data_val;
                Eigen::VectorXi target_train_sub;
                Eigen::VectorXi target_val;

                for(int i=0; i < target_train.rows(); i++)
                {
                    if(i%k == current_val_set_id)
                    {
                        int num_entries = target_val.rows();
                        data_val.conservativeResize(num_entries + 1, data_train_shuffled.cols());
                        data_val.row(num_entries) = data_train_shuffled.row(i);
                        target_val.conservativeResize(num_entries+1);
                        target_val(num_entries) = target_train_shuffled(i);
                    }
                    else
                    {
                        int num_entries = data_train_sub.rows();
                        data_train_sub.conservativeResize(num_entries + 1, data_train_shuffled.cols());
                        data_train_sub.row(num_entries) = data_train_shuffled.row(i);
                        target_train_sub.conservativeResize(num_entries+1);
                        target_train_sub(num_entries) = target_train_shuffled(i);
                    }
                }
                train(data_train_sub, target_train_sub);

                Eigen::MatrixXi target_pred;
                predict( data_val, target_pred );
                confusion_matrices_v[current_val_set_id] = computeConfusionMatrix( target_val, target_pred.col(0), num_classes );
                std::cout << "confusion matrix ( " << current_val_set_id << ")" << std::endl << confusion_matrices_v[current_val_set_id] << std::endl;
            }

            Eigen::MatrixXi total_confusion_matrix = Eigen::MatrixXi::Zero(num_classes, num_classes);
            for(size_t i=0; i< k; i++)
                total_confusion_matrix += confusion_matrices_v[i];

            std::cout << "Total confusion matrix:" << std::endl << total_confusion_matrix << std::endl << std::endl;

            size_t sum=0;
            size_t trace=0;
            for(int i=0; i<total_confusion_matrix.rows(); i++)
            {
                for(int jj=0; jj<total_confusion_matrix.cols(); jj++)
                {
                    sum += total_confusion_matrix(i,jj);
                    if (i == jj)
                        trace += total_confusion_matrix(i,jj);
                }
            }
            avg_performance = static_cast<double>(trace) / sum;

            std::cout << "My average performance is " << avg_performance << std::endl;

            if(avg_performance > bestTestPerformanceValue)
            {
                bestTestPerformanceValue = avg_performance;
                bestC = C;
                bestGamma = gamma;
                for(size_t i=0; i<k; i++)
                {
                    best_confusion_matrices_v[i] = confusion_matrices_v[i];
                    std::cout << "best confusion matrix ( " << i << ")" << std::endl << best_confusion_matrices_v[i] << std::endl;
                }
            }

            if( param_.svm_.kernel_type != ::RBF && param_.svm_.kernel_type != ::POLY && param_.svm_.kernel_type != ::SIGMOID)
                break;  // for these kernel types the gamma value should not matter
        }
    }
    param_.svm_.C = bestC;
    param_.svm_.gamma = bestGamma;

    Eigen::MatrixXi confusion_matrix = Eigen::MatrixXi::Zero(num_classes, num_classes);
    for(size_t i=0; i< k; i++)
    {
        confusion_matrix += best_confusion_matrices_v[i];
        std::cout << "Confusion matrix (part " << i << "/" << k << "): " << std::endl << best_confusion_matrices_v[i] << std::endl << std::endl;
    }
    std::cout << "SVM cross-validation achieved the best performance(" << bestTestPerformanceValue<< ") for C=" << bestC <<
                 " and gamma=" << bestGamma << ". " << std::endl <<
                 "Total confusion matrix:" << std::endl << confusion_matrix << std::endl << std::endl;
}


void svmClassifier::train(const Eigen::MatrixXf &training_data, const Eigen::VectorXi & training_label)
{
    CHECK(training_data.rows() == training_label.rows() );

    if(param_.do_cross_validation_)
        dokFoldCrossValidation(training_data, training_label, 5);

    ::svm_problem *svm_prob = new ::svm_problem;

    svm_prob->l = training_data.rows(); //number of training examples
    svm_prob->x = new ::svm_node *[svm_prob->l];

    for(int i = 0; i<svm_prob->l; i++)
        svm_prob->x[i] = new ::svm_node[ training_data.cols()+1 ];  // add one additional dimension and set that index to -1 (libsvm requirement)

    svm_prob->y = new double[svm_prob->l];

    for(int i=0; i<svm_prob->l; i++)
    {
        for(int kk=0; kk < training_data.cols(); kk++)
        {
            svm_prob->x[i][kk].value = training_data(i,kk);
            svm_prob->x[i][kk].index = kk+1;
        }
        svm_prob->x[i][ training_data.cols() ].index = -1;
        svm_prob->y[i] = training_label(i);
    }
    svm_mod_ = ::svm_train(svm_prob, &param_.svm_);

    // free memory
//    for(int i = 0; i<svm_prob->l; i++)
//        delete [] svm_prob->x[i];
//    delete [] svm_prob->x;
//    delete [] svm_prob->y;
//    delete svm_prob;
}

void svmClassifier::saveModel(const std::string &filename) const
{
    v4r::io::createDirForFileIfNotExist( filename );

    try
    {
        ::svm_save_model(filename.c_str(), svm_mod_);
    }
    catch (std::exception& e)
    {
        std::cerr << "Could not save svm model to file " << filename << ". " << std::endl;
    }
}

void svmClassifier::loadModel(const std::string &filename)
{
    svm_mod_ = ::svm_load_model(filename.c_str());
}

}
