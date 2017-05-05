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
    int num_examples = query_data.rows();
    int num_attributes = query_data.cols();

    if(param_.svm_.probability)
        predicted_label.resize(num_examples, param_.knn_);
    else
        predicted_label.resize(num_examples, 1);

    Eigen::MatrixXf query_data_scaled = query_data;

    if(param_.do_scaling_)
    {
        for(int row = 0 ; row < num_examples; row++)
        {
            query_data_scaled.row(row).array() *= scale_.array();
        }
    }


    for(int i=0; i<num_examples; i++)
    {
        ::svm_node *svm_n_test = new ::svm_node[ num_attributes+1 ];

        for(int kk=0; kk<num_attributes; kk++)
        {
            svm_n_test[kk].value = query_data_scaled(i, kk);
            svm_n_test[kk].index = kk+1;
        }
        svm_n_test[ num_attributes ].index = -1;

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

void svmClassifier::train(const Eigen::MatrixXf &training_data, const Eigen::VectorXi & training_label)
{
    CHECK(training_data.rows() == training_label.rows() );

    int num_examples = training_data.rows();
    int num_attributes = training_data.cols();

    if (param_.svm_.gamma < 0)
        param_.svm_.gamma = 1. / num_attributes;

    if( !param_.svm_.probability && param_.knn_ > 1)
    {
        LOG(WARNING) << "KNN set with k>1 but probability estimate is turned off. Will turn on SVM probability to estimate not only winner.";
        param_.svm_.probability = 1;
    }

    Eigen::MatrixXf training_data_scaled = training_data;
    if(param_.do_scaling_)
    {
        scale_ = Eigen::VectorXf::Ones( num_attributes );
//        offset_ = Eigen::VectorXf::Zero( num_attributes );

        const Eigen::VectorXf maxa = training_data.colwise().maxCoeff();
        const Eigen::VectorXf mina = Eigen::VectorXf::Zero( num_attributes );//training_data.colwise().minCoeff();
        const Eigen::VectorXf range = maxa - mina;

        for(int attr_id = 0 ; attr_id < num_attributes; attr_id++)
        {
            if (range(attr_id) > std::numeric_limits<float>::epsilon())
            {
                scale_(attr_id) = 1.f/range(attr_id);
            }
        }

        for(int row = 0 ; row < num_examples; row++)
        {
            training_data_scaled.row(row).array() *= scale_.array();
        }
    }


    // fill training data into an SVM problem
    ::svm_problem *svm_prob = new ::svm_problem;
    svm_prob->l = num_examples; //number of training examples
    svm_prob->x = new ::svm_node *[svm_prob->l];

    for(int i = 0; i<svm_prob->l; i++)
        svm_prob->x[i] = new ::svm_node[ num_attributes+1 ];  // add one additional dimension and set that index to -1 (libsvm requirement)

    svm_prob->y = new double[svm_prob->l];

    for(int i=0; i<svm_prob->l; i++)
    {
        for(int kk=0; kk < num_attributes; kk++)
        {
            svm_prob->x[i][kk].value = (double)training_data_scaled(i,kk);
            svm_prob->x[i][kk].index = kk+1;
        }
        svm_prob->x[i][ num_attributes ].index = -1;
        svm_prob->y[i] = training_label(i);
    }

    if(v4r::io::existsFile(param_.filename_))
    {
        VLOG(1) << "Loading SVM model from file " << param_.filename_ << ".............";
        loadModel( param_.filename_ );
        LOG(INFO) << "Loaded SVM model from file " << param_.filename_;
    }
    else
    {

        if( param_.do_cross_validation_ > 1 )
        {
            LOG(INFO) << "Performing " << param_.do_cross_validation_ << "-fold cross validation.";
            std::set<int> labels; // to know how many different labels there are
            for(int i=0; i<training_label.rows(); i++)
            {
                int label = training_label(i);
                if( label > (int)labels.size()+1 )
                    std::cerr << "Training labels are not sorted. Take care with unsorted training labels when using probabilities. The order will then correspond to the time of occurence in the training labels." << std::endl;
                labels.insert(label);
            }

            size_t num_classes = labels.size();
            float best_performance = std::numeric_limits<float>::min();
            ::svm_parameter best_parameter = param_.svm_;

            for(double C = param_.cross_validation_range_C_[0]; C <= param_.cross_validation_range_C_[1]; C *= param_.cross_validation_range_C_[2])
            {
                for(double gamma = param_.cross_validation_range_gamma_[0]; gamma <= param_.cross_validation_range_gamma_[1]; gamma *= param_.cross_validation_range_gamma_[2])
                {
                    param_.svm_.C = C;
                    param_.svm_.gamma = gamma;

                    if( (param_.svm_.kernel_type == ::LINEAR) && (gamma>param_.cross_validation_range_gamma_[0]) )
                    {
                        VLOG(1) << "skipping remaing gamma values as linear kernel does not use gamma.";
                        break;
                    }

                    LOG(INFO) << "Cross-validate parameters C=" << param_.svm_.C << " and gamma=" << param_.svm_.gamma;

                    double* target = (double*)malloc( svm_prob->l  * sizeof(double) );
                    ::svm_cross_validation( svm_prob, &param_.svm_, param_.do_cross_validation_, target);

                    Eigen::VectorXi predicted_label( svm_prob->l);

                    for(int i=0; i <svm_prob->l; i++)
                        predicted_label(i) = target[i ];

                    Eigen::MatrixXi conf_matrix = computeConfusionMatrix( training_label, predicted_label.col(0), num_classes );
                    float performance = (float)conf_matrix.trace() / conf_matrix.sum();

                    LOG(INFO) << "Accuracy for parameters C=" << param_.svm_.C << " and gamma=" << param_.svm_.gamma << ": " << performance << " with confusion matrix: " << std::endl << conf_matrix << std::endl;

                    if (performance > best_performance)
                    {
                        best_performance = performance;
                        best_parameter = param_.svm_;
                    }

                    delete[] target;
                }
            }

            param_.svm_ = best_parameter;
            LOG(INFO) << "Best parameters achieved from cross-validation: C=" << param_.svm_.C << " and gamma=" << param_.svm_.gamma;
        }
        std::ofstream ofparam("svm_param.txt");
        ofparam << "C: " << param_.svm_.C << ", gamma: " << param_.svm_.gamma;
        ofparam.close();

        svm_mod_ = ::svm_train(svm_prob, &param_.svm_);

    //    v4r::io::createDirForFileIfNotExist( filename );
        this->saveModel( "model.svm");

    }
    // free memory
//    for(int i = 0; i<svm_prob->l; i++)
//        delete [] svm_prob->x[i];
//    delete [] svm_prob->x;
//    delete [] svm_prob->y;
//    delete svm_prob;
}

void svmClassifier::saveModel(const std::string &filename) const
{
    try
    {
        ::svm_save_model(filename.c_str(), svm_mod_);
    }
    catch (std::exception& e)
    {
        LOG(ERROR) << "Could not save svm model to file " << filename << ". ";
    }
}

void svmClassifier::loadModel(const std::string &filename)
{
    if( !v4r::io::existsFile(filename) )
        throw std::runtime_error("Given config file " + filename + " does not exist! Current working directory is " + boost::filesystem::current_path().string() + ".");

    svm_mod_ = ::svm_load_model(filename.c_str());
}

}
