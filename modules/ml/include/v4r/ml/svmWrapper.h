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

#pragma once

#include <v4r/core/macros.h>
#include <v4r/ml/classifier.h>

#include <boost/program_options.hpp>
#include <libsvm/svm.h>
#include <vector>

namespace po = boost::program_options;

namespace v4r
{


class V4R_EXPORTS SVMParameter
{
public:
    int do_cross_validation_; /// if greater 1, performs k-fold cross validation with k equal set by this variable
    int knn_;   ///< return the knn most probably classes when parameter probability is set to true
    ::svm_parameter svm_;

    std::vector<double> cross_validation_range_C_; ///< cross validation range for parameter C (first element minimum, second element maximum, third element step size as a multiplier)
    std::vector<double> cross_validation_range_gamma_; ///< cross validation range for parameter gamma (first element minimum, second element maximum, third element step size as a multiplier)

    SVMParameter(
            int svm_type = ::C_SVC,
            int kernel_type = ::LINEAR, //::RBF,
            //                    int degree = 2,	/* for poly */
            double gamma = -1,	/* for poly/rbf/sigmoid */
            //                    double coef0 = 1,	/* for poly/sigmoid */

            /* these are for training only */
            double cache_size = 100, /* in MB */
            double eps = 0.001,	/* stopping criteria */
            double C = 10,	/* for C_SVC, EPSILON_SVR and NU_SVR */
            int nr_weight = 0,		/* for C_SVC */
            int *weight_label = NULL,	/* for C_SVC */
            double* weight = NULL,		/* for C_SVC */
            //                    double nu = 0.5,	/* for NU_SVC, ONE_CLASS, and NU_SVR */
            //                    double p = 1,	/* for EPSILON_SVR */
            int shrinking = 1,	/* use the shrinking heuristics */
            int probability = 0 /* do probability estimates */
            )
        :
          do_cross_validation_ ( 0 ),
          knn_ (3),
          cross_validation_range_C_( {exp2(-6), exp2(6), 2} ),
          cross_validation_range_gamma_( {exp2(-5), exp2(5), 2} )
    {
        svm_.svm_type = svm_type;
        svm_.kernel_type = kernel_type;
        //                svm_.degree = degree;
        svm_.gamma = gamma;// default 1/ (num_features);
        //                svm_.coef0 = coef0;

        svm_.cache_size = cache_size;
        svm_.eps = eps;
        svm_.C = C;
        svm_.nr_weight = nr_weight;
        svm_.weight_label = weight_label;
        svm_.weight = weight;
        //                svm_.nu = nu;
        //                svm_.p = p;
        svm_.shrinking = shrinking;
        svm_.probability = probability;
    }

    /**
         * @brief init parameters
         * @param command_line_arguments (according to Boost program options library)
         * @return unused parameters (given parameters that were not used in this initialization call)
         */
    std::vector<std::string>
    init(int argc, char **argv)
    {
        std::vector<std::string> arguments(argv + 1, argv + argc);
        return init(arguments);
    }

    /**
         * @brief init parameters
         * @param command_line_arguments (according to Boost program options library)
         * @return unused parameters (given parameters that were not used in this initialization call)
         */
    std::vector<std::string>
    init(const std::vector<std::string> &command_line_arguments)
    {
        po::options_description desc("SVM Classification Parameter\n=====================\n");
        desc.add_options()
                ("help,h", "produce help message")
                ("svm_do_cross_validation", po::value<int>(&do_cross_validation_)->default_value(do_cross_validation_), "if greater 1, performs k-fold cross validation with k equal set by this variable")
                ("svm_knn", po::value<int>(&knn_)->default_value(knn_), "return the knn most probably classes when parameter probability is set to true")
                ("svm_type", po::value<int>(&svm_.svm_type)->default_value(svm_.svm_type), "according to LIBSVM")
                ("svm_kernel_type", po::value<int>(&svm_.kernel_type)->default_value(svm_.kernel_type), "according to LIBSVM")
                ("svm_gamma", po::value<double>(&svm_.gamma)->default_value(svm_.gamma), "for poly/rbf/sigmoid")
                ("svm_cache_size", po::value<double>(&svm_.cache_size)->default_value(svm_.cache_size), "in MB")
                ("svm_eps", po::value<double>(&svm_.eps)->default_value(svm_.eps), "stopping criteria")
                ("svm_C", po::value<double>(&svm_.C)->default_value(svm_.C), "for C_SVC, EPSILON_SVR and NU_SVR")
                ("svm_nr_weight", po::value<int>(&svm_.nr_weight)->default_value(svm_.nr_weight), "")
                ("svm_shrinking", po::value<int>(&svm_.shrinking)->default_value(svm_.shrinking), "use the shrinking heuristics")
                ("svm_probability", po::value<int>(&svm_.probability)->default_value(svm_.probability), "do probability estimates")
                ("svm_cross_validation_range_C", po::value<std::vector<double> >(&cross_validation_range_C_)->multitoken(), "cross validation range for parameter C (first element minimum, second element maximum, third element step size as a multiplier)")
                ("svm_cross_validation_range_gamma", po::value<std::vector<double> >(&cross_validation_range_gamma_)->multitoken(), "cross validation range for parameter gamma (first element minimum, second element maximum, third element step size as a multiplier)")
                ;
        po::variables_map vm;
        po::parsed_options parsed = po::command_line_parser(command_line_arguments).options(desc).allow_unregistered().run();
        std::vector<std::string> to_pass_further = po::collect_unrecognized(parsed.options, po::include_positional);
        po::store(parsed, vm);
        if (vm.count("help")) { std::cout << desc << std::endl; to_pass_further.push_back("-h"); }
        try { po::notify(vm); }
        catch(std::exception& e) {  std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl; }
        return to_pass_further;
    }
};


class V4R_EXPORTS svmClassifier : public Classifier
{
private:
    SVMParameter param_;
    ::svm_model *svm_mod_;

    void
    dokFoldCrossValidation(
            const Eigen::MatrixXf &data_train,
            const Eigen::VectorXi &target_train,
            size_t k = 5,
            double model_para_C_min = exp2(-6),
            double model_para_C_max = exp2(6),
            double step_multiplicator_C = 2,
            double model_para_gamma_min = exp(-5),
            double model_para_gamma_max = exp(5),
            double step_multiplicator_gamma = 2);

public:
    svmClassifier(const SVMParameter &p = SVMParameter()) : param_(p)
    { }

    void
    predict(const Eigen::MatrixXf &query_data, Eigen::MatrixXi &predicted_label) const;

    /**
         * @brief saveModel save current svm model
         * @param filename filename to save trained model
         */
    void
    saveModel(const std::string &filename) const;

    /**
         * @brief loadModel load an SVM model from file
         * @param filename filename to read svm model
         */
    void
    loadModel(const std::string &filename);

    void
    train(const Eigen::MatrixXf &training_data, const Eigen::VectorXi & training_label);

    int getType() const { return ClassifierType::SVM; }

    typedef boost::shared_ptr< svmClassifier > Ptr;
    typedef boost::shared_ptr< svmClassifier const> ConstPtr;
};
}
