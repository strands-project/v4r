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

#ifndef V4R_SVMWRAPPER_H__
#define V4R_SVMWRAPPER_H__

#include <v4r/ml/classifier.h>
#include <libsvm/svm.h>
#include <vector>

#include <v4r/core/macros.h>

namespace v4r
{
    struct V4R_EXPORTS svmData
    {
        Eigen::VectorXf x;
        int y;
    };


    class V4R_EXPORTS svmClassifier : public Classifier
    {
    private:
        size_t num_classes_;    /// @brief number of target labels
        std::string in_filename_;   /// @brief filename to read svm model. If set (file exists), training is skipped and this model loaded instead.
        std::string out_filename_; /// @brief filename to save trained model

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        class V4R_EXPORTS Parameter
        {
        public:
            bool do_cross_validation_;
            int knn_;
            ::svm_parameter svm_;
            Parameter(
                    bool do_cross_validation = false,
                    int knn = 3,
                    int svm_type = ::C_SVC,
                    int kernel_type = ::LINEAR, //::RBF,
//                    int degree = 2,	/* for poly */
                    double gamma = 0.01,	/* for poly/rbf/sigmoid */
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
                    int probability = 1 /* do probability estimates */
                    )
                :
                    do_cross_validation_ ( do_cross_validation ),
                    knn_ (knn)
            {
                svm_.svm_type = svm_type;
                svm_.kernel_type = kernel_type;
//                svm_.degree = degree;
                svm_.gamma = gamma;// default 1/k;
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
        } param_;


        ::svm_model  *svm_mod_;

        svmClassifier(const Parameter &p = Parameter()) : param_(p)
        {
            num_classes_ = 0;
            out_filename_ = "/tmp/model.svm";
        }

        void
        predict(const Eigen::MatrixXf &query_data, Eigen::MatrixXi &predicted_label);

        void
        saveModel(const std::string &filename);

        void
        computeConfusionMatrix(const Eigen::MatrixXf &test_data,
                               const Eigen::VectorXi &actual_label,
                               Eigen::VectorXi &predicted_label,
                               Eigen::MatrixXi &confusion_matrix);

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

        void
        shuffleTrainingData( Eigen::MatrixXf &data_train, Eigen::VectorXi &target_train);

        void
        train(const Eigen::MatrixXf &training_data, const Eigen::VectorXi & training_label);

        void
        trainSVM(const Eigen::MatrixXf &training_data, const Eigen::VectorXi & training_label);

        void
        setNumClasses(const size_t num_classes)
        {
            num_classes_ =  num_classes;
        }

        void
        setOutFilename( const std::string &fn)
        {
            out_filename_ = fn;
        }

        void
        setInFilename( const std::string &fn)
        {
            in_filename_ = fn;
        }

        void
        sortTrainingData(Eigen::MatrixXf &data_train, Eigen::VectorXi &target_train);

        int
        getType(){
            return ClassifierType::SVM;
        }

        typedef boost::shared_ptr< svmClassifier > Ptr;
        typedef boost::shared_ptr< svmClassifier const> ConstPtr;
};
}
#endif //SVMWRAPPER_H
