/******************************************************************************
 * Copyright (c) 2016 Thomas Faeulhammer
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

#ifndef V4R_CLASSIFER_H__
#define V4R_CLASSIFER_H__

#include <v4r/core/macros.h>
#include <v4r/ml/types.h>
#include <boost/shared_ptr.hpp>
#include <Eigen/Eigen>
#include <iostream>
#include <vector>

namespace v4r
{
class V4R_EXPORTS Classifier
{
public:
    Classifier()
    {}

    /**
     * @brief train the classifer
     * @param training_data (each training data point is a row entry, the feature dimensions are equal to the number of columns)
     * @param training_label (the label for each training data point)
     */
    virtual void
    train( const Eigen::MatrixXf &training_data, const Eigen::VectorXi & training_label) = 0;

    /**
     * @brief predict the target value of a query feature
     * @param query_data (each query is a row entry, the feature dimensions are equal to the number of columns)
     * @param predicted_label (each query produces a row of predicted labels, the columns of the predicted labels correspond to the most probable predictions. Predictions are sorted - most likely one is on the left)
     */
    virtual void
    predict(const Eigen::MatrixXf &query_data, Eigen::MatrixXi &predicted_label) = 0;

    virtual void
    computeConfusionMatrix(const Eigen::MatrixXf &test_data,
                           const Eigen::VectorXi &actual_label,
                           Eigen::VectorXi &predicted_label,
                           Eigen::MatrixXi &confusion_matrix)
    {
        (void)test_data;
        (void)actual_label;
        (void)predicted_label;
        (void)confusion_matrix;
        std::cerr << "Computing confusion matrix is not implemented right now." << std::endl;
    }

    virtual void
    getTrainingSampleIDSforPredictions(Eigen::MatrixXi &predicted_training_sample_indices, Eigen::MatrixXf &distances)
    {
        (void)predicted_training_sample_indices;
        (void)distances;
        std::cerr << "getTrainingSampleIDSforPredictions is not implemented right now." << std::endl;
    }

    virtual int
    getType() = 0;

    typedef boost::shared_ptr< Classifier > Ptr;
    typedef boost::shared_ptr< Classifier const> ConstPtr;
};
}

#endif
