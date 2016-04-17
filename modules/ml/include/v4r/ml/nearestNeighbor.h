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

#ifndef V4R_NEAREST_NEIGHBOR_CLASSIFIER_H__
#define V4R_NEAREST_NEIGHBOR_CLASSIFIER_H__

#include <v4r/ml/classifier.h>
#include <v4r/core/macros.h>
#include <v4r/common/flann.h>

namespace v4r
{
    class V4R_EXPORTS NearestNeighborClassifier : public Classifier
    {
    public:

        class V4R_EXPORTS Parameter
        {
        public:
            int kdtree_splits_;
            size_t knn_;  /// @brief nearest neighbors to search for when checking feature descriptions of the scene
            int distance_metric_; /// @brief defines the norm used for feature matching (1... L1 norm, 2... L2 norm)

            Parameter(
                    int kdtree_splits = 512,
                    size_t knn = 1,
                    int distance_metric = 2
                    )
                : kdtree_splits_ (kdtree_splits),
                  knn_ ( knn ),
                  distance_metric_ (distance_metric)
            {}
        }param_;

    private:
        EigenFLANN::Ptr flann_;
        boost::shared_ptr<flann::Index<flann::L1<float> > > flann_index_l1_;
        boost::shared_ptr<flann::Index<flann::L2<float> > > flann_index_l2_;
        Eigen::MatrixXi knn_indices_;
        Eigen::MatrixXf knn_distances_;
        Eigen::VectorXi training_label_;

    public:
        NearestNeighborClassifier(const Parameter &p = Parameter() ) : param_(p)
        {}

        void
        predict(const Eigen::MatrixXf &query_data, Eigen::MatrixXi &predicted_label);

        void
        train(const Eigen::MatrixXf &training_data, const Eigen::VectorXi & training_label);

        /**
         * @brief getTrainingSampleIDSforPredictions
         * @param predicted_training_sample_indices
         * @param distances of the training sample to the corresponding query data
         */
        void
        getTrainingSampleIDSforPredictions(Eigen::MatrixXi &predicted_training_sample_indices, Eigen::MatrixXf &distances)
        {
            predicted_training_sample_indices = knn_indices_;
            distances = knn_distances_;
        }

        int
        getType(){
            return ClassifierType::KNN;
        }

        typedef boost::shared_ptr< NearestNeighborClassifier> Ptr;
        typedef boost::shared_ptr< NearestNeighborClassifier const> ConstPtr;
    };
}

#endif
