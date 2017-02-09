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

#pragma once

#include <v4r/ml/classifier.h>
#include <v4r/core/macros.h>
#include <v4r/common/flann.h>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

namespace v4r
{
    class V4R_EXPORTS NearestNeighborClassifierParameter
    {
    public:
        int kdtree_splits_;
        size_t knn_;  ///< nearest neighbors to search for when checking feature descriptions of the scene
        int distance_metric_; ///< defines the norm used for feature matching (1... L1 norm, 2... L2 norm)

        NearestNeighborClassifierParameter(
                int kdtree_splits = 512,
                size_t knn = 1,
                int distance_metric = 2
                )
            : kdtree_splits_ (kdtree_splits),
              knn_ ( knn ),
              distance_metric_ (distance_metric)
        {}


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
            po::options_description desc("Nearest Neighbor Classifier Parameter\n=====================\n");
            desc.add_options()
                    ("help,h", "produce help message")
                    ("nn_kdtree_splits", po::value<int>(&kdtree_splits_)->default_value(kdtree_splits_), "")
                    ("nn_knn", po::value<size_t>(&knn_)->default_value(knn_), "nearest neighbors to search for when checking feature descriptions of the scene")
                    ("nn_distance_metric", po::value<int>(&distance_metric_)->default_value(distance_metric_), "defines the norm used for feature matching (1... L1 norm, 2... L2 norm)")
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

    class V4R_EXPORTS NearestNeighborClassifier : public Classifier
    {
    private:
        EigenFLANN::Ptr flann_;
        boost::shared_ptr<flann::Index<flann::L1<float> > > flann_index_l1_;
        boost::shared_ptr<flann::Index<flann::L2<float> > > flann_index_l2_;
        mutable Eigen::MatrixXi knn_indices_;
        mutable Eigen::MatrixXf knn_distances_;
        Eigen::VectorXi training_label_;
        NearestNeighborClassifierParameter param_;

    public:
        NearestNeighborClassifier(const NearestNeighborClassifierParameter &p = NearestNeighborClassifierParameter() )
            : param_(p)
        {}

        void
        predict(const Eigen::MatrixXf &query_data, Eigen::MatrixXi &predicted_label) const;

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

        int getType() const { return ClassifierType::KNN; }

        typedef boost::shared_ptr< NearestNeighborClassifier> Ptr;
        typedef boost::shared_ptr< NearestNeighborClassifier const> ConstPtr;
    };
}
