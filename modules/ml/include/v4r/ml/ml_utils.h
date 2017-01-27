#pragma once
#include <v4r/core/macros.h>
#include <Eigen/Core>

namespace v4r
{

/**
 * @brief sortTrainingData sorts training data such that the labels are in ascending order
 * @param data_train
 * @param target_train
 */
V4R_EXPORTS
void sortTrainingData(Eigen::MatrixXf &data_train, Eigen::VectorXi &target_train);

/**
 * @brief shuffleTrainingData shuffles training data randomly together with the target label
 * @param data_train
 * @param target_train
 */
V4R_EXPORTS
void shuffleTrainingData(Eigen::MatrixXf &data, Eigen::VectorXi &target);


/**
 * @brief computeConfusionMatrix computes confusion matrix
 * @param actual_label actual label
 * @param predicted_label predicted label
 * @return confusion_matrix confusion matrix
 */
Eigen::MatrixXi
computeConfusionMatrix(const Eigen::VectorXi &actual_label, const Eigen::VectorXi &predicted_label, size_t num_classes);

}

