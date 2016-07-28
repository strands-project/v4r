/******************************************************************************
 * Copyright (c) 2012 Aitor Aldoma, Thomas Faeulhammer
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


#ifndef FAAT_PCL_REC_FRAMEWORK_GLOBAL_PIPELINE_H_
#define FAAT_PCL_REC_FRAMEWORK_GLOBAL_PIPELINE_H_

#include <glog/logging.h>
#include <flann/flann.h>
#include <pcl/common/common.h>
#include <v4r/recognition/global_recognizer.h>
#include <v4r/recognition/source.h>
#include <v4r/common/faat_3d_rec_framework_defines.h>

namespace v4r
{

/**
     * \brief Nearest neighbor search based classification of PCL point type features.
     * FLANN is used to identify a neighborhood, based on which different scoring schemes
     * can be employed to obtain likelihood values for a specified list of classes.
     * Available features: ESF (,VFH, CVFH)
     * See apps/3d_rec_framework/tools/apps/global_classification.cpp for usage
     * \author Aitor Aldoma, Thomas Faeulhammer
     * \date March, 2012
     */
template<template<class > class Distance, typename PointInT>
class V4R_EXPORTS GlobalNNClassifier
{

protected:
    typedef typename pcl::PointCloud<PointInT>::Ptr PointInTPtr;

    std::string training_dir_;  /// @brief directory containing training data

    PointInTPtr input_; /// @brief Point cloud to be classified

    std::vector<int> indices_; /// @brief indices of the object to be classified

    std::vector<std::string> categories_;   /// @brief classification results

    std::vector<float> confidences_;   /// @brief confidences associated to the classification results (normalized to 0...1)

    typename boost::shared_ptr<GlobalEstimator<PointInT> > estimator_; /// @brief estimator used for describing the object

    struct index_score
    {
        int idx_models_;
        float score_;

        std::string model_name_;
    };

    struct sortIndexScores
    {
        bool
        operator() (const index_score& d1, const index_score& d2)
        {
            return d1.score_ < d2.score_;
        }
    } sortIndexScoresOp;

    struct sortIndexScoresDesc
    {
        bool
        operator() (const index_score& d1, const index_score& d2)
        {
            return d1.score_ > d2.score_;
        }
    } sortIndexScoresOpDesc;

    typedef typename pcl::PointCloud<PointInT>::Ptr PointTPtr;
    typedef Distance<float> DistT;
    typedef Model<PointInT> ModelT;
    typedef boost::shared_ptr<ModelT> ModelTPtr;

    /** \brief Model data source */
    typename Source<PointInT>::Ptr source_;

    /** \brief Descriptor name */
    std::string descr_name_;

    typedef std::pair<ModelTPtr, std::vector<float> > flann_model;
    flann::Matrix<float> flann_data_;
    flann::Index<DistT> * flann_index_;
    std::vector<flann_model> flann_models_;

    /** @brief load features from disk and create flann structure */
    void
    loadFeaturesAndCreateFLANN ();

    inline void
    convertToFLANN (const std::vector<flann_model> &models, flann::Matrix<float> &data)
    {
        CHECK(!models.empty());

        data.rows = models.size ();
        data.cols = models[0].second.size (); // number of histogram bins

        flann::Matrix<float> flann_data (new float[models.size () * models[0].second.size ()], models.size (), models[0].second.size ());

        for (size_t i = 0; i < data.rows; ++i)
            for (size_t j = 0; j < data.cols; ++j)
            {
                flann_data.ptr ()[i * data.cols + j] = models[i].second[j];
            }

        data = flann_data;
    }

    void
    nearestKSearch (flann::Index<DistT> * index, const flann_model &model, int k, flann::Matrix<int> &indices, flann::Matrix<float> &distances);

    size_t NN_;
    std::string first_nn_category_;

public:

    GlobalNNClassifier ()
    {
        NN_ = 1;
    }

    ~GlobalNNClassifier ()
    {
    }

    void
    setNN (int nn)
    {
        NN_ = nn;
    }


    /** \brief Initializes the FLANN structure from the provided source */
    bool
    initialize (bool force_retrain = false);


    /** \brief Performs classification */
    void
    classify ();

    /** \brief Sets the model data source */
    void
    setDataSource (const typename boost::shared_ptr<Source<PointInT> > & source)
    {
        source_ = source;
    }

    void
    setDescriptorName (const std::string & name)
    {
        descr_name_ = name;
    }

    /** @brief sets the indices of the object to be classified */
    void
    setIndices (const std::vector<int> & indices)
    {
        indices_ = indices;
    }

    /** \brief Sets the input cloud to be classified */
    void
    setInputCloud (const PointInTPtr & cloud)
    {
        input_ = cloud;
    }

    void
    setTrainingDir (const std::string & dir)
    {
        training_dir_ = dir;
    }

    void
    getCategory (std::vector<std::string> & categories) const
    {
        categories = categories_;
    }

    void
    getConfidence (std::vector<float> & conf) const
    {
        conf = confidences_;
    }

    void
    setFeatureEstimator (const typename boost::shared_ptr<GlobalEstimator<PointInT> > & feat)
    {
        estimator_ = feat;
    }
};
}
#endif /* REC_FRAMEWORK_GLOBAL_PIPELINE_H_ */
