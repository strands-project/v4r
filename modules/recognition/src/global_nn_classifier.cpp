
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


#include <v4r/recognition/global_nn_classifier.h>
#include <v4r/io/eigen.h>
#include <v4r/recognition/metrics.h>

namespace v4r
{

template<template<class > class Distance, typename PointInT>
void
GlobalNNClassifier<Distance, PointInT>::loadFeaturesAndCreateFLANN ()
{
    std::vector<ModelTPtr> models = source_->getModels();

    for (size_t i = 0; i < models.size (); i++)
    {
        const ModelTPtr &m = models[i];
        const std::string path = training_dir_ + "/" + m->class_ + "/" + m->id_ + "/" + descr_name_;

        std::vector<std::string> descriptor_files = io::getFilesInDirectory(path, ".*descriptor.*.txt", false);

        for(const std::string &d:descriptor_files)
        {
            const std::string fn = path + "/" + d;
            std::ifstream ifs( fn );
            std::vector< float > descriptor;
            float val;
            while( ifs >> val )
                descriptor.push_back( val );

            flann_model descr_model;
            descr_model.first = m;
            descr_model.second = descriptor;
            flann_models_.push_back (descr_model);
        }
    }

    CHECK(!flann_models_.empty());

    convertToFLANN (flann_models_, flann_data_);
    flann_index_ = new flann::Index<DistT> (flann_data_, flann::LinearIndexParams ());
    flann_index_->buildIndex ();
}

template<template<class > class Distance, typename PointInT>
void
GlobalNNClassifier<Distance, PointInT>::nearestKSearch (flann::Index<DistT> * index, const flann_model &model,
                                                                int k, flann::Matrix<int> &indices,
                                                                flann::Matrix<float> &distances)
{
    flann::Matrix<float> p = flann::Matrix<float> (new float[model.second.size ()], 1, model.second.size ());
    memcpy (&p.ptr ()[0], &model.second[0], p.cols * p.rows * sizeof(float));

    indices = flann::Matrix<int> (new int[k], 1, k);
    distances = flann::Matrix<float> (new float[k], 1, k);
    index->knnSearch (p, indices, distances, k, flann::SearchParams (512));
    delete[] p.ptr ();
}

template<template<class > class Distance, typename PointInT>
void
GlobalNNClassifier<Distance, PointInT>::classify ()
{
    categories_.clear ();
    confidences_.clear ();
    first_nn_category_ = std::string ("");

    estimator_->setInputCloud(input_);
    estimator_->setIndices(indices_);
    std::vector<float> signature;
    Eigen::MatrixXf signature_m;
    estimator_->compute (signature_m);
    if(!signature_m.cols())
        return;

    signature.resize( signature_m.cols() );
    for(size_t f=0; f<signature_m.cols(); f++)
        signature[f] = signature_m(0,f);

    std::vector<index_score> indices_scores;

    ModelTPtr empty;

    flann_model histogram (empty, signature);
    flann::Matrix<int> indices;
    flann::Matrix<float> distances;
    nearestKSearch (flann_index_, histogram, NN_, indices, distances);

    //gather NN-search results
    float score = 0;
    for (size_t i = 0; i < NN_; ++i)
    {
        score = distances[0][i];
        index_score is;
        is.idx_models_ = indices[0][i];
        is.score_ = score;
        indices_scores.push_back (is);
        std::cout << i << ": " << indices[0][i] << " with score " << score << " and model id: " << flann_models_[indices_scores[i].idx_models_].first->class_ << "/" << flann_models_[indices_scores[i].idx_models_].first->id_ <<std::endl;
    }

    std::sort (indices_scores.begin (), indices_scores.end (), sortIndexScoresOp);
    flann_model &fm = flann_models_[indices_scores[0].idx_models_];
    first_nn_category_ = fm.first->class_;

    std::cout << "first id: " << fm.first->id_ << std::endl;

    std::map<std::string, float> category_map;
    size_t num_n = std::min (NN_, indices_scores.size ());    // why?

    std::map<std::string, float>::iterator it;
    float normalization_term = 0;

    for (size_t i = 0; i < num_n; ++i)
    {
        std::string cat = fm.first->class_;
        it = category_map.find (cat);
        if (it == category_map.end ())
        {
            category_map[cat] = 1;
            //category_map[cat] = indices_scores[i].score_;   // is the confidence better if score is higher or lower?
        }
        else
        {
            it->second++;
            //it->second += indices_scores[i].score_;
        }
        normalization_term += indices_scores[i].score_;
    }

    //------ sort classification result by the confidence value---------
    std::vector<index_score> final_indices_scores;
    for (it = category_map.begin (); it != category_map.end (); it++)
    {
        float prob = it->second / static_cast<float> (num_n);
        //float prob = static_cast<float> (it->second) / static_cast<float> (normalization_term);
        //categories_.push_back (it->first);
        //confidences_.push_back (prob);
        index_score is;
        is.model_name_ = it->first;
        //is.idx_input_ = static_cast<int> (idx);
        is.score_ = prob;
        final_indices_scores.push_back (is);
    }

    std::sort (final_indices_scores.begin (), final_indices_scores.end (), sortIndexScoresOpDesc);

    for (size_t i=0; i < final_indices_scores.size(); i++)
    {
        categories_.push_back (final_indices_scores[i].model_name_);
        confidences_.push_back (final_indices_scores[i].score_);
    }
}

template<template<class > class Distance, typename PointInT>
bool
GlobalNNClassifier<Distance, PointInT>::initialize (bool force_retrain)
{
    //use the source to know what has to be trained and what not, checking if the descr_name directory exists
    //unless force_retrain is true, then train everything
    std::vector<ModelTPtr> models = source_->getModels();
    std::cout << "Models size:" << models.size () << std::endl;

    if (force_retrain)
    {
        for (size_t i = 0; i < models.size (); i++)
            source_->removeDescDirectory (*models[i], training_dir_, descr_name_);
    }

    for (size_t i = 0; i < models.size (); i++)
    {
        const ModelTPtr &m = models[i];
        const std::string out_dir = training_dir_ + "/" + m->class_ + "/" + m->id_ + "/" + descr_name_;

        bool view_is_already_trained = false;
        if ( v4r::io::existsFolder(out_dir) )   // check if training directory exists and the number of descriptors is equal to the number of views
        {
            std::vector<std::string> descriptor_files = v4r::io::getFilesInDirectory(out_dir, ".*descriptor.*.txt", false);
            if(descriptor_files.size()== m->views_.size())
                view_is_already_trained = true;
        }

        if ( !view_is_already_trained )
        {
            io::createDirIfNotExist(out_dir);
            LOG(INFO) << "Computing signatures for " << m->class_ << " for id " <<  m->id_ << " with " << m->views_.size() << " views.";

            for (size_t v = 0; v < m->views_.size (); v++)
            {
                std::vector<float> signature;
                Eigen::MatrixXf signature_m;
                estimator_->setInputCloud(m->views_[v]);

                estimator_->compute (signature_m);
                if(!signature_m.cols())
                    continue;

                signature.resize( signature_m.cols() );
                for(size_t f=0; f<signature_m.cols(); f++)
                    signature[f] = signature_m(0,f);

                std::stringstream path_entropy;
                path_entropy << out_dir << "/entropy_" << v << ".txt";
                io::writeFloatToFile (path_entropy.str (), m->self_occlusions_[v]);

                Eigen::Vector4f centroid;
                pcl::compute3DCentroid(*m->views_[v], centroid);
                std::vector<float> centroid_v(3);
                centroid_v[0] = centroid[0];
                centroid_v[1] = centroid[1];
                centroid_v[2] = centroid[2];

                std::stringstream centroid_file;
                centroid_file << out_dir << "/centroid_" << v << ".txt";
                io::writeVectorToFile (centroid_file.str (), centroid_v);

                std::stringstream descriptor_file;
                descriptor_file << out_dir << "/descriptor_" << v << ".txt";
                std::ofstream f (descriptor_file.str().c_str());
                for(size_t j=0; j<signature.size(); j++)
                    f << signature[j] << " ";
                f.close();
            }
        }
        else
            LOG(INFO) << "Model " << m->class_ << " with id " <<  m->id_ << " (" << m->views_.size() << " views) has already been trained.";
    }

    loadFeaturesAndCreateFLANN ();
    return true;
}

//Instantiation
template class V4R_EXPORTS v4r::GlobalNNClassifier<flann::L1, pcl::PointXYZ>;
template class V4R_EXPORTS v4r::GlobalNNClassifier<v4r::Metrics::HistIntersectionUnionDistance, pcl::PointXYZ>;
}


