/******************************************************************************
 * Copyright (c) 2017 Thomas Faeulhammer
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

#include <v4r/common/graph_geometric_consistency.h>
#include <v4r/io/filesystem.h>
#include <v4r/recognition/local_feature_matching.h>
#include <v4r/recognition/recognition_pipeline.h>

#include <pcl/recognition/cg/correspondence_grouping.h>
#include <omp.h>


namespace v4r
{

class V4R_EXPORTS LocalRecognitionPipelineParameter
{
public:
    bool merge_close_hypotheses_; ///< if true, close correspondence clusters (object hypotheses) of the same object model are merged together and this big cluster is refined
    double merge_close_hypotheses_dist_; ///< defines the maximum distance of the centroids in meter for clusters to be merged together
    double merge_close_hypotheses_angle_; ///< defines the maximum angle in degrees for clusters to be merged together

    LocalRecognitionPipelineParameter(
            bool merge_close_hypotheses = true,
            double merge_close_hypotheses_dist = 0.02f,
            double merge_close_hypotheses_angle = 10.f
            )
        : merge_close_hypotheses_ (merge_close_hypotheses),
          merge_close_hypotheses_dist_ (merge_close_hypotheses_dist),
          merge_close_hypotheses_angle_ (merge_close_hypotheses_angle)
    {}


    void
    save(const std::string &filename) const
    {
        std::ofstream ofs(filename);
        boost::archive::xml_oarchive oa(ofs);
        oa << BOOST_SERIALIZATION_NVP( *this );
        ofs.close();
    }

    LocalRecognitionPipelineParameter(const std::string &filename)
    {
        if( !v4r::io::existsFile(filename) )
            throw std::runtime_error("Given config file " + filename + " does not exist! Current working directory is " + boost::filesystem::current_path().string() + ".");

        std::ifstream ifs(filename);
        boost::archive::xml_iarchive ia(ifs);
        ia >> BOOST_SERIALIZATION_NVP( *this );
        ifs.close();
    }

private:
    friend class boost::serialization::access;
    template<class Archive> V4R_EXPORTS void serialize(Archive & ar, const unsigned int version)
    {
        (void) version;
         ar & BOOST_SERIALIZATION_NVP(merge_close_hypotheses_)
            & BOOST_SERIALIZATION_NVP(merge_close_hypotheses_dist_)
            & BOOST_SERIALIZATION_NVP(merge_close_hypotheses_angle_);
    }
};


/**
 * @brief This class merges keypoint correspondences from several local recognizers and generate object hypotheses.
 * @author Thomas Faeulhammer
 * @date Jan 2017
 */
template<typename PointT>
class V4R_EXPORTS LocalRecognitionPipeline : public RecognitionPipeline<PointT>
{
private:
    using RecognitionPipeline<PointT>::m_db_;
    using RecognitionPipeline<PointT>::normal_estimator_;
    using RecognitionPipeline<PointT>::obj_hypotheses_;
    using RecognitionPipeline<PointT>::scene_;
    using RecognitionPipeline<PointT>::scene_normals_;
    using RecognitionPipeline<PointT>::vis_param_;

    std::vector<typename LocalFeatureMatcher<PointT>::Ptr > local_feature_matchers_; ///< set of local recognizer generating keypoint correspondences

    typename boost::shared_ptr< pcl::CorrespondenceGrouping<pcl::PointXYZ, pcl::PointXYZ> > cg_algorithm_;  ///< algorithm for correspondence grouping
    std::map<std::string, LocalObjectHypothesis<PointT> > local_obj_hypotheses_;   ///< stores feature correspondences
    std::map<std::string, typename LocalObjectModel::ConstPtr> model_keypoints_; ///< object model database used for local recognition
    std::vector< std::map<std::string, size_t > > model_kp_idx_range_start_; ///< since keypoints are coming from multiple local recognizer, we need to store which range belongs to which recognizer. This variable is the starting parting t

    LocalRecognitionPipelineParameter param_;

    /**
     * @brief correspondenceGrouping
     */
    void
    correspondenceGrouping();

public:
    LocalRecognitionPipeline (const LocalRecognitionPipelineParameter &p = LocalRecognitionPipelineParameter() )
     : param_(p)
    { }

    /**
     * @brief setCGAlgorithm
     * @param alg
     */
    void
    setCGAlgorithm (const boost::shared_ptr<pcl::CorrespondenceGrouping<pcl::PointXYZ, pcl::PointXYZ> > & alg)
    {
        cg_algorithm_ = alg;
    }

    void initialize(const std::string &trained_dir, bool force_retrain = false);

    /**
     * @brief recognize
     */
    void
    recognize();

    /**
     * @brief addRecognizer
     * @param l_feature_matcher local feature matcher
     */
    void
    addLocalFeatureMatcher(const typename LocalFeatureMatcher<PointT>::Ptr & l_feature_matcher)
    {
        local_feature_matchers_.push_back( l_feature_matcher );
    }


    /**
     * @brief needNormals
     * @return
     */
    bool
    needNormals() const
    {
        for(size_t r_id=0; r_id < local_feature_matchers_.size(); r_id++)
        {
            if( local_feature_matchers_[r_id]->needNormals())
                return true;
        }

        // Graph-based correspondence grouping requires normals but interface does not exist in base class - so need to try pointer casting
        typename GraphGeometricConsistencyGrouping<pcl::PointXYZ, pcl::PointXYZ>::Ptr gcg_algorithm = boost::dynamic_pointer_cast<  GraphGeometricConsistencyGrouping<pcl::PointXYZ, pcl::PointXYZ> > (cg_algorithm_);
        if( gcg_algorithm )
            return true;

        return false;
    }


    /**
     * @brief getFeatureType
     * @return
     */
    size_t
    getFeatureType() const
    {
        size_t feat_type = 0;
        for(size_t r_id=0; r_id < local_feature_matchers_.size(); r_id++)
            feat_type += local_feature_matchers_[r_id]->getFeatureType();

        return feat_type;
    }

    bool
    requiresSegmentation() const
    {
        return false;
    }


    /**
     * @brief getLocalObjectModelDatabase
     * @return local object model database
     */
    std::map<std::string, typename LocalObjectModel::ConstPtr>
    getLocalObjectModelDatabase() const
    {
        return model_keypoints_;
    }

    typedef boost::shared_ptr< LocalRecognitionPipeline<PointT> > Ptr;
    typedef boost::shared_ptr< LocalRecognitionPipeline<PointT> const> ConstPtr;
};

}
