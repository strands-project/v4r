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

#include <boost/program_options.hpp>
#include <v4r/common/graph_geometric_consistency.h>
#include <v4r/io/filesystem.h>
#include <v4r/recognition/local_feature_matching.h>
#include <v4r/recognition/recognition_pipeline.h>

#include <pcl/recognition/cg/correspondence_grouping.h>
#include <omp.h>

namespace po = boost::program_options;


namespace v4r
{

class V4R_EXPORTS LocalRecognitionPipelineParameter
{
public:
    bool merge_close_hypotheses_; ///< if true, close correspondence clusters (object hypotheses) of the same object model are merged together and this big cluster is refined
    float merge_close_hypotheses_dist_; ///< defines the maximum distance of the centroids in meter for clusters to be merged together
    float merge_close_hypotheses_angle_; ///< defines the maximum angle in degrees for clusters to be merged together

    float min_dist_; ///< minimum distance two points need to be apart to be counted as redundant
    float max_dotp_; ///< maximum dot-product between the surface normals of two oriented points to be counted redundant

    LocalRecognitionPipelineParameter( ) :
        merge_close_hypotheses_ (true),
        merge_close_hypotheses_dist_ (0.02f),
        merge_close_hypotheses_angle_ (10.f),
        min_dist_(0.005f),
        max_dotp_(0.95f)
    {}


    void
    save(const std::string &filename) const
    {
        std::ofstream ofs(filename);
        boost::archive::xml_oarchive oa(ofs);
        oa << BOOST_SERIALIZATION_NVP( *this );
        ofs.close();
    }

    void
    load(const std::string &filename)
    {
        if( !v4r::io::existsFile(filename) )
            throw std::runtime_error("Given config file " + filename + " does not exist! Current working directory is " + boost::filesystem::current_path().string() + ".");

        std::ifstream ifs(filename);
        boost::archive::xml_iarchive ia(ifs);
        ia >> BOOST_SERIALIZATION_NVP( *this );
        ifs.close();
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
        po::options_description desc("Local Recognition Pipeline Parameters\n=====================");
        desc.add_options()
                ("help,h", "produce help message")
                ("local_rec_merge_close_hypotheses", po::value<bool>(&merge_close_hypotheses_)->default_value(merge_close_hypotheses_), "")
                ("local_rec_merge_close_hypotheses_dist", po::value<float>(&merge_close_hypotheses_dist_)->default_value(merge_close_hypotheses_dist_), "")
                ("local_rec_merge_close_hypotheses_angle", po::value<float>(&merge_close_hypotheses_angle_)->default_value(merge_close_hypotheses_angle_), "")
                ("local_rec_min_dist_", po::value<float>(&min_dist_)->default_value(min_dist_), "")
                ("local_rec_max_dotp_", po::value<float>(&max_dotp_)->default_value(max_dotp_), "")
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
