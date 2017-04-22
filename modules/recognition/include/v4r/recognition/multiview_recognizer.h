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

/**
*
*      @author Thomas Faeulhammer (faeulhammer@acin.tuwien.ac.at)
*      @date January, 2017
*      @brief multiview object instance recognizer
*/

#pragma once

#include <vector>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <v4r_config.h>
#include <v4r/recognition/recognition_pipeline.h>
#include <v4r/recognition/local_feature_matching.h>
#include <pcl/recognition/cg/correspondence_grouping.h>

namespace v4r
{

class V4R_EXPORTS MultiviewRecognizerParameter
{
public:
    bool transfer_only_verified_hypotheses_;
    size_t max_views_;  ///<

    bool transfer_keypoint_correspondences_; ///< if true, transfers keypoint correspondences instead of full hypotheses (requires correspondence grouping)
    bool merge_close_hypotheses_; ///< if true, close correspondence clusters (object hypotheses) of the same object model are merged together and this big cluster is refined
    float merge_close_hypotheses_dist_; ///< defines the maximum distance of the centroids in meter for clusters to be merged together
    float merge_close_hypotheses_angle_; ///< defines the maximum angle in degrees for clusters to be merged together

    float min_dist_; ///< minimum distance two points need to be apart to be counted as redundant
    float max_dotp_; ///< maximum dot-product between the surface normals of two oriented points to be counted redundant

    MultiviewRecognizerParameter( ) :
        transfer_only_verified_hypotheses_ (true),
        max_views_(3),
        transfer_keypoint_correspondences_ (false),
        merge_close_hypotheses_ (true),
        merge_close_hypotheses_dist_ (0.02f),
        merge_close_hypotheses_angle_ (10.f),
        min_dist_(0.01f),
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
                ("mv_rec_merge_close_hypotheses", po::value<bool>(&merge_close_hypotheses_)->default_value(merge_close_hypotheses_), "")
                ("mv_rec_transfer_keypoint_correspondences", po::value<bool>(&transfer_keypoint_correspondences_)->default_value(transfer_keypoint_correspondences_), "")
                ("mv_rec_merge_close_hypotheses_dist", po::value<float>(&merge_close_hypotheses_dist_)->default_value(merge_close_hypotheses_dist_), "")
                ("mv_rec_merge_close_hypotheses_angle", po::value<float>(&merge_close_hypotheses_angle_)->default_value(merge_close_hypotheses_angle_), "")
                ("mv_rec_min_dist_", po::value<float>(&min_dist_)->default_value(min_dist_), "")
                ("mv_rec_max_dotp_", po::value<float>(&max_dotp_)->default_value(max_dotp_), "")
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

template<typename PointT>
class V4R_EXPORTS MultiviewRecognizer : public RecognitionPipeline<PointT>
{
private:
    using RecognitionPipeline<PointT>::scene_;
    using RecognitionPipeline<PointT>::scene_normals_;
    using RecognitionPipeline<PointT>::m_db_;
    using RecognitionPipeline<PointT>::obj_hypotheses_;
    using RecognitionPipeline<PointT>::table_plane_;
    using RecognitionPipeline<PointT>::table_plane_set_;

    typename RecognitionPipeline<PointT>::Ptr recognition_pipeline_;



    MultiviewRecognizerParameter param_;

    struct View
    {
        Eigen::Matrix4f camera_pose_;   ///< camera pose of the view which aligns cloud in registered cloud when multiplied
        std::vector< ObjectHypothesesGroup > obj_hypotheses_;   ///< generated object hypotheses

        std::map<std::string, LocalObjectHypothesis<PointT> > local_obj_hypotheses_;   ///< stores feature correspondences
        std::map<std::string, typename LocalObjectModel::ConstPtr> model_keypoints_; ///< object model database used for local recognition
        pcl::PointCloud<pcl::PointXYZ>::ConstPtr scene_cloud_xyz_;
        pcl::PointCloud<pcl::Normal>::ConstPtr scene_cloud_normals_;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        View() :
              camera_pose_ (Eigen::Matrix4f::Identity())
        {}
    };

    std::vector<View> views_;

    // for keypoint correspondence transfer
    typename boost::shared_ptr< pcl::CorrespondenceGrouping<pcl::PointXYZ, pcl::PointXYZ> > cg_algorithm_;  ///< algorithm for correspondence grouping
    std::map<std::string, LocalObjectHypothesis<PointT> > local_obj_hypotheses_;   ///< stores feature correspondences
    std::map<std::string, typename LocalObjectModel::ConstPtr> model_keypoints_; ///< object model database used for local recognition

    pcl::PointCloud<pcl::PointXYZ>::Ptr scene_cloud_xyz_merged_;
    pcl::PointCloud<pcl::Normal>::Ptr scene_cloud_normals_merged_;

public:
    MultiviewRecognizer(const MultiviewRecognizerParameter &p = MultiviewRecognizerParameter() )
        : param_ (p)
    { }

    void
    initialize(const std::string &trained_dir = "", bool retrain = false);

    /**
    * @brief recognize
    */
    void
    do_recognize();

    /**
    * @brief oh_tmp
    * @param rec recognition pipeline (local or global)
    */
    void
    setSingleViewRecognitionPipeline(typename RecognitionPipeline<PointT>::Ptr & rec)
    {
        recognition_pipeline_ = rec;
    }

    /**
    * @brief needNormals
    * @return true if normals are needed, false otherwise
    */
    bool
    needNormals() const
    {
        return recognition_pipeline_->needNormals();
    }

    /**
         * @brief getFeatureType
         * @return
         */
    size_t
    getFeatureType() const
    {
        return recognition_pipeline_->getFeatureType();
    }

    /**
    * @brief requiresSegmentation
    * @return
    */
    bool
    requiresSegmentation() const
    {
        return recognition_pipeline_->requiresSegmentation();
    }

    void
    clear()
    {
        views_.clear();
    }


    /**
     * @brief setCGAlgorithm
     * @param alg
     */
    void
    setCGAlgorithm (const boost::shared_ptr<pcl::CorrespondenceGrouping<pcl::PointXYZ, pcl::PointXYZ> > & alg)
    {
        cg_algorithm_ = alg;
    }

    void
    correspondenceGrouping();

    typedef boost::shared_ptr< MultiviewRecognizer<PointT> > Ptr;
    typedef boost::shared_ptr< MultiviewRecognizer<PointT> const> ConstPtr;
};

}

