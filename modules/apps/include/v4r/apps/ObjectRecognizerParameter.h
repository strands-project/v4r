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

#include <fstream>
#include <iostream>
#include <vector>

#include <boost/program_options.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

#include <glog/logging.h>

#include <v4r/core/macros.h>
#include <v4r/common/normals.h>
#include <v4r/features/global_concatenated.h>
#include <v4r/keypoints/types.h>
#include <v4r/ml/types.h>
#include <v4r/segmentation/types.h>


namespace po = boost::program_options;


namespace v4r
{

namespace apps
{

class V4R_EXPORTS ObjectRecognizerParameter
{
private:

public:
    std::string hv_config_xml_;
    std::string shot_config_xml_;
    std::vector<std::string> global_recognition_pipeline_config_;
    std::string camera_config_xml_;
    std::string depth_img_mask_;
    std::string sift_config_xml_;

    float cg_size_; ///< Size for correspondence grouping.
    int cg_thresh_; ///< Threshold for correspondence grouping. The lower the more hypotheses are generated, the higher the more confident and accurate. Minimum 3.
    bool use_graph_based_gc_grouping_; ///< if true, uses graph-based geometric consistency grouping

    // pipeline setup
    bool do_sift_;
    bool do_shot_;
    int segmentation_method_;
    std::vector< int > global_feature_types_; ///< Concatenate all feature descriptors which corresponding feature type bit id (v4r/features/types.h) is set in this variable. Each (outer) element will be a seperate global recognition pipeline.
    std::vector< int > classification_methods_;
    int shot_keypoint_extractor_method_;
    int normal_computation_method_; ///< normal computation method
    std::vector<float> keypoint_support_radii_;
    double chop_z_; ///< Cut-off distance in meter

    bool remove_planes_;    ///< if enabled, removes the dominant plane in the input cloud (given thera are at least N inliers)
    float plane_inlier_threshold_; ///< maximum distance for plane inliers
    size_t min_plane_inliers_; ///< required inliers for plane to be removed

    int icp_iterations_;

    // multi-view parameters
    bool use_multiview_; ///< if true, transfers verified hypotheses across views
    bool use_multiview_hv_; ///< if true, verifies hypotheses against the registered scene cloud from all input views
    bool use_change_detection_; ///< if true, uses change detection to find dynamic elements within observation period (only for multi-view recognition)
    float tolerance_for_cloud_diff_; ///< tolerance in meter for change detection's cloud differencing
    size_t min_points_for_hyp_removal_; ///< how many removed points must overlap hypothesis to be also considered removed
    size_t max_views_; ///< maximum number of views used for multi-view recognition (if more views are available, information from oldest views will be ignored)

    ObjectRecognizerParameter()
        :
          hv_config_xml_ ("cfg/hv_config.xml" ),
          shot_config_xml_ ( "cfg/shot_config.xml" ),
          global_recognition_pipeline_config_ (  {} ),//{"cfg/esf_config.xml", "cfg/alexnet_config.xml"} ),
          camera_config_xml_ ( "cfg/camera.xml" ),
          depth_img_mask_ ( "cfg/xtion_depth_mask.png" ),
          sift_config_xml_ ( "cfg/sift_config.xml" ),
          cg_size_ ( 0.01f ),
          cg_thresh_ ( 4 ),
          use_graph_based_gc_grouping_ ( true ),
          do_sift_ ( true ),
          do_shot_ ( false ),
          segmentation_method_ ( SegmentationType::OrganizedConnectedComponents ),
          global_feature_types_ ( { FeatureType::ESF | FeatureType::SIMPLE_SHAPE | FeatureType::GLOBAL_COLOR, FeatureType::ALEXNET }  ),
          classification_methods_ (  { ClassifierType::SVM, 0 }  ),
          shot_keypoint_extractor_method_ (  KeypointType::HARRIS3D  ),
          normal_computation_method_ ( NormalEstimatorType::PCL_INTEGRAL_NORMAL ),
          keypoint_support_radii_ ( {0.04, 0.08} ),
          chop_z_ ( 3.f ),
          remove_planes_ ( true ),
          plane_inlier_threshold_ ( 0.02f ),
          min_plane_inliers_ ( 20000 ),
          icp_iterations_ ( 0 ),
          use_multiview_ (false),
          use_multiview_hv_ (true),
          use_change_detection_ (true),
          tolerance_for_cloud_diff_ (0.02f),
          min_points_for_hyp_removal_ (50),
          max_views_ (3)
    {
        validate();
    }

    void
    validate()
    {
        if( global_feature_types_.size() != classification_methods_.size()
               || global_recognition_pipeline_config_.size()  != classification_methods_.size() )
        {
            size_t minn = std::min<size_t> ( global_feature_types_.size(), classification_methods_.size() ) ;
            minn = std::min<size_t> ( minn, global_recognition_pipeline_config_.size() );

            LOG(ERROR) << "The given parameter for feature types, classification methods " <<
                          "and configuration files for global recognition are not the same size!";
            if(minn)
                LOG(ERROR) << " Will only use the first " << minn << " global recognizers for which all three elements are set! ";
            else
                LOG(ERROR) << "Global recognition is disabled!";

            global_feature_types_.resize(minn);
            classification_methods_.resize(minn);
            global_recognition_pipeline_config_.resize(minn);
        }
    }

    void
    save(const std::string &filename) const
    {
        std::ofstream ofs(filename);
        boost::archive::xml_oarchive oa(ofs);
        oa << boost::serialization::make_nvp("ObjectRecognizerParameter", *this );
        ofs.close();
    }

    void
    load (const std::string &filename)
    {
        if( !v4r::io::existsFile(filename) )
            throw std::runtime_error("Given config file " + filename + " does not exist! Current working directory is " + boost::filesystem::current_path().string() + ".");

        VLOG(1) << "Loading parameters from file " << filename;

        std::ifstream ifs(filename);
        boost::archive::xml_iarchive ia(ifs);
        ia >> boost::serialization::make_nvp("ObjectRecognizerParameter", *this );
        ifs.close();

        validate();
    }

    void
    output() const
    {
        std::stringstream ss;
        boost::archive::text_oarchive oa(ss);
        oa << boost::serialization::make_nvp("ObjectRecognizerParameter", *this );

        LOG(INFO) << "Loaded Parameters: " << std::endl << ss.str();
    }


    /**
     * @brief init parameters
     * @param command_line_arguments (according to Boost program options library)
     * @return unused parameters (given parameters that were not used in this initialization call)
     */
    std::vector<std::string>
    init(const std::vector<std::string> &command_line_arguments)
    {
        po::options_description desc("Object Recognizer Parameters\n=====================");
        desc.add_options()
                ("help,h", "produce help message")
                ("or_hv_config_xml", po::value<std::string>(&hv_config_xml_)->default_value(hv_config_xml_), "")
                ("or_shot_config_xml", po::value<std::string>(&shot_config_xml_)->default_value(shot_config_xml_), "")
                ("or_sift_config_xml", po::value<std::string>(&sift_config_xml_)->default_value(sift_config_xml_), "")
                ("or_cg_size_", po::value<float>(&cg_size_)->default_value(cg_size_), "")
                ("or_cg_thresh_", po::value<int>(&cg_thresh_)->default_value(cg_thresh_), "")
                ("or_remove_planes", po::value<bool>(&remove_planes_)->default_value(remove_planes_), "")
                ("or_use_graph_based_gc_grouping", po::value<bool>(&use_graph_based_gc_grouping_)->default_value(use_graph_based_gc_grouping_), "")
                ("or_use_multiview", po::value<bool>(&use_multiview_)->default_value(use_multiview_), "")
                ("or_use_multiview_hv", po::value<bool>(&use_multiview_hv_)->default_value(use_multiview_hv_), "")
                ("or_use_change_detection", po::value<bool>(&use_change_detection_)->default_value(use_change_detection_), "")
                ("or_multivew_max_views", po::value<size_t>(&max_views_)->default_value(max_views_), "maximum number of views used for multi-view recognition (if more views are available, information from oldest views will be ignored)")
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
        ar & BOOST_SERIALIZATION_NVP(hv_config_xml_)
                & BOOST_SERIALIZATION_NVP(shot_config_xml_)
                & BOOST_SERIALIZATION_NVP(global_recognition_pipeline_config_)
                & BOOST_SERIALIZATION_NVP(camera_config_xml_)
                & BOOST_SERIALIZATION_NVP(depth_img_mask_)
                & BOOST_SERIALIZATION_NVP(sift_config_xml_)
                & BOOST_SERIALIZATION_NVP(cg_size_)
                & BOOST_SERIALIZATION_NVP(cg_thresh_)
                & BOOST_SERIALIZATION_NVP(use_graph_based_gc_grouping_)
                & BOOST_SERIALIZATION_NVP(do_sift_)
                & BOOST_SERIALIZATION_NVP(do_shot_)
                & BOOST_SERIALIZATION_NVP(segmentation_method_)
                & BOOST_SERIALIZATION_NVP(global_feature_types_)
                & BOOST_SERIALIZATION_NVP(classification_methods_)
                & BOOST_SERIALIZATION_NVP(shot_keypoint_extractor_method_)
                & BOOST_SERIALIZATION_NVP(normal_computation_method_)
                & BOOST_SERIALIZATION_NVP(keypoint_support_radii_)
                & BOOST_SERIALIZATION_NVP(chop_z_)
                & BOOST_SERIALIZATION_NVP(remove_planes_)
                & BOOST_SERIALIZATION_NVP(plane_inlier_threshold_)
                & BOOST_SERIALIZATION_NVP(min_plane_inliers_)
                & BOOST_SERIALIZATION_NVP(use_multiview_)
                & BOOST_SERIALIZATION_NVP(use_multiview_hv_)
                & BOOST_SERIALIZATION_NVP(use_change_detection_)
                & BOOST_SERIALIZATION_NVP(max_views_)
                ;
    }
};

}

}
