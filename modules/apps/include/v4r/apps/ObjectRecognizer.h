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

#include <boost/serialization/vector.hpp>
#include <v4r/apps/CloudSegmenter.h>
#include <v4r/apps/visualization.h>
#include <v4r/common/normals.h>
#include <v4r/core/macros.h>
#include <v4r/features/global_concatenated.h>
#include <v4r/io/filesystem.h>
#include <v4r/keypoints/types.h>
#include <v4r/ml/types.h>
#include <v4r/recognition/local_recognition_pipeline.h>
#include <v4r/recognition/multi_pipeline_recognizer.h>
#include <v4r/recognition/hypotheses_verification.h>
#include <v4r/segmentation/all_headers.h>
#include <v4r/segmentation/types.h>

#pragma once

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

    bool use_multiview_;

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
          use_multiview_ (false)
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

        std::ifstream ifs(filename);
        boost::archive::xml_iarchive ia(ifs);
        ia >> boost::serialization::make_nvp("ObjectRecognizerParameter", *this );
        ifs.close();

        validate();
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
                ;
    }
};

template<typename PointT>
class V4R_EXPORTS ObjectRecognizer
{
private:
    typename v4r::RecognitionPipeline<PointT>::Ptr mrec_; ///< multi-pipeline recognizer
    typename v4r::LocalRecognitionPipeline<PointT>::Ptr local_recognition_pipeline_; ///< local recognition pipeline (member variable just because of visualization of keypoints)
    typename v4r::HypothesisVerification<PointT, PointT>::Ptr hv_; ///< hypothesis verification object
    typename v4r::NormalEstimator<PointT>::Ptr normal_estimator_;    ///< normal estimator used for computing surface normals (currently only used at training)

    typename v4r::ObjectRecognitionVisualizer<PointT>::Ptr rec_vis_; ///< visualization object

    std::vector<ObjectHypothesesGroup<PointT> > generated_object_hypotheses_;
    std::vector<ObjectHypothesesGroup<PointT> > generated_object_hypotheses_refined_;
    std::vector<typename ObjectHypothesis<PointT>::Ptr > verified_hypotheses_;

    typename v4r::apps::CloudSegmenter<PointT>::Ptr cloud_segmenter_; ///< cloud segmenter for plane removal (if enabled)

    bool visualize_; ///< if true, visualizes objects
    bool skip_verification_; ///< if true, will only generate hypotheses but not verify them
    std::string models_dir_;

    ObjectRecognizerParameter param_;

    Camera::Ptr camera_;

    typename Source<PointT>::Ptr model_database_;

    void refinePose(const typename pcl::PointCloud<PointT>::ConstPtr &scene);   ///< does ICP on the generated object hypotheses to refine their pose

public:
    ObjectRecognizer(const ObjectRecognizerParameter &p = ObjectRecognizerParameter() ) :
        visualize_ (false),
        skip_verification_(false),
        param_(p)
    {}

    /**
     * @brief initialize initialize Object recognizer (sets up model database, recognition pipeline and hypotheses verification)
     * @param argc
     * @param argv
     */
    void initialize(int argc, char ** argv)
    {
        std::vector<std::string> arguments(argv + 1, argv + argc);
        initialize(arguments);
    }

    /**
     * @brief initialize initialize Object recognizer (sets up model database, recognition pipeline and hypotheses verification)
     * @param arguments
     */
    void initialize(const std::vector<std::string> &command_line_arguments);

    /**
     * @brief recognize recognize objects in point cloud
     * @param cloud (organized) point cloud
     * @return
     */
    std::vector<typename ObjectHypothesis<PointT>::Ptr >
    recognize(const typename pcl::PointCloud<PointT>::Ptr &cloud);

    /**
     * @brief getObjectHypothesis
     * @return generated object hypothesis
     */
    std::vector<ObjectHypothesesGroup<PointT> >
    getGeneratedObjectHypothesis() const
    {
        return generated_object_hypotheses_;
    }

    typename pcl::PointCloud<PointT>::ConstPtr
    getModel( const std::string &model_name, int resolution_mm ) const
    {
        bool found;
        typename Source<PointT>::ConstPtr mdb = mrec_->getModelDatabase();
        typename Model<PointT>::ConstPtr model = mdb->getModelById("", model_name, found);
        if(!found)
        {
            std::cerr << "Could not find model with name " << model_name << std::endl;
            typename pcl::PointCloud<PointT>::ConstPtr foo;
            return foo;
        }

        return model->getAssembled( resolution_mm );
    }

    std::string getModelsDir() const
    {
        return models_dir_;
    }

    void setModelsDir(const std::string &dir)
    {
        models_dir_ = dir;
    }
};

}

}
