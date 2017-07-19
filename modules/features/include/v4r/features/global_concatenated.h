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

#include <v4r_config.h>
#include <v4r/core/macros.h>
#include <v4r/io/filesystem.h>
#include <v4r/features/global_estimator.h>
#ifdef HAVE_CAFFE
#include <v4r/features/global_alexnet_cnn_estimator.h>
#endif
#include <v4r/features/esf_estimator.h>
#include <v4r/features/global_simple_shape_estimator.h>
#include <v4r/features/global_color_estimator.h>
#include <v4r/features/ourcvfh_estimator.h>
#include <v4r/features/types.h>

#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/program_options.hpp>
#include <boost/serialization/serialization.hpp>

#include <fstream>

#include <glog/logging.h>

namespace po = boost::program_options;

namespace v4r
{
class V4R_EXPORTS GlobalConcatEstimatorParameter
{
public:
    int feature_type; ///< Concatenate all feature descriptors which corresponding feature type bit id (v4r/features/types.h) is set in this variable.

    GlobalConcatEstimatorParameter() :
        feature_type ( FeatureType::ESF | FeatureType::SIMPLE_SHAPE | FeatureType::GLOBAL_COLOR )
    { }


    friend class boost::serialization::access;
    template<class Archive> V4R_EXPORTS void serialize(Archive & ar, const unsigned int version)
    {
      ar & BOOST_SERIALIZATION_NVP(feature_type)
        ;
        (void) version;
    }

    void
    save(const std::string &filename) const
    {
        std::ofstream ofs(filename);
        boost::archive::xml_oarchive oa(ofs);
        oa << boost::serialization::make_nvp("GlobalConcatEstimatorParameter", *this );
        ofs.close();
    }

    GlobalConcatEstimatorParameter(const std::string &filename)
    {
        if( !v4r::io::existsFile(filename) )
            throw std::runtime_error("Given config file " + filename + " does not exist! Current working directory is " + boost::filesystem::current_path().string() + ".");

        std::ifstream ifs(filename);
        boost::archive::xml_iarchive ia(ifs);
        ia >> boost::serialization::make_nvp("GlobalConcatEstimatorParameter", *this );
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
        po::options_description desc("Global Concatenate Feature Estimator Parameter\n=====================\n");
        desc.add_options()
                ("help,h", "produce help message")
                ("global_concat_feature_types", po::value<int>(&feature_type)->default_value(feature_type), "Concatenate all feature descriptors which corresponding feature type bit id (v4r/features/types.h) is set in this variable.")
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

/**
 *@brief The GlobalConcatEstimator class implements a global descriptor
 * that combines the description of multiple global descriptor by
 * simple concatenation.
 * @author Thomas Faeulhammer
 */
template<typename PointT>
class V4R_EXPORTS GlobalConcatEstimator : public GlobalEstimator<PointT>
{
private:
    using GlobalEstimator<PointT>::indices_;
    using GlobalEstimator<PointT>::cloud_;
    using GlobalEstimator<PointT>::normals_;
    using GlobalEstimator<PointT>::descr_name_;
    using GlobalEstimator<PointT>::descr_type_;
    using GlobalEstimator<PointT>::feature_dimensions_;
    using GlobalEstimator<PointT>::transforms_;

    bool need_normals_;

    typename ESFEstimation<PointT>::Ptr esf_estimator_;
    typename SimpleShapeEstimator<PointT>::Ptr simple_shape_estimator_;
    typename GlobalColorEstimator<PointT>::Ptr color_estimator_;
    typename OURCVFHEstimator<PointT>::Ptr ourcvfh_estimator_;
#ifdef HAVE_CAFFE
    typename CNN_Feat_Extractor<PointT>::Ptr cnn_feat_estimator_;
#endif

    GlobalConcatEstimatorParameter param_;

public:
    GlobalConcatEstimator( std::vector<std::string> &boost_command_line_arguments,
                           const GlobalConcatEstimatorParameter &p = GlobalConcatEstimatorParameter() );

    bool compute (Eigen::MatrixXf &signature);

    bool needNormals() const { return need_normals_; }

    typedef boost::shared_ptr< GlobalConcatEstimator<PointT> > Ptr;
    typedef boost::shared_ptr< GlobalConcatEstimator<PointT> const> ConstPtr;
};
}
