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

#pragma once

#include <pcl/io/pcd_io.h>
#include <v4r/features/local_estimator.h>
#include <v4r/features/types.h>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

//This stuff is needed to be able to make the SHOT histograms persistent
POINT_CLOUD_REGISTER_POINT_STRUCT (pcl::Histogram<352>, (float[352], histogram, histogram352) )

namespace v4r
{
    class V4R_EXPORTS SHOTLocalEstimationParameter
    {
        public:
        float support_radius_;

        SHOTLocalEstimationParameter() :
            support_radius_ ( 0.05f)
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
            po::options_description desc("SHOT Parameter\n=====================\n");
            desc.add_options()
                    ("help,h", "produce help message")
                    ("shot_support_radius", po::value<float>(&support_radius_)->default_value(support_radius_), "shot support radius")
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

    template<typename PointT>
      class V4R_EXPORTS SHOTLocalEstimation : public LocalEstimator<PointT>
      {
      private:
          using LocalEstimator<PointT>::indices_;
          using LocalEstimator<PointT>::cloud_;
          using LocalEstimator<PointT>::normals_;
          using LocalEstimator<PointT>::keypoint_indices_;
          using LocalEstimator<PointT>::descr_name_;
          using LocalEstimator<PointT>::descr_type_;
          using LocalEstimator<PointT>::descr_dims_;

          SHOTLocalEstimationParameter param_;

      public:
        SHOTLocalEstimation ( const SHOTLocalEstimationParameter &p = SHOTLocalEstimationParameter() ):
            param_( p )
        {
            descr_name_ = "shot";
            descr_type_ = FeatureType::SHOT;
            descr_dims_ = 352;
        }

        bool
        acceptsIndices() const
        {
          return true;
        }

        void
        compute(std::vector<std::vector<float> > & signatures);

        bool
        needNormals () const
        {
            return true;
        }

        std::string
                getUniqueId() const
        {
            std::stringstream id;
            id << static_cast<int>( param_.support_radius_ * 1000.f );
            return id.str();
        }

        typedef boost::shared_ptr< SHOTLocalEstimation<PointT> > Ptr;
        typedef boost::shared_ptr< SHOTLocalEstimation<PointT> const> ConstPtr;
      };
}
