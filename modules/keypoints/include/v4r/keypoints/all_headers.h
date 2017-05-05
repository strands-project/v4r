/******************************************************************************
 * Copyright (c) 2016 Thomas Faeulhammer
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

#include <v4r/keypoints/types.h>

#include <v4r/keypoints/harris3d_keypoint_extractor.h>
#include <v4r/keypoints/uniform_sampling_extractor.h>
#include <v4r/keypoints/narf_keypoint_extractor.h>
#include <v4r/keypoints/iss_keypoint_extractor.h>

namespace v4r
{

template<typename PointT>
std::vector<typename KeypointExtractor<PointT>::Ptr>
initKeypointExtractors(int method, std::vector<std::string> &params)
{
    std::vector<typename KeypointExtractor<PointT>::Ptr > keypoint_extractor;

    if(method & KeypointType::UniformSampling)
    {
        UniformSamplingExtractorParameter param;
        params = param.init(params);
        typename UniformSamplingExtractor<PointT>::Ptr ke (new UniformSamplingExtractor<PointT> (param));
        keypoint_extractor.push_back( boost::dynamic_pointer_cast<KeypointExtractor<PointT> > (ke) );
    }
    if(method & KeypointType::ISS)
    {
        IssKeypointExtractorParameter param;
        params = param.init(params);
        typename IssKeypointExtractor<PointT>::Ptr ke (new IssKeypointExtractor<PointT> (param));
        keypoint_extractor.push_back( boost::dynamic_pointer_cast<KeypointExtractor<PointT> > (ke) );
    }
    if(method & KeypointType::NARF)
    {
        NarfKeypointExtractorParameter param;
        params = param.init(params);
        typename NarfKeypointExtractor<PointT>::Ptr ke (new NarfKeypointExtractor<PointT> (param));
        keypoint_extractor.push_back( boost::dynamic_pointer_cast<KeypointExtractor<PointT> > (ke) );
    }
    if(method & KeypointType::HARRIS3D)
    {
        Harris3DKeypointExtractorParameter param;
        params = param.init(params);
        typename Harris3DKeypointExtractor<PointT>::Ptr ke (new Harris3DKeypointExtractor<PointT> (param));
        keypoint_extractor.push_back( boost::dynamic_pointer_cast<KeypointExtractor<PointT> > (ke) );
    }
    if( keypoint_extractor.empty() )
    {
        std::cerr << "Keypoint extractor method " << method << " is not implemented! " << std::endl;
    }

    return keypoint_extractor;
}


}
