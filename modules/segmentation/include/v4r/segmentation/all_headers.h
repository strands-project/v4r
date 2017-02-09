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

#include <v4r/segmentation/dominant_plane_segmenter.h>
#include <v4r/segmentation/euclidean_segmenter.h>
#include <v4r/segmentation/multiplane_segmenter.h>
#include <v4r/segmentation/organized_multiplane_segmenter.h>
#include <v4r/segmentation/smooth_Euclidean_segmenter.h>
#include <v4r/segmentation/types.h>

namespace v4r
{
/**
 * @brief initSegmenter set up a segmentation object
 * @param method segmentation method as stated in segmentation/types.h
 * @param params boost parameters for segmentation object
 * @return segmenter
 */
template<typename PointT>
typename Segmenter<PointT>::Ptr
initSegmenter(int method = SegmentationType::DominantPlane, const std::string &config_file = std::string())
{
    (void)method;
    (void)config_file;
    typename Segmenter<PointT>::Ptr cast_segmenter;
    std::cerr << "Currently not implemented" << std::endl;
    return cast_segmenter;
}


/**
 * @brief initSegmenter set up a segmentation object
 * @param method segmentation method as stated in segmentation/types.h
 * @param params boost parameters for segmentation object
 * @return segmenter
 */
template<typename PointT>
typename Segmenter<PointT>::Ptr
initSegmenter(int method, std::vector<std::string> &params )
{
    typename Segmenter<PointT>::Ptr cast_segmenter;
    if(method == SegmentationType::DominantPlane)
    {
        DominantPlaneSegmenterParameter param;
        params = param.init(params);
        typename DominantPlaneSegmenter<PointT>::Ptr seg (new DominantPlaneSegmenter<PointT> (param));
        cast_segmenter = boost::dynamic_pointer_cast<Segmenter<PointT> > (seg);
    }
    else if(method == SegmentationType::MultiPlane)
    {
        MultiplaneSegmenterParameter param;
        params = param.init(params);
        typename MultiplaneSegmenter<PointT>::Ptr seg (new MultiplaneSegmenter<PointT> (param));
        cast_segmenter = boost::dynamic_pointer_cast<Segmenter<PointT> > (seg);
    }
    else if(method == SegmentationType::EuclideanSegmentation)
    {
        EuclideanSegmenterParameter param;
        params = param.init(params);
        typename EuclideanSegmenter<PointT>::Ptr seg (new EuclideanSegmenter<PointT> (param));
        cast_segmenter = boost::dynamic_pointer_cast<Segmenter<PointT> > (seg);
    }
    else if(method == SegmentationType::SmoothEuclideanClustering)
    {
        SmoothEuclideanSegmenterParameter param;
        params = param.init(params);
        typename SmoothEuclideanSegmenter<PointT>::Ptr seg (new SmoothEuclideanSegmenter<PointT> (param));
        cast_segmenter = boost::dynamic_pointer_cast<Segmenter<PointT> > (seg);
    }
    else if(method == SegmentationType::OrganizedMultiplaneSegmentation)
    {
        OrganizedMultiplaneSegmenterParameter param;
        params = param.init(params);
        typename OrganizedMultiplaneSegmenter<PointT>::Ptr seg (new OrganizedMultiplaneSegmenter<PointT> (param));
        cast_segmenter = boost::dynamic_pointer_cast<Segmenter<PointT> > (seg);
    }
    else
    {
        std::cerr << "Segmentation method " << method << " not implemented!" << std::endl;
    }

    return cast_segmenter;
}

}
