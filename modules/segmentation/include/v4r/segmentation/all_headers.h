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

#include <v4r/segmentation/plane_extractor.h>
#include <v4r/segmentation/segmenter.h>
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
initSegmenter(int method, std::vector<std::string> &params );

/**
 * @brief initPlaneExtractor set up a plane extraction object
 * @param method plane extraction method as stated in segmentation/types.h
 * @param params boost parameters for segmentation object
 * @return plane_extractor
 */
template<typename PointT>
typename PlaneExtractor<PointT>::Ptr
initPlaneExtractor(int method, std::vector<std::string> &params );

}
