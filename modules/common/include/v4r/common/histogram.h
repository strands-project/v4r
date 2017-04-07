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



#include <v4r/core/macros.h>
#include <Eigen/Core>
#ifndef V4R_COMMON_HISTOGRAM_H_
#define V4R_COMMON_HISTOGRAM_H_

namespace v4r
{

/**
 * @brief compute histogram of the row entries of a matrix
 * @param[in] data (columns are the elements, rows are the different dimensions
 * @param[out] histogram
 * @param[in] number of bins
 * @param[in] range minimum
 * @param[in] range maximum
 */
V4R_EXPORTS void
computeHistogram (const Eigen::MatrixXf &data, Eigen::MatrixXi &histogram, size_t bins=100, float min=0.f, float max=1.f);



/**
 * @brief compute cumulative histogram
 * @param[in] histogram
 * @param[out] cumulative histogram
 */
V4R_EXPORTS void
computeCumulativeHistogram (const Eigen::VectorXi &histogram, Eigen::VectorXi &cumulative_histogram);



/**
 * @brief computes histogram intersection (does not normalize histograms!)
 * @param[in] histA
 * @param[in] histB
 * @return intersection value
 */
V4R_EXPORTS int
computeHistogramIntersection (const Eigen::VectorXi &histA, const Eigen::VectorXi &histB);



/**
 * @brief shift histogram values by one bin
 * @param[in] hist
 * @param[out] hist_shifted
 * @param[in] direction_is_right (if true, shift histogram to the right. Otherwise to the left)
 */
V4R_EXPORTS void
shiftHistogram (const Eigen::VectorXi &hist, Eigen::VectorXi &hist_shifted, bool direction_is_right=true);



/**
 * @brief specifyHistogram (based on http://fourier.eng.hmc.edu/e161/lectures/contrast_transform/node3.html)
 * @param input_image color values of input image
 * @param desired_color color values of desired image
 * @param bins histogram bins
 * @param min minimum color value
 * @param max maximum color value
 * @return specified histogram
 */
V4R_EXPORTS Eigen::VectorXf
specifyHistogram (const Eigen::VectorXf &input_image, const Eigen::VectorXf &desired_image, size_t bins=100, float min=0.f, float max=1.f);

}

#endif
