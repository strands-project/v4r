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
#include <vector>
#include <Eigen/Core>
#ifndef V4R_APPEARANCE_HISTOGRAM_EQUALIZATION_H_
#define V4R_APPEARANCE_HISTOGRAM_EQUALIZATION_H_

namespace v4r
{

/**
 * @brief
 * The histogram equalization is an approach to enhance a given image.
 * The approach is to design a transformation T such that the gray values in the output is uniformly distributed in [0, 1].
 * Based on http://www.programming-techniques.com/2013/01/histogram-equalization-using-c-image.html
 * @date July 2016
 * @author Thomas Faeulhammer
 */
class V4R_EXPORTS HistogramEqualizer
{
private:

public:
    void equalize(const Eigen::VectorXf &input, Eigen::VectorXf &output);
};
}
#endif
