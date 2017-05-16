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

#include <Eigen/Eigen>
#include <v4r/core/macros.h>

namespace v4r
{

/**
 * @brief CIE76 standard for color comparison --> sqrt ( norm(L_diff) + norm(A_diff) + norm(B_diff) )
 * @param color matrix a (each row is a color)
 * @param color matrix a (each row is a color)
 * @param diff
 */
V4R_EXPORTS float CIE76(const Eigen::Vector3f &a, const Eigen::Vector3f &b);

V4R_EXPORTS float CIE94(const Eigen::Vector3f &a, const Eigen::Vector3f &b, float K1, float K2, float Kl); // default parameters for graphics
//V4R_EXPORTS float CIE94(const Eigen::Vector3f &a, const Eigen::Vector3f &b, float K1=1.f, float K2=.045f, float Kl=.015f); // default parameters for graphics
//V4R_EXPORTS float CIE94(const Eigen::Vector3f &a, const Eigen::Vector3f &b, float K1=2.f, float K2=.048f, float Kl=.014f); // default parameters for textiles

V4R_EXPORTS float CIE94_DEFAULT(const Eigen::Vector3f &a, const Eigen::Vector3f &b);

V4R_EXPORTS float CIEDE2000(const Eigen::Vector3f &a, const Eigen::Vector3f &b);


enum ColorComparisonMethod
{
    cie76,
    cie94,
    ciede2000
};

}
