/******************************************************************************
 * Copyright (c) 2013 Aitor Aldoma
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
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <v4r/core/macros.h>
#include <vector>

#include <omp.h>

namespace v4r
{

class V4R_EXPORTS ColorTransform
{
public:
    typedef boost::shared_ptr< ColorTransform > Ptr;

    virtual ~ColorTransform() {}

    virtual
    Eigen::VectorXf
    do_conversion(unsigned char R, unsigned char G, unsigned char B) const = 0;

    virtual void
    do_inverse_conversion(const Eigen::VectorXf &converted_color, unsigned char &R, unsigned char &G, unsigned char &B) const
    {
        (void) converted_color;
        (void) R;
        (void) G;
        (void) B;
        std::cerr << "Inverse conversion is not implemented!" << std::endl;
    }

    virtual size_t getOutputNumColorCompenents() const = 0;

    template<typename PointT>
    V4R_EXPORTS void
    convert(const pcl::PointCloud<PointT> &cloud, Eigen::MatrixXf &converted_color) const
    {
        converted_color = Eigen::MatrixXf (cloud.points.size(), getOutputNumColorCompenents());

#pragma omp parallel for schedule (dynamic)
        for(size_t i=0; i < cloud.points.size(); i++)
        {
            const PointT &p = cloud.points[i];
            unsigned char r = (unsigned char)p.r;
            unsigned char g = (unsigned char)p.g;
            unsigned char b = (unsigned char)p.b;
            converted_color.row(i) = do_conversion( r, g, b);
        }
    }
};


class V4R_EXPORTS RGB2GrayScale : public ColorTransform
{
public:
    typedef boost::shared_ptr< RGB2GrayScale > Ptr;

    size_t getOutputNumColorCompenents() const { return 1; }

    Eigen::VectorXf
    do_conversion(unsigned char R, unsigned char G, unsigned char B) const
    {
        Eigen::VectorXf c(1);
        c(0) = 0.2126f * R/255.f + 0.7152f * G/255.f + 0.0722f * B/255.f;
        return c;
    }
};

}
