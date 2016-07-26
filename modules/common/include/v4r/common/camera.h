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

#ifndef V4R_CAMERA_H__
#define V4R_CAMERA_H__

#include <v4r/core/macros.h>
#include <boost/shared_ptr.hpp>
#include <Eigen/Core>
#include <stdlib.h>

namespace v4r
{
class V4R_EXPORTS Camera
{
protected:
    size_t width_, height_;
    float f_, cx_, cy_;
    Eigen::Matrix4f extrinsics_;

public:

    typedef boost::shared_ptr< Camera > Ptr;
    typedef boost::shared_ptr< Camera const> ConstPtr;

    Camera(float focal_length = 525.f, int width = 640, int height = 480, float cx = 319.5f, float cy = 239.5f, const Eigen::Matrix4f &ext = Eigen::Matrix4f::Identity())
        : width_ (width), height_(height), f_(focal_length), cx_(cx), cy_ (cy), extrinsics_ (ext)
    {}

    size_t getWidth() const { return width_; }
    size_t getHeight() const { return height_; }
    float getFocalLength() const { return f_; }
    float getCx() const { return cx_; }
    float getCy() const { return cy_; }
    Eigen::Matrix4f getExtrinsic() const { return extrinsics_; }
    void setExtrinsics( const Eigen::Matrix4f &ext ) { extrinsics_ = ext; }
};

}

#endif
