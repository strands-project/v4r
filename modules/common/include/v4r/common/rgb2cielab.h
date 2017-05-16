/******************************************************************************
 * Copyright (c) 2013 Aitor Aldoma
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

#ifndef V4R_RGB2CIELAB_TRANSFORMS___
#define V4R_RGB2CIELAB_TRANSFORMS___

#include <v4r/common/color_transforms.h>

namespace v4r
{

class V4R_EXPORTS RGB2CIELAB : public ColorTransform
{
private:
    std::vector<float> sRGB_LUT;
    std::vector<float> sXYZ_LUT;

//    static omp_lock_t initialization_lock;
//    static bool is_initialized_;

    void initializeLUT();

public:
    typedef boost::shared_ptr< RGB2CIELAB > Ptr;

    RGB2CIELAB()
    {
        initializeLUT();
    }

    /**
     * @brief Converts RGB color in LAB color space defined by CIE
     * @param R (0...255)
     * @param G (0...255)
     * @param B (0...255)
     * @param L (0...100)
     * @param A (approx -170...100)
     * @param B2 (approx -100...150)
     */
    Eigen::VectorXf do_conversion(unsigned char R, unsigned char G, unsigned char B) const;

    /**
     * @brief Converts RGB color into normalized LAB color space
     * @param R (0...255)
     * @param G (0...255)
     * @param B (0...255)
     * @param L (-1...1)
     * @param A (-1...1)
     * @param B2 (-1...1)
     */
//    static void
//    RGB2CIELAB_normalized (unsigned char R, unsigned char G, unsigned char B, float &L, float &A,float &B2);

    void
    do_inverse_conversion(const Eigen::VectorXf &converted_color, unsigned char &R, unsigned char &G, unsigned char &B) const;

    size_t getOutputNumColorCompenents() const { return 3; }
};
}
#endif
