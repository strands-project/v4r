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

#ifndef V4R_COLOR_TRANSFORMS___
#define V4R_COLOR_TRANSFORMS___
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <v4r/core/macros.h>
#include <vector>

#include <omp.h>

namespace v4r
{

class V4R_EXPORTS ColorTransform
{
private:
    static std::vector<float> sRGB_LUT;
    static std::vector<float> sXYZ_LUT;

public:
    /**
     * @brief Converts RGB color in LAB color space defined by CIE
     * @param R (0...255)
     * @param G (0...255)
     * @param B (0...255)
     * @param L (-50...50)
     * @param A (0...128)
     * @param B2 (0...128)
     */
    static void
    RGB2CIELAB (unsigned char R, unsigned char G, unsigned char B, float &L, float &A,float &B2);

    /**
     * @brief Converts RGB color into normalized LAB color space
     * @param R (0...255)
     * @param G (0...255)
     * @param B (0...255)
     * @param L (-1...1)
     * @param A (-1...1)
     * @param B2 (-1...1)
     */
    static void
    RGB2CIELAB_normalized (unsigned char R, unsigned char G, unsigned char B, float &L, float &A,float &B2);

    static void
    CIELAB2RGB(float L, float a, float b, unsigned char &R, unsigned char &G, unsigned char &B);
};

/**
 * @brief The ColorTransformOMP class
 * transforms from one color space to another
 * it can be called from multiple threads (because it has an initialization lock for creating the Look-Up table)
 */
class V4R_EXPORTS ColorTransformOMP
{
private:
    static std::vector<float> sRGB_LUT;
    static std::vector<float> sXYZ_LUT;

    static omp_lock_t initialization_lock;
    static bool is_initialized_;
public:
    ColorTransformOMP()
    {
    }

    /**
     * @brief initializes the Look-Up Table used to convert the color. Must be called before first conversion call!
     */
    static void initializeLUT();

    /**
     * @brief Converts RGB color in LAB color space defined by CIE
     * @param R (0...255)
     * @param G (0...255)
     * @param B (0...255)
     * @param L (-50...50)
     * @param A (0...128)
     * @param B2 (0...128)
     */
    static void
    RGB2CIELAB (unsigned char R, unsigned char G, unsigned char B, float &L, float &A,float &B2);

    /**
     * @brief Converts RGB color into normalized LAB color space
     * @param R (0...255)
     * @param G (0...255)
     * @param B (0...255)
     * @param L (-1...1)
     * @param A (-1...1)
     * @param B2 (-1...1)
     */
    static void
    RGB2CIELAB_normalized (unsigned char R, unsigned char G, unsigned char B, float &L, float &A,float &B2);

    /**
     * @brief converts all rgb color points within point cloud into desired color space and stores it into matrix
     * @param[in] point cloud to be converted
     * @param[out] output matrix (rows correspond to points and cols to color channels)
     */
    template<typename PointT>
     V4R_EXPORTS static void
    convertColor(const typename pcl::PointCloud<PointT> &cloud, Eigen::MatrixXf &color_mat, int color_space=ColorSpace::LAB);

    enum ColorSpace
    {
       LAB,
       RGB,
       GRAYSCALE
    };
};
}
#endif
