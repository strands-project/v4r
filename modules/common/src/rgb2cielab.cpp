#include <v4r/common/rgb2cielab.h>
#include <v4r/common/color_transforms.h>
#include <math.h>
#include <algorithm>
#include <glog/logging.h>

namespace v4r
{

void
RGB2CIELAB::initializeLUT()
{
    sRGB_LUT.resize(256);
    sXYZ_LUT.resize(4000);

    #pragma omp parallel for schedule (dynamic)
    for (int i = 0; i < 256; i++)
    {
        float f = i / 255.f;
        if (f > 0.04045f)
            sRGB_LUT[i] = powf ((f + 0.055f) / 1.055f, 2.4f);
        else
            sRGB_LUT[i] = f / 12.92f;
    }

    #pragma omp parallel for schedule (dynamic)
    for (int i = 0; i < 4000; i++)
    {
        float f = i / 4000.f;
        if (f > 0.008856f)
            sXYZ_LUT[i] = powf (f, 0.3333f);
        else
            sXYZ_LUT[i] = (7.787f * f) + (16.f / 116.f);
    }
//    is_initialized_ = true;
}

Eigen::VectorXf
RGB2CIELAB::do_conversion(unsigned char R, unsigned char G, unsigned char B) const
{
    Eigen::VectorXf lab (getOutputNumColorCompenents());

    float fr = sRGB_LUT[R];
    float fg = sRGB_LUT[G];
    float fb = sRGB_LUT[B];

    // Use white = D65
    const float x = fr * 0.412453f + fg * 0.357580f + fb * 0.180423f;
    const float y = fr * 0.212671f + fg * 0.715160f + fb * 0.072169f;
    const float z = fr * 0.019334f + fg * 0.119193f + fb * 0.950227f;

    float vx = x / 0.95047f;
    float vy = y;
    float vz = z / 1.08883f;

    vx = sXYZ_LUT[ std::min<int>(int(vx*4000), 4000-1) ];
    vy = sXYZ_LUT[ std::min<int>(int(vy*4000), 4000-1) ];
    vz = sXYZ_LUT[ std::min<int>(int(vz*4000), 4000-1) ];

    lab(0) = std::min(100.f, std::max(   0.f, 116.f * vy - 16.f) );
    lab(1) = std::min(120.f, std::max(-120.f, 500.f * (vx - vy) ));
    lab(2) = std::min(120.f, std::max(-120.f, 200.f * (vy - vz) ));

    return lab;
}

void
RGB2CIELAB::do_inverse_conversion(const Eigen::VectorXf &converted_color, unsigned char &R, unsigned char &G, unsigned char &B) const
{
    float L = converted_color(0);
    float a = converted_color(1);
    float b = converted_color(2);

    double X, Y, Z;

    // Lab -> normalized XYZ (X,Y,Z are all in 0...1)

    Y = L * (1.0/116.0) + 16.0/116.0;
    X = a * (1.0/500.0) + Y;
    Z = b * (-1.0/200.0) + Y;

    X = X > 6.0/29.0 ? X * X * X : X * (108.0/841.0) - 432.0/24389.0;
    Y = L > 8.0 ? Y * Y * Y : L * (27.0/24389.0);
    Z = Z > 6.0/29.0 ? Z * Z * Z : Z * (108.0/841.0) - 432.0/24389.0;

    // normalized XYZ -> linear sRGB (in 0...1)

    double Rf, Gf, Bf;

    Rf = X * (1219569.0/395920.0)     + Y * (-608687.0/395920.0)    + Z * (-107481.0/197960.0);
    Gf = X * (-80960619.0/87888100.0) + Y * (82435961.0/43944050.0) + Z * (3976797.0/87888100.0);
    Bf = X * (93813.0/1774030.0)      + Y * (-180961.0/887015.0)    + Z * (107481.0/93370.0);

    // linear sRGB -> gamma-compressed sRGB (in 0...1)

    Rf = Rf > 0.0031308 ? pow(Rf, 1.0 / 2.4) * 1.055 - 0.055 : Rf * 12.92;
    Gf = Gf > 0.0031308 ? pow(Gf, 1.0 / 2.4) * 1.055 - 0.055 : Gf * 12.92;
    Bf = Bf > 0.0031308 ? pow(Bf, 1.0 / 2.4) * 1.055 - 0.055 : Bf * 12.92;

    Rf *=255.f;
    Gf *=255.f;
    Bf *=255.f;

    R = static_cast<unsigned char>( std::max(0, std::min( static_cast<int>(Rf+0.5f), 255) ) );
    G = static_cast<unsigned char>( std::max(0, std::min( static_cast<int>(Gf+0.5f), 255) ) );
    B = static_cast<unsigned char>( std::max(0, std::min( static_cast<int>(Bf+0.5f), 255) ) );
}

//void
//RGB2CIELAB::RGB2CIELAB_normalized(unsigned char R, unsigned char G, unsigned char B, float &L, float &A,float &B2)
//{
//    RGB2CIELAB(R,G,B,L,A,B2);
//    L = (L - 50) / 50;
//    A /= 128.0f;
//    B2 /= 128.0f;
//}

//void
//RGB2CIELAB::RGB2CIELAB_normalized(unsigned char R, unsigned char G, unsigned char B, float &L, float &A,float &B2)
//{
//    RGB2CIELAB(R,G,B,L,A,B2);
//    L = (L - 50) / 50;
//    A /= 128.0f;
//    B2 /= 128.0f;
//}

//template V4R_EXPORTS void RGB2CIELAB::do_conversion<pcl::PointXYZRGB>(const typename pcl::PointCloud<pcl::PointXYZRGB> &, Eigen::MatrixXf &, int);
}

