#include <v4r/common/color_transforms.h>
#include <math.h>
#include <algorithm>

namespace v4r
{
void
ColorTransformOMP::initializeLUT()
{
    sRGB_LUT.resize(256);
    sXYZ_LUT.resize(4000);

    #pragma omp parallel for schedule (dynamic)
    for (int i = 0; i < 256; i++)
    {
        float f = static_cast<float> (i) / 255.0f;
        if (f > 0.04045)
            sRGB_LUT[i] = powf ((f + 0.055f) / 1.055f, 2.4f);
        else
            sRGB_LUT[i] = f / 12.92f;
    }

    #pragma omp parallel for schedule (dynamic)
    for (int i = 0; i < 4000; i++)
    {
        float f = static_cast<float> (i) / 4000.0f;
        if (f > 0.008856)
            sXYZ_LUT[i] = static_cast<float> (powf (f, 0.3333f));
        else
            sXYZ_LUT[i] = static_cast<float>((7.787 * f) + (16.0 / 116.0));
    }
    is_initialized_ = true;
}

void
ColorTransformOMP::RGB2CIELAB (unsigned char R, unsigned char G, unsigned char B, float &L, float &A,float &B2)
{
    if(!is_initialized_)
        throw std::runtime_error("Color Look-Up table is not initialized. Did you forget to call the initializeLUT function?");

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

    vx = sXYZ_LUT[ std::min(int(vx*4000), 4000-1) ];
    vy = sXYZ_LUT[ std::min(int(vy*4000), 4000-1) ];
    vz = sXYZ_LUT[ std::min(int(vz*4000), 4000-1) ];

    L = 116.0f * vy - 16.0f;
    if (L > 100)
        L = 100.0f;

    A = 500.0f * (vx - vy);
    if (A > 120)
        A = 120.0f;
    else if (A <- 120)
        A = -120.0f;

    B2 = 200.0f * (vy - vz);
    if (B2 > 120)
        B2 = 120.0f;
    else if (B2<- 120)
        B2 = -120.0f;
}

void
ColorTransform::CIELAB2RGB(float L, float a, float b, unsigned char &R, unsigned char &G, unsigned char &B)
{
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

void
ColorTransformOMP::RGB2CIELAB_normalized(unsigned char R, unsigned char G, unsigned char B, float &L, float &A,float &B2)
{
    RGB2CIELAB(R,G,B,L,A,B2);
    L = (L - 50) / 50;
    A /= 128.0f;
    B2 /= 128.0f;
}

template<>
V4R_EXPORTS void
ColorTransformOMP::convertColor<pcl::PointXYZ>(const typename pcl::PointCloud<pcl::PointXYZ> &cloud, Eigen::MatrixXf &color_mat, int color_space)
{
    (void)cloud;
    (void)color_mat;
    (void)color_space;
    std::cerr << "A point cloud without color information cannot be converted. Please provide another point cloud type!" << std::endl;
}

template<typename PointT>
 V4R_EXPORTS  void
ColorTransformOMP::convertColor(const typename pcl::PointCloud<PointT> &cloud, Eigen::MatrixXf &color_mat, int color_space)
{
    size_t num_color_channels = 0;
    switch (color_space)
    {
        case ColorTransformOMP::LAB: case ColorTransformOMP::RGB: num_color_channels = 3; break;
        case ColorTransformOMP::GRAYSCALE: num_color_channels = 1; break;
        default: throw std::runtime_error("Color space not implemented!");
    }

    color_mat = Eigen::MatrixXf::Zero ( cloud.points.size(), num_color_channels);

    #pragma omp parallel for schedule (dynamic)
    for(size_t j=0; j < cloud.points.size(); j++)
    {
        const PointT &p = cloud.points[j];

        switch (color_space)
        {
            case ColorTransformOMP::LAB:
            {
                unsigned char r = (unsigned char)p.r;
                unsigned char g = (unsigned char)p.g;
                unsigned char b = (unsigned char)p.b;
                float LRefm, aRefm, bRefm;
                RGB2CIELAB(r, g, b, LRefm, aRefm, bRefm);
                color_mat(j, 0) = LRefm;
                color_mat(j, 1) = aRefm;
                color_mat(j, 2) = bRefm;
                break;
            }
            case ColorTransformOMP::RGB:
                color_mat(j, 0) = p.r/255.f;
                color_mat(j, 1) = p.g/255.f;
                color_mat(j, 2) = p.b/255.f;
                break;
            case ColorTransformOMP::GRAYSCALE:
                color_mat(j, 0) = .2126 * p.r/255.f + .7152 * p.g/255.f + .0722 * p.b/255.f;
        }
    }
}

void
ColorTransform::RGB2CIELAB (unsigned char R, unsigned char G, unsigned char B, float &L, float &A,float &B2)
{
    if (sRGB_LUT.empty())   // initialize
    {
        sRGB_LUT.resize(256);
        sXYZ_LUT.resize(4000);

#pragma omp parallel for schedule (dynamic)
        for (int i = 0; i < 256; i++)
        {
            float f = static_cast<float> (i) / 255.0f;
            if (f > 0.04045)
                sRGB_LUT[i] = powf ((f + 0.055f) / 1.055f, 2.4f);
            else
                sRGB_LUT[i] = f / 12.92f;
        }

#pragma omp parallel for schedule (dynamic)
        for (int i = 0; i < 4000; i++)
        {
            float f = static_cast<float> (i) / 4000.0f;
            if (f > 0.008856)
                sXYZ_LUT[i] = static_cast<float> (powf (f, 0.3333f));
            else
                sXYZ_LUT[i] = static_cast<float>((7.787 * f) + (16.0 / 116.0));
        }
    }

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

    vx = sXYZ_LUT[ std::min(int(vx*4000), 4000-1) ];
    vy = sXYZ_LUT[ std::min(int(vy*4000), 4000-1) ];
    vz = sXYZ_LUT[ std::min(int(vz*4000), 4000-1) ];

    L = 116.0f * vy - 16.0f;
    if (L > 100)
        L = 100.0f;

    A = 500.0f * (vx - vy);
    if (A > 120)
        A = 120.0f;
    else if (A <- 120)
        A = -120.0f;

    B2 = 200.0f * (vy - vz);
    if (B2 > 120)
        B2 = 120.0f;
    else if (B2<- 120)
        B2 = -120.0f;
}


void
ColorTransform::RGB2CIELAB_normalized(unsigned char R, unsigned char G, unsigned char B, float &L, float &A,float &B2)
{
    RGB2CIELAB(R,G,B,L,A,B2);
    L = (L - 50) / 50;
    A /= 128.0f;
    B2 /= 128.0f;
}

std::vector<float> ColorTransform::sRGB_LUT; //definition required
std::vector<float> ColorTransform::sXYZ_LUT;
std::vector<float> ColorTransformOMP::sRGB_LUT;
std::vector<float> ColorTransformOMP::sXYZ_LUT;
bool ColorTransformOMP::is_initialized_ = false;
template V4R_EXPORTS void ColorTransformOMP::convertColor<pcl::PointXYZRGB>(const typename pcl::PointCloud<pcl::PointXYZRGB> &, Eigen::MatrixXf &, int);
}
