#include <v4r/common/color_transforms.h>
#include <math.h>
#include <algorithm>

namespace v4r
{
void
ColorTransformOMP::initializeLUT()
{
    omp_set_lock(&initialization_lock_);

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

    omp_unset_lock(&initialization_lock_);
}

void
ColorTransformOMP::RGB2CIELAB (unsigned char R, unsigned char G, unsigned char B, float &L, float &A,float &B2)
{
    if (sRGB_LUT.empty())
        initializeLUT();

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
ColorTransformOMP::RGB2CIELAB_normalized(unsigned char R, unsigned char G, unsigned char B, float &L, float &A,float &B2)
{
    RGB2CIELAB(R,G,B,L,A,B2);
    L = (L - 50) / 50;
    A /= 128.0f;
    B2 /= 128.0f;
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
}
