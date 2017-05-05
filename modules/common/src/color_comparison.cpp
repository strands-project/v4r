#include <v4r/common/color_comparison.h>
#include <glog/logging.h>
#include <math.h>       /* sin */
#include <pcl/common/angles.h>
#include <omp.h>

namespace v4r
{

float CIE76(const Eigen::Vector3f &a, const Eigen::Vector3f &b)
{
    return (a-b).norm();
}

float CIE94_DEFAULT(const Eigen::Vector3f &a, const Eigen::Vector3f &b)
{
    return CIE94( a, b, 1.f, .045f, .015f);
}


float CIE94(const Eigen::Vector3f &a, const Eigen::Vector3f &b, float K1, float K2, float Kl)
{
    float deltaL = a(0) - b(0);
    float deltaA = a(1) - b(1);
    float deltaB = a(2) - b(2);

    float c1 = sqrt( a(1) * a(1) + a(2) * a(2) );
    float c2 = sqrt( b(1) * b(1) + b(2) * b(2) );
    float deltaC = c1 - c2;

    float deltaH = deltaA * deltaA + deltaB * deltaB - deltaC * deltaC;
    deltaH = deltaH < 0 ? 0 : sqrt(deltaH);

    const double sl = 1.0;
    const double kc = 1.0;
    const double kh = 1.0;

    float sc = 1.0f + K1 * c1;
    float sh = 1.0f + K2 * c1;

    float deltaLKlsl = deltaL / (Kl * sl);
    float deltaCkcsc = deltaC / (kc * sc);
    float deltaHkhsh = deltaH / (kh * sh);
    float i = deltaLKlsl * deltaLKlsl + deltaCkcsc * deltaCkcsc + deltaHkhsh * deltaHkhsh;
    return i < 0 ? 0 : sqrt(i);
}

float CIEDE2000(const Eigen::Vector3f &a, const Eigen::Vector3f &b)
{
    //Set weighting factors to 1
    double k_L = 1.0;
    double k_C = 1.0;
    double k_H = 1.0;

    //Calculate Cprime1, Cprime2, Cabbar
    double c_star_1_ab = sqrt(a(1) * a(1) + a(2) * a(2));
    double c_star_2_ab = sqrt(b(1) * b(1) + b(2) * b(2));
    double c_star_average_ab = (c_star_1_ab + c_star_2_ab) / 2;

    double c_star_average_ab_pot7 = c_star_average_ab * c_star_average_ab * c_star_average_ab;
    c_star_average_ab_pot7 *= c_star_average_ab_pot7 * c_star_average_ab;

    double G = 0.5 * (1 - sqrt(c_star_average_ab_pot7 / (c_star_average_ab_pot7 + 6103515625))); //25^7
    double a1_prime = (1. + G) * a(1);
    double a2_prime = (1. + G) * b(1);

    double C_prime_1 = sqrt(a1_prime * a1_prime + a(2) * a(2));
    double C_prime_2 = sqrt(a2_prime * a2_prime + b(2) * b(2));
    //Angles in Degree.
    double h_prime_1 = fmod(((atan2(a(2), a1_prime) * 180. / M_PI) + 360.), 360.);
    double h_prime_2 = fmod(((atan2(b(2), a2_prime) * 180. / M_PI) + 360.), 360.);

    double delta_L_prime = b(0) - a(0);
    double delta_C_prime = C_prime_2 - C_prime_1;

    double h_bar = std::abs(h_prime_1 - h_prime_2);
    double delta_h_prime;
    if (C_prime_1 * C_prime_2 == 0) delta_h_prime = 0;
    else
    {
        if (h_bar <= 180.)
        {
            delta_h_prime = h_prime_2 - h_prime_1;
        }
        else if (h_bar > 180. && h_prime_2 <= h_prime_1)
        {
            delta_h_prime = h_prime_2 - h_prime_1 + 360.;
        }
        else
        {
            delta_h_prime = h_prime_2 - h_prime_1 - 360.;
        }
    }
    double delta_H_prime = 2 * sqrt(C_prime_1 * C_prime_2) * sin(delta_h_prime * M_PI / 360.);

    // Calculate CIEDE2000
    double L_prime_average = (a(0) + b(0)) / 2.0;
    double C_prime_average = (C_prime_1 + C_prime_2) / 2.0;

    //Calculate h_prime_average

    double h_prime_average;
    if (C_prime_1 * C_prime_2 == 0)
        h_prime_average = 0;
    else
    {
        if (h_bar <= 180)
        {
            h_prime_average = (h_prime_1 + h_prime_2) / 2;
        }
        else if (h_bar > 180. && (h_prime_1 + h_prime_2) < 360)
        {
            h_prime_average = (h_prime_1 + h_prime_2 + 360) / 2;
        }
        else
        {
            h_prime_average = (h_prime_1 + h_prime_2 - 360) / 2;
        }
    }
    double L_prime_average_minus_50_square = (L_prime_average - 50);
    L_prime_average_minus_50_square *= L_prime_average_minus_50_square;

    double S_L = 1 + ((.015 * L_prime_average_minus_50_square) / sqrt(20. + L_prime_average_minus_50_square));
    double S_C = 1 + .045 * C_prime_average;
    double T = 1
        - .17 * cos(pcl::deg2rad(h_prime_average - 30))
        + .24 * cos(pcl::deg2rad(h_prime_average * 2))
        + .32 * cos(pcl::deg2rad(h_prime_average * 3 + 6))
        - .2 * cos(pcl::deg2rad(h_prime_average * 4 - 63));
    double S_H = 1 + .015 * T * C_prime_average;
    double h_prime_average_minus_275_div_25_square = (h_prime_average - 275) / (25);
    h_prime_average_minus_275_div_25_square *= h_prime_average_minus_275_div_25_square;
    double delta_theta = 30 * std::exp(-h_prime_average_minus_275_div_25_square);

    double C_prime_average_pot_7 = C_prime_average * C_prime_average * C_prime_average;
    C_prime_average_pot_7 *= C_prime_average_pot_7 * C_prime_average;
    double R_C = 2 * sqrt(C_prime_average_pot_7 / (C_prime_average_pot_7 + 6103515625));

    double R_T = -sin(pcl::deg2rad(2 * delta_theta)) * R_C;

    double delta_L_prime_div_k_L_S_L = delta_L_prime / (S_L * k_L);
    double delta_C_prime_div_k_C_S_C = delta_C_prime / (S_C * k_C);
    double delta_H_prime_div_k_H_S_H = delta_H_prime / (S_H * k_H);

    double CIEDE2000 = sqrt(
        delta_L_prime_div_k_L_S_L * delta_L_prime_div_k_L_S_L
        + delta_C_prime_div_k_C_S_C * delta_C_prime_div_k_C_S_C
        + delta_H_prime_div_k_H_S_H * delta_H_prime_div_k_H_S_H
        + R_T * delta_C_prime_div_k_C_S_C * delta_H_prime_div_k_H_S_H
        );

    return CIEDE2000;
}

//Eigen::VectorXf CIE76(const Eigen::MatrixXf &a, const Eigen::MatrixXf &b)
//{
//    CHECK(a.rows() == b.rows() && a.cols() == b.cols());

//    Eigen::VectorXf diff(a.rows());

//#pragma omp parallel for schedule(dynamic)

//    for(size_t i=0; i<a.rows(); i++)
//        diff(i) = CIE76(a.row(i), b.row(i));

//    return diff;
//}

}
