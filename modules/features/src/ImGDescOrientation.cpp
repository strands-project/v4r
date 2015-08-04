/**
 * $Id$
 * 
 * Software License Agreement (GNU General Public License)
 *
 *  Copyright (C) 2015:
 *
 *    Johann Prankl, prankl@acin.tuwien.ac.at
 *    Aitor Aldoma, aldoma@acin.tuwien.ac.at
 *
 *      Automation and Control Institute
 *      Vienna University of Technology
 *      Gusshausstraße 25-29
 *      1170 Vienn, Austria
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * @author Johann Prankl, Aitor Aldoma
 *
 */

#include <v4r/features/ImGDescOrientation.h>

namespace v4r
{


using namespace std;


/************************************************************************************
 * Constructor/Destructor
 */
ImGDescOrientation::ImGDescOrientation(const Parameter &p)
 : param(p)
{ 
  hist.resize(360);
}

ImGDescOrientation::~ImGDescOrientation()
{
}

/**
 * ComputeGradients
 */
void ImGDescOrientation::ComputeGradients(const cv::Mat_<unsigned char> &im)
{
  if (param.smooth)
    cv::blur(im, im_smooth, cv::Size(3,3));
  else im_smooth = im;

  cv::Sobel(im_smooth, im_dx, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT );
  cv::Sobel(im_smooth, im_dy, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT );
}

/**
 * ComputeDescriptor
 */
void ImGDescOrientation::ComputeAngle(const cv::Mat_<float> &weight, float &angle)
{
  short *ptr_dx, *ptr_dy;

  hist.clear();
  hist.resize(360,0.);

  float N_RAD_GRAD = 180./M_PI;

  for (int v=1; v<im_dx.rows-1; v++)
  {
    ptr_dx = &im_dx(v,1);
    ptr_dy = &im_dy(v,1);

    for (int u=1; u<im_dx.cols-1; u++, ptr_dx++, ptr_dy++)
    {
      hist[int(atan2(*ptr_dy,*ptr_dx)*N_RAD_GRAD)+180] += float(fabs(*ptr_dx)+fabs(*ptr_dy)) * weight(v,u);
    }
  } 

  float max=0;
  unsigned idx=0;
  for (unsigned i=0; i<hist.size(); i++)
  {
    if (hist[i]>max)
    {
      max = hist[i];
      idx=i;
    }
  }

  angle = idx;
}

/**
 * ComputeLTGaussCirc
 */
void ImGDescOrientation::ComputeLTGaussCirc(const cv::Mat_<unsigned char> &im)
{  
  if (lt_gauss.rows==im.rows && lt_gauss.cols==im.cols)
    return;

  if (im.rows!=im.cols || (im.rows-2)%4 != 0)
    throw std::runtime_error("[ImGDescOrientation::ComputeLTGaussCirc] Invalid patch size!");

  lt_gauss=cv::Mat_<float>(im.rows,im.cols);

  float h_size = im.rows/2;
  float invSqrSigma;

  invSqrSigma = param.sigma*(float)(h_size-1);
  invSqrSigma = -1./(invSqrSigma*invSqrSigma);
  
  for (int v=-h_size; v<h_size; v++)
  {
    for (int u=-h_size; u<h_size; u++)
    {
      lt_gauss(v+h_size,u+h_size) = exp(invSqrSigma*(u*u+v*v));
    }
  }
}



/***************************************************************************************/

/**
 * compute dominant orientation for gradient descriptors
 * we use a 3x3 sobel kernel
 * @param im image patch to compute the descriptor (min. 18x18 = 16x16 + 1px boarder)
 */
void ImGDescOrientation::compute(const cv::Mat_<unsigned char> &im, float &angle)
{
  ComputeLTGaussCirc(im);

  ComputeGradients(im);

  ComputeAngle(lt_gauss, angle);
}

/**
 * compute dominant orientation for gradient descriptors
 * we use a 3x3 sobel kernel
 * @param im image patch (min. 18x18 = 16x16 + 1px boarder)
 */
void ImGDescOrientation::compute(const cv::Mat_<unsigned char> &im, const cv::Mat_<float> &weight, float &angle)
{
  ComputeGradients(im);

  ComputeAngle(weight, angle);
}

}

