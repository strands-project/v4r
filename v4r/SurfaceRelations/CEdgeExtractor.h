/**
 *  Copyright (C) 2012  
 *    Andreas Richtsfeld, Johann Prankl, Thomas Mörwald
 *    Automation and Control Institute
 *    Vienna University of Technology
 *    Gusshausstraße 25-29
 *    1170 Vienn, Austria
 *    ari(at)acin.tuwien.ac.at
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
 *  along with this program.  If not, see http://www.gnu.org/licenses/
 */

/**
 * @file CEdgeExtractor.h
 * @author Andreas Richtsfeld
 * @date March 2012
 * @version 0.1
 * @brief Extract canny edges with CEdge
 */


#ifndef CEDGE_EXTRACTOR_H
#define CEDGE_EXTRACTOR_H

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/legacy/compat.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/legacy/blobtrack.hpp>
#include <opencv2/legacy/streams.hpp>

namespace surface
{
  
/**
 * @class CEdgeExtractor
 */
class CEdgeExtractor
{
private:
  bool deb;                     // debug flag
  bool useCol;                  // use color
  int apertureSize;
  std::vector<bool> texture;    // texture indices
  
public:
  
private:
  
  void SobelGrey(IplImage *img, IplImage *dx, IplImage *dy);
  void SobelCol(IplImage *img, IplImage *dx, IplImage *dy);
  void Sobel(IplImage *img, IplImage *dx, IplImage *dy);
  void Canny(IplImage *indx, IplImage *indy, IplImage *idst, double lowThr, double highThr);

  
public:
  CEdgeExtractor();
  ~CEdgeExtractor() {}
  
  /** Extract canny edges from iplImage **/
  void extract(IplImage *iplImage);

  /** Extract canny edges from iplImage **/
  void getTexture(std::vector<bool> &_texture) {_texture = texture;}
  
};
  


// -------------- INLINE FUNCTIONs -------------- //

inline short Max(short a, short b, short c)
{
  return ( std::abs(a)>std::abs(b)?(std::abs(a)>std::abs(c)?a:c):std::abs(b)>std::abs(c)?b:c);
}


/********************* set image pixel *****************************/
inline void SetPx8UC1(IplImage *img, short x, short y, uchar c)
{
  ((uchar*)(img->imageData + img->widthStep*y))[x] = c;
}

inline void SetPx16SC1(IplImage *img, short x, short y, short c)
{
  ((short*)(img->imageData + img->widthStep*y))[x] = c;
}

inline void SetPx8UC3(IplImage *img, short x, short y, uchar r, uchar g, uchar b)
{
  uchar *d =  &((uchar*)(img->imageData + img->widthStep*y))[x*3];
  d[0] = r;
  d[1] = g;
  d[2] = b;
}

inline void SetPx32FC1(IplImage *img, short x, short y, float c)
{
  ((float*)(img->imageData + img->widthStep*y))[x] = c;
}


/******************** get image pixel *******************************/
inline uchar GetPx8UC1(IplImage *img, short x, short y)
{
  return ((uchar*)(img->imageData + img->widthStep*y))[x];
}

inline uchar* GetPx8UC3(IplImage *img, short x, short y)
{
  return &((uchar*)(img->imageData + img->widthStep*y))[x*3];
}

inline float GetPx32FC1(IplImage *img, short x, short y)
{
  return ((float*)(img->imageData + img->widthStep*y))[x];
}

inline short GetPx16SC1(IplImage *img, short x, short y)
{
  return ((short*)(img->imageData + img->widthStep*y))[x];
}

/******************** test image format ****************************/
inline bool IsImage8UC1(IplImage *img)
{
  if (img->depth!=IPL_DEPTH_8U || img->nChannels!=1)
    return false;
  return true;
}

inline bool IsImage8UC3(IplImage *img)
{
  if (img->depth!=IPL_DEPTH_8U || img->nChannels!=3)
    return false;
  return true;
}

inline bool IsImage16SC1(IplImage *img)
{
  if (img->depth!=(int)IPL_DEPTH_16S || img->nChannels!=1)
    return false;
  return true;
}

inline bool IsImage32FC1(IplImage *img)
{
  if (img->depth!=(int)IPL_DEPTH_32F || img->nChannels!=1)
    return false;
  return true;
}

inline bool IsImage32FC3(IplImage *img)
{
  if (img->depth!=(int)IPL_DEPTH_32F || img->nChannels!=3)
    return false;
  return true;
}

inline bool IsImageSizeEqual(IplImage *img1, IplImage *img2)
{
  if (img1->width!=img2->width || img1->height!=img2->height)
    return false;
  return true;
}

inline bool IsImageEqual(IplImage *img1, IplImage *img2)
{
  if (img1->width!=img2->width || img1->height!=img2->height || 
      img1->depth!=img2->depth || img1->nChannels!=img2->nChannels)
    return false;
  return true;
}


}

#endif
