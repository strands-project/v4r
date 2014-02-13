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
 * @file CEdgeExtractor.cpp
 * @author Andreas Richtsfeld
 * @date March 2012
 * @version 0.1
 * @brief Extract canny edges with CEdge
 */


#include "CEdgeExtractor.h"

namespace surface
{
  
  
// -- private functions -- //

void CEdgeExtractor::SobelGrey(IplImage *img, IplImage *dx, IplImage *dy)
{
  if(!IsImage8UC1(img)) {
    printf("[CEdgeExtractor::SobelGrey] Input image should be IPL_DEPTH_8U, 1 channels!\n");
    exit(-1);
  }
  if(!IsImage16SC1(dx) || !IsImage16SC1(dy)) {
    printf("[CEdgeExtractor::SobelGrey] Output images should be IPL_DEPTH_16S, 1 channel!\n");
    exit(-1);
  }
  if(!IsImageSizeEqual(img,dx) || !IsImageSizeEqual(img,dy)) {
    printf("[CEdgeExtractor::SobelGrey] Size of images must be equal!\n");
    exit(-1);
  }
  
  cvSobel( img, dx, 1, 0, apertureSize );
  cvSobel( img, dy, 0, 1, apertureSize );
}

void CEdgeExtractor::SobelCol(IplImage *img, IplImage *dx, IplImage *dy)
{
  if(!IsImage8UC3(img)) {
    printf("[CEdgeExtractor::SobelCol] Input image should be IPL_DEPTH_8U, 3 channels!\n");
    exit(-1);
  }
  if(!IsImage16SC1(dx) || !IsImage16SC1(dy)) {
    printf("[CEdgeExtractor::SobelCol] Output images should be IPL_DEPTH_16S, 1 channel!\n");
    exit(-1);
  }
  if(!IsImageSizeEqual(img,dx) || !IsImageSizeEqual(img,dy)) {
    printf("[CEdgeExtractor::SobelCol] Size of images must be equal!\n");
    exit(-1);
  }

  IplImage *edge = cvCreateImage(cvGetSize(img), IPL_DEPTH_16S, img->nChannels );
  short *d, *d_end, *e;

  cvSobel( img, edge, 1, 0, apertureSize );
  
  for (int v=0; v<edge->height; v++) {
    e = (short*)(edge->imageData + edge->widthStep*v);
    d = (short*)(dx->imageData + dx->widthStep*v);
    d_end = d + dx->width;

    for (;d!=d_end; d++, e+=3)
      *d = (short)(Max(e[0],e[1],e[2]));
  }

  cvSobel( img, edge, 0, 1, apertureSize );

  for (int v=0; v<edge->height; v++) {
    e = (short*)(edge->imageData + edge->widthStep*v);
    d = (short*)(dy->imageData + dy->widthStep*v);
    d_end = d + dy->width;

    for (;d!=d_end; d++, e+=3) {
      *d = (short)(Max(e[0],e[1],e[2]));
    }
  }
  cvReleaseImage(&edge);
}

void CEdgeExtractor::Sobel(IplImage *img, IplImage *dx, IplImage *dy)
{
  if (useCol)
    SobelCol(img, dx, dy);
  else
    SobelGrey(img, dx, dy);
}

void CEdgeExtractor::Canny(IplImage *indx, IplImage *indy, IplImage *idst, double lowThr, double highThr)
{
  if(!IsImage16SC1(indx) || !IsImage16SC1(indy)) {
    printf("[CEdgeExtractor::Canny] Input images should be IPL_DEPTH_16S, 1 channel!\n");
    exit(-1);
  }
  if(!IsImage8UC1(idst)) {
    printf("[CEdgeExtractor::Canny] Output image should be IPL_DEPTH_8U, 1 channel!\n");
    exit(-1);
  }
  if (!IsImageSizeEqual(indx,indy) || !IsImageSizeEqual(indx,idst)) {
    printf("[CEdgeExtractor::Canny] Size of images must be equal!\n");
    exit(-1);
  }

  CvMat dxstub, *dx = (CvMat*)indx;
  CvMat dystub, *dy = (CvMat*)indy;
  CvMat dststub, *dst = (CvMat*)idst;

  int low, high;
  void *buffer = 0;
  int* mag_buf[3];
  uchar **stack_top, **stack_bottom = 0;
  CvSize size = cvGetSize(dx);
  int flags = apertureSize;
  uchar* map;
  int mapstep, maxsize;
  int i, j;
  CvMat mag_row;

  dx = cvGetMat( dx, &dxstub );
  dy = cvGetMat( dy, &dystub );
  dst = cvGetMat( dst, &dststub );

  if( flags & CV_CANNY_L2_GRADIENT )
  {
      Cv32suf ul, uh;
      ul.f = (float)lowThr;
      uh.f = (float)highThr;
      low = ul.i;
      high = uh.i;
  }
  else
  {
      low = cvFloor( lowThr );
      high = cvFloor( highThr );
  }
  

  buffer = cvAlloc( (size.width+2)*(size.height+2) + (size.width+2)*3*sizeof(int));
 
  mag_buf[0] = (int*)buffer;
  mag_buf[1] = mag_buf[0] + size.width + 2;
  mag_buf[2] = mag_buf[1] + size.width + 2; 

  map = (uchar*)(mag_buf[2] + size.width + 2);
  mapstep = size.width + 2;

  maxsize = MAX( 1 << 10, size.width*size.height/10 );
  stack_top = stack_bottom = (uchar**)cvAlloc( maxsize*sizeof(stack_top[0]) );

  memset( mag_buf[0], 0, (size.width+2)*sizeof(int) );
  memset( map, 1, mapstep );
  memset( map + mapstep*(size.height + 1), 1, mapstep );

  /* sector numbers 
     (Top-Left Origin)

      1   2   3
       *  *  * 
        * * *  
      0*******0
        * * *  
       *  *  * 
      3   2   1
  */

  #define CE_CANNY_PUSH(d)    *(d) = (uchar)2, *stack_top++ = (d)
  #define CE_CANNY_POP(d)     (d) = *--stack_top

  mag_row = cvMat( 1, size.width, CV_32F );

  // calculate magnitude and angle of gradient, perform non-maxima supression.
  // fill the map with one of the following values:
  //   0 - the pixel might belong to an edge
  //   1 - the pixel can not belong to an edge
  //   2 - the pixel does belong to an edge
  for( i = 0; i <= size.height; i++ )
  {
      int* _mag = mag_buf[(i > 0) + 1] + 1;
      float* _magf = (float*)_mag;
      const short* _dx = (short*)(dx->data.ptr + dx->step*i);
      const short* _dy = (short*)(dy->data.ptr + dy->step*i);
      uchar* _map;
      int x, y;
      int magstep1, magstep2;
      int prev_flag = 0;

      if( i < size.height )
      {
          _mag[-1] = _mag[size.width] = 0;

          if( !(flags & CV_CANNY_L2_GRADIENT) )
              for( j = 0; j < size.width; j++ )
                  _mag[j] = abs(_dx[j]) + abs(_dy[j]);
          else
          {
              for( j = 0; j < size.width; j++ )
              {
                  x = _dx[j]; y = _dy[j];
                  _magf[j] = (float)std::sqrt((double)x*x + (double)y*y);
              }
          }
      }
      else
          memset( _mag-1, 0, (size.width + 2)*sizeof(int) );

      // at the very beginning we do not have a complete ring
      // buffer of 3 magnitude rows for non-maxima suppression
      if( i == 0 )
          continue;

      _map = map + mapstep*i + 1;
      _map[-1] = _map[size.width] = 1;

      _mag = mag_buf[1] + 1; // take the central row
      _dx = (short*)(dx->data.ptr + dx->step*(i-1));
      _dy = (short*)(dy->data.ptr + dy->step*(i-1));

      magstep1 = (int)(mag_buf[2] - mag_buf[1]);
      magstep2 = (int)(mag_buf[0] - mag_buf[1]);

      if( (stack_top - stack_bottom) + size.width > maxsize )
      {
          uchar** new_stack_bottom;
          maxsize = MAX( maxsize * 3/2, maxsize + size.width );
          new_stack_bottom = (uchar**)cvAlloc( maxsize * sizeof(stack_top[0])) ;
          memcpy( new_stack_bottom, stack_bottom, (stack_top - stack_bottom)*sizeof(stack_top[0]) );
          stack_top = new_stack_bottom + (stack_top - stack_bottom);
          cvFree( &stack_bottom );
          stack_bottom = new_stack_bottom;
      }

      for( j = 0; j < size.width; j++ )
      {
          #define CE_CANNY_SHIFT 15
          #define CE_TG22  (int)(0.4142135623730950488016887242097*(1<<CE_CANNY_SHIFT) + 0.5)

          x = _dx[j];
          y = _dy[j];
          int s = x ^ y;
          int m = _mag[j];

          x = abs(x);
          y = abs(y);
          if( m > low )
          {
              int tg22x = x * CE_TG22;
              int tg67x = tg22x + ((x + x) << CE_CANNY_SHIFT);

              y <<= CE_CANNY_SHIFT;

              if( y < tg22x )
              {
                  if( m > _mag[j-1] && m >= _mag[j+1] )
                  {
                      if( m > high && !prev_flag && _map[j-mapstep] != 2 )
                      {
                          CE_CANNY_PUSH( _map + j );
                          prev_flag = 1;
                      }
                      else
                          _map[j] = (uchar)0;
                      continue;
                  }
              }
              else if( y > tg67x )
              {
                  if( m > _mag[j+magstep2] && m >= _mag[j+magstep1] )
                  {
                      if( m > high && !prev_flag && _map[j-mapstep] != 2 )
                      {
                          CE_CANNY_PUSH( _map + j );
                          prev_flag = 1;
                      }
                      else
                          _map[j] = (uchar)0;
                      continue;
                  }
              }
             else
              {
                  s = s < 0 ? -1 : 1;
                  if( m > _mag[j+magstep2-s] && m > _mag[j+magstep1+s] )
                  {
                      if( m > high && !prev_flag && _map[j-mapstep] != 2 )
                      {
                          CE_CANNY_PUSH( _map + j );
                          prev_flag = 1;
                      }
                      else
                          _map[j] = (uchar)0;
                      continue;
                  }
              }
          }
          prev_flag = 0;
          _map[j] = (uchar)1;
      }

      // scroll the ring buffer
      _mag = mag_buf[0];
      mag_buf[0] = mag_buf[1];
      mag_buf[1] = mag_buf[2];
      mag_buf[2] = _mag;
  }

 // now track the edges (hysteresis thresholding)
  while( stack_top > stack_bottom ) {
      uchar* m;
      if( (stack_top - stack_bottom) + 8 > maxsize ) {
          uchar** new_stack_bottom;
          maxsize = MAX( maxsize * 3/2, maxsize + 8 );
          new_stack_bottom = (uchar**)cvAlloc( maxsize * sizeof(stack_top[0])) ;
          memcpy( new_stack_bottom, stack_bottom, (stack_top - stack_bottom)*sizeof(stack_top[0]) );
          stack_top = new_stack_bottom + (stack_top - stack_bottom);
          cvFree( &stack_bottom );
          stack_bottom = new_stack_bottom;
      }

      CE_CANNY_POP(m);

      if( !m[-1] )
          CE_CANNY_PUSH( m - 1 );
      if( !m[1] )
          CE_CANNY_PUSH( m + 1 );
      if( !m[-mapstep-1] )
          CE_CANNY_PUSH( m - mapstep - 1 );
      if( !m[-mapstep] )
          CE_CANNY_PUSH( m - mapstep );
      if( !m[-mapstep+1] )
          CE_CANNY_PUSH( m - mapstep + 1 );
      if( !m[mapstep-1] )
          CE_CANNY_PUSH( m + mapstep - 1 );
      if( !m[mapstep] )
          CE_CANNY_PUSH( m + mapstep );
      if( !m[mapstep+1] )
          CE_CANNY_PUSH( m + mapstep + 1 );
  }

  // the final pass, form the final image
  for( i = 0; i < size.height; i++ )
  {
      const uchar* _map = map + mapstep*(i+1) + 1;
      uchar* _dst = dst->data.ptr + dst->step*i;

      for( j = 0; j < size.width; j++ )
          _dst[j] = (uchar)-(_map[j] >> 1);
  }

  cvFree( &buffer );
  cvFree( &stack_bottom );
}



// -- public CEdgeExtractor -- //
  
CEdgeExtractor::CEdgeExtractor()
{
  deb = false;
  useCol = true;
  apertureSize = 3;
}

void CEdgeExtractor::extract(IplImage *iplImage)
{
  if(deb) printf("[CEdgeExtractor::extract] Start extracting edges.\n");
  texture.clear();
  const IplImage *himg = iplImage;

  IplImage *edges = cvCreateImage(cvGetSize(himg), IPL_DEPTH_8U, 1 );
  IplImage *dx = cvCreateImage(cvGetSize(himg), IPL_DEPTH_16S, 1 );
  IplImage *dy = cvCreateImage(cvGetSize(himg), IPL_DEPTH_16S, 1 );

  Sobel((IplImage*) himg, dx, dy);
  Canny(dx, dy, edges, 5, 140);           /// CANNY PARAMETER => good: 70, 140 (30/140)

  for(int y=0; y<edges->height; y++) {
    for(int x=0; x<edges->width; x++)  {
      int val = ((uchar*)(edges->imageData + edges->widthStep*y))[x];
      if(val == 255)
        texture.push_back(true);
      else
        texture.push_back(false);
    }
  }
  
  cvReleaseImage(&edges);
  cvReleaseImage(&dx);
  cvReleaseImage(&dy);
  if(deb) printf("[CEdgeExtractor::extract] Done.\n");
}
  
}
