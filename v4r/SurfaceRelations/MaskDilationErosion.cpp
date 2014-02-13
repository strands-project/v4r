/**
 * @file MaskDilationErosion.cpp
 * @author Richtsfeld
 * @date Januar 2012
 * @version 0.1
 * @brief Dilation and erosion for a mask.
 */

#include "MaskDilationErosion.h"
#include <stdio.h>

namespace surface
{


/************************************************************************************
 * Constructor/Destructor
 */

MaskDilationErosion::MaskDilationErosion()
{
  have_input_mask = false;
  is_erosion = true;
  size = 1;

  width = 640;
  height = 480;
  operation_elem = 0;   // Dilation type = MORPH_RECT
  show_mask = false;
}

MaskDilationErosion::~MaskDilationErosion()
{}

// ================================= Private functions ================================= //
void MaskDilationErosion::CreateMatImage()
{
  mask_src = cv::Mat_<uchar>(height, width);
  mask_src.setTo(0);
  for(unsigned i=0; i<mask.size(); i++) {
    int row = mask[i] / width;
    int col = mask[i] % width;
    mask_src.at<char>(row, col) = 255;
  }
}

void MaskDilationErosion::Dilation()
{
  int dilation_type = 0;
  if( operation_elem == 0 ){ dilation_type = cv::MORPH_RECT; }
  else if( operation_elem == 1 ){ dilation_type = cv::MORPH_CROSS; }
  else if( operation_elem == 2) { dilation_type = cv::MORPH_ELLIPSE; }

  cv::Mat element = cv::getStructuringElement(dilation_type,
                                              cv::Size(2*size + 1, 2*size+1),
                                              cv::Point(size, size));
  cv::dilate(mask_src, mask_dst, element);
}

void MaskDilationErosion::Erosion()
{
  int erosion_type = 0;
  if( operation_elem == 0 ){ erosion_type = cv::MORPH_RECT; }
  else if( operation_elem == 1 ){ erosion_type = cv::MORPH_CROSS; }
  else if( operation_elem == 2) { erosion_type = cv::MORPH_ELLIPSE; }

  cv::Mat element = cv::getStructuringElement(erosion_type,
                                              cv::Size(2*size + 1, 2*size+1),
                                              cv::Point(size, size));
  cv::erode(mask_src, mask_dst, element);
}

void MaskDilationErosion::CreateMask()
{
  mask.clear();
  for(unsigned row=0; row<height; row++)
    for(unsigned col=0; col<width; col++)
      if(mask_dst.at<uchar>(row, col) > 0)
        mask.push_back(row*width + col);
}

// ================================= Public functions ================================= //

void MaskDilationErosion::setErosion(bool erosion = true)
{
  is_erosion = erosion;
}

void MaskDilationErosion::setImageSize(unsigned _width = 640, unsigned _heigth = 480)
{
  width = _width;
  height = _heigth;
}

void MaskDilationErosion::setSize(unsigned _size)
{
  size = _size;
}

void MaskDilationErosion::compute(std::vector<int> &_mask)
{
  mask = _mask;
  CreateMatImage();
  if(is_erosion)
    Erosion();
  else
    Dilation();
  CreateMask();
  _mask = mask;

  // show input/output mask
  if(show_mask) {
    cv::namedWindow("Dilation start", CV_WINDOW_AUTOSIZE);
    cv::imshow("Dilation start", mask_src);
    cv::namedWindow("Dilation result", CV_WINDOW_AUTOSIZE);
    cv::imshow("Dilation result", mask_dst);
    cv::waitKey(0);
  }
}

void MaskDilationErosion::getMask(std::vector<int> &_mask)
{
  _mask = mask;
}

}