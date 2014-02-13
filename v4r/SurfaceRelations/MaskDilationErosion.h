/**
 * @file MaskDilationErosion.h
 * @author Richtsfeld
 * @date Januar 2012
 * @version 0.1
 * @brief Dilation and erosion for a mask.
 */

#ifndef SURFACE_MASK_DILATION_EROSION_H
#define SURFACE_MASK_DILATION_EROSION_H


#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace surface
{

class MaskDilationErosion
{
public:
  
protected:

private:
  bool show_mask;             // show input and output mask of the erosion/dilation

  unsigned width, height;     // Image width and height
  bool is_erosion;            // true for erosion / false for dilation
  
  bool have_input_mask;
  std::vector<int> mask;      // mask

  unsigned size;              // Size of dilation/erosion
  int operation_elem;         // type of dilation/erosion element (0: rect, 1: cross, 2: ellipse)
  
  cv::Mat mask_src;           // Source of mask
  cv::Mat mask_dst;           // Destination of mask after erosion/delation
  
  void CreateMatImage();      // Create image from mask
  void Dilation();            // Process dilation
  void Erosion();             // Process erosion
  void CreateMask();          // Create mask from dilated image
  
public:
  MaskDilationErosion();
  ~MaskDilationErosion();

  /** Set Erosion (or dilation) **/
  void setErosion(bool erosion);
  
  /** Set image size **/
  void setImageSize(unsigned _width, unsigned _heigth);
  
  /** Init amount of dilation **/
  void setSize(unsigned _size);
  
  /** Compute dilation and give mask back **/
  void compute(std::vector<int> &_mask);
  
  /** Get reults from comparison **/
  void getMask(std::vector<int> &_mask);
  
  /** Show the input and output mask **/
  void showMasks(bool _show) {show_mask = _show;}

};

/*************************** INLINE METHODES **************************/

} //--END--

#endif

