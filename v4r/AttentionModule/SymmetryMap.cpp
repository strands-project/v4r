 /* * * * * * * * * * * * * * * * * * * * * * * *
 *              Symmetry_map               	*
 *						*
 * Author:   Dominik Kohl			*
 * Email:    e0726126@sudent.tuwien.ac.at	*
 * Date:     15 June 2011 			*
 * Supervisor: Ekaterina Potapova		*
 *						*
 * Based on the Code from Gert Kootstra		*
 * and Niklas Bergstrom [1] and [2]		*
 * (http://www.csc.kth.se/~kootstra/) 		*
 *						*
 *						*
 * * * * * * * * * * * * * * * * * * * * * * * */


 /*
 * DESCRIPTION
 *
 * This is software for the bottom-up detection of unknown objects.
 * It is based on a symmetry-saliency map, as described in [1] and [2]
 *
 * Symmetry is an important Gestalt principle and can be used for
 * figure-ground segregation or to find the centerpoint of an object.
 *
 * The code works with ROS and OpenCV 2.1 or higher and IPP 7.0
 * To get the symmetry-map all usable OpenCV instructions are used
 * and with IPP for speed-optimisation.
 *
 *
 * REFERENCES
 *
 * [1] Kootstra, G., Bergstr&ouml;m, N., & Kragic, D. (2010). Using Symmetry to
 * Select Fixation Points for Segmentation. To be presented at the
 * International Conference on Pattern Recognition (ICPR), August 23-26,
 * 2010, Istanbul, Turkey.
 *
 * [2] Kootstra, G. & Schomaker, L.R.B. (2009) Using Symmetrical
 * Regions-of-Interest to Improve Visual SLAM. In: Proceedings of the
 * International Conference on Intelligent RObots and Systems (IROS),
 * pp. 930-935, Oct 11-15, 2009, St. Louis, USA. doi:
 * 10.1109/IROS.2009.5354402.
*/


#include "SymmetryMap.hpp"

namespace AttentionModule
{
  
SymmetryMapParameters::SymmetryMapParameters()
{
  image = cv::Mat_<float>::zeros(0,0);
  normalization_type = EPUtils::NT_NONE;
  filter_size = 5;
  map = cv::Mat_<float>::zeros(0,0);
  width = 0;
  height = 0;
  startlevel = 1;
  maxlevel = 5;
  R1 = 7;
  R2 = 17;
  S = 16;
  saliencyMapLevel = 0;
}
  
int CalculateSymmetryMap(SymmetryMapParameters &parameters)
{
  if((( (parameters.width == 0) || (parameters.height == 0) ) && ( (parameters.map.rows == 0) || (parameters.map.cols == 0))) ||
     (  (parameters.image.rows == 0) || (parameters.image.cols == 0) ))
  {
    return(AM_IMAGE);
  }

  if((parameters.width == 0) || (parameters.height == 0))
  {
    parameters.height = parameters.map.rows;
    parameters.width  = parameters.map.cols;
  }
  
  if((parameters.image.cols != parameters.width) || (parameters.image.rows != parameters.height) || (parameters.image.channels() != 3))
  {
    return(AM_IMAGE);
  }
  
  cv::Mat image;
  parameters.image.copyTo(image);

  int startlevel = parameters.startlevel;//Set the highest scale
  int maxlevel = parameters.maxlevel;    //Set the smalest scale (All scales between will be calculated)
  int R1 = parameters.R1;                //Set the smaler Box that woun't be calculated
  int R2 = parameters.R2;                //Set the bigger Box
  int S = parameters.S;                  //Set the standard deviaion for the distance between the pixel

  //Variables for Image handle
  cv::Mat image_grey;
  std::vector<cv::Mat> image_pyramid_grey;

  //Variables for calculation of the symmetry_pixel
  int x0, y0, x1, y1, d;
  int dy, dy1, dy2, dx, dx1, dx2;
  float dX, dY, symV;
  float angle, g0, g1, gwf;
  float totalSym=0;

  //LOG
  /*int logArSize = 10000;
  cv::Mat logAr = cv::Mat_<float>::zeros(logArSize,1);
  for(int i=0; i < logArSize; ++i)
  {
    logAr.at<float>(i,1) = log( 1 + sqrt(72)*(float)i/(logArSize-1) );
  }*/

  // Make a Gaussian distance weight array
  float *distanceWeight;
  int lengthDW = 2 * ( (R2*2)*(R2*2) ) + 1;
  distanceWeight = new float[lengthDW];
  for(int i=0; i<lengthDW; i++)
  {
    distanceWeight[i] = (1/(S*sqrt(2*M_PI))) * exp( -i / (2*S*S));
  }

  //PixelAngels
  float **pixelAngles;
  pixelAngles = new float*[R2*4+1];
  for(int i=0; i<R2*4+1; i++)
  {
    pixelAngles[i] = new float[R2*4+1];
  }

  for(int y=-R2*2; y<R2*2+1; y++)
  {
    for(int x=-R2*2; x<R2*2+1; x++)
    {
      pixelAngles[y+R2*2][x+R2*2] = atan2(y, x);
    }
  }

  // Make cos-function from -4pi - 4pi
  int  cosArSize = 10000;
  cv::Mat cosAr = cv::Mat_<float>::zeros(2*cosArSize+1,1);
  for(int i = -cosArSize; i < cosArSize+1; ++i)
  {
    cosAr.at<float>(i+cosArSize,1) = cos(4*M_PI*(float)i/cosArSize);
  }

  // END of LookUp-Tabel Calculation

  if(image.channels() > 1)
    cv::cvtColor(image,image_grey,CV_RGB2GRAY);
  else
    image.copyTo(image_grey);

  //Build image pyramid down to the maximum scale
  cv::buildPyramid(image_grey,image_pyramid_grey,maxlevel);
  std::vector<cv::Mat> image_pyramid;
  image_pyramid.resize(maxlevel+1);

  //Calculate the symmetry for all used scales
  for(int i=startlevel; i<=maxlevel; ++i)
  {
    cv::Mat image_X;
    cv::Mat image_Y;

    cv::Sobel(image_pyramid_grey.at(i),image_X,CV_32F,1,0);
    cv::Sobel(image_pyramid_grey.at(i),image_Y,CV_32F,0,1);

    // Calculate the Angle and Magnitude for every Pixel with cartToPolar
    cv::Mat image_magnitude, image_angle;
    cv::cartToPolar(image_X,image_Y,image_magnitude,image_angle,false);

    image_pyramid.at(i) = cv::Mat_<float>::zeros(image_pyramid_grey.at(i).rows,image_pyramid_grey.at(i).cols);

    for(int y = 0; y < image_pyramid_grey.at(i).rows; ++y)  // START of Iteration over all Pixel ***************************
    {
      for(int x = 0; x < image_pyramid_grey.at(i).cols; ++x)
      {
        // Excluding the borders, since the gradients there are not vaild
        dy1 = std::max(R2 - y +1, 1);
        dy2 = std::max(y + R2 + 1 - image_magnitude.rows + 1, 1);
        dy  = std::max(dy1, dy2);

        dx1 = std::max(R2 - x +1 ,1);
        dx2 = std::max(x + R2 + 1 - image_magnitude.cols + 1, 1);
        dx  = std::max(dx1, dx2);

        // Reset for next Iteration
        symV = 0;
        totalSym = 0;


      // ------------------------
      // |                      |
      // |                      |
      // |      *********       |
      // |      *       *       |
      // |      *       * R1    | R2
      // |      *       *       |
      // |      *********       |
      // |                      |
      // |                      |
      // ------------------------

        for(int j=dy; j < (R2+1); j++)  // Start Iteration over the Mask (Boundary Box) --------------------------------------------
        {
          for(int i = dx; i < (R2*2+1 - dx); i++)
          {
            if((j >= R2) && (i >= R2))  // When at the center of the mask, break
              break;

            if( !((j>(R2-R1)) && (j<(R2+R1)) && (i>(R2-R1)) && (i<(R2+R1))) )
            {
               x0 = x - R2 + i;
               y0 = y - R2 + j;
               x1 = x + R2 - i;
               y1 = y + R2 - j;
               dX = x1 - x0;
               dY = y1 - y0;
               d = (int)rint(dX*dX+dY*dY);  // L2 distance
               // Get the angle of the line between the two pixels use LookUp-table
               angle = pixelAngles[(y1-y0)+R2*2][(x1-x0)+R2*2];

              // Subtract the angle between the two pixels from the gradient angles to get the normalized angles
              g0 = image_angle.at<float>(y0,x0) - angle;
              g1 = image_angle.at<float>(y1,x1) - angle;

              // Calculate the strength of both gradient magnitudes

             //Use normal logarithmus
             gwf = log( ( 1+image_magnitude.at<float>(y0,x0) ) * ( 1+image_magnitude.at<float>(y1,x1) ) );
             //Use LookUp-Table
             symV = (1 - cosAr.at<float>((int)(cosArSize*(g0 + g1)/(4*M_PI))  + cosArSize,1)) *
                    (1 - cosAr.at<float>((int)(cosArSize*(g0 - g1)/(4*M_PI))  + cosArSize,1)) * gwf * distanceWeight[d];

             totalSym += symV;   //Add to the center Point
             // END Iteration over the Mask ------------------------------------------------------------
           }
         }
       }
       //Save the symmetry information of this pixel in the correct scale image
       image_pyramid.at(i).at<float>(y,x) = totalSym;
      // END of Iteration over all Pixel **************************************************************************************
     }
   }
 }

 parameters.map = cv::Mat_<float>::zeros(parameters.height,parameters.width);
 for(int i=startlevel; i <= maxlevel; ++i) // Iterate over all scales ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 {
   cv::Mat temp;
   resize(image_pyramid.at(i),temp,cv::Size(image.cols,image.rows));
   cv::add(parameters.map,temp,parameters.map);
 }
 //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 EPUtils::normalize(parameters.map,parameters.normalization_type);
 
 return(0);
}

} //namespace AttentionModule