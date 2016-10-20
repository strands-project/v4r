/**
 *  Copyright (C) 2012  
 *    Ekaterina Potapova
 *    Automation and Control Institute
 *    Vienna University of Technology
 *    Gusshausstraße 25-29
 *    1040 Vienna, Austria
 *    potapova(at)acin.tuwien.ac.at
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


#ifndef NORMALIZATION_HPP
#define NORMALIZATION_HPP

#include <v4r/core/macros.h>
#include "v4r/attention_segmentation/eputils_headers.h"

namespace v4r
{

enum NormalizationTypes
{
  NT_NONE        = 0,
  NT_NONMAX,
  NT_FRINTROP_NORM,
  NT_EMPTY,
  NT_MAX_DIVIDE,
  NT_NONE_REAL,
};
  
V4R_EXPORTS void computeLocalMax(cv::Mat &image, int &numLocalMax, float &averageLocalMax, float threshold = 0);
/**
 * normalizes image
 * */
V4R_EXPORTS void normalize(cv::Mat &map, int normalization_type = NT_NONE, float newMaxValue = 1, float newMinValue = 0);
V4R_EXPORTS void normalizeNonMax(cv::Mat &map);
V4R_EXPORTS void normalizeFrintrop(cv::Mat &map);
V4R_EXPORTS void normalizeMin2Zero(cv::Mat &map);
/**
 * normalizes image by simply dividind image by its maximum value
 * */
V4R_EXPORTS void normalizeMax2One(cv::Mat &map);

} //namespace v4r

#endif //NORMALIZATION_HPP
