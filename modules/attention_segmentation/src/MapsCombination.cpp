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


#include "v4r/attention_segmentation/MapsCombination.h"

namespace v4r
{

int CombineMaps(std::vector<cv::Mat> &maps, cv::Mat &combinedMap, int combination_type, int normalization_type)
{
  if(!maps.size())
  {
    printf("[ERROR] CombineMaps: there should be at least one map!\n");
    return(AM_ZEROSIZE);
  }

  switch(combination_type)
  {
    case AM_SUM:
      combinedMap = cv::Mat_<float>::zeros(maps.at(0).rows,maps.at(0).cols);
      for(unsigned int i = 0; i < maps.size(); ++i)
      {
        cv::add(maps.at(i),combinedMap,combinedMap);
      }
      combinedMap = combinedMap / maps.size();
      v4r::normalize(combinedMap,normalization_type);
      //v4r::normalize(combinedMap,v4r::NT_NONE);
      return(AM_OK);
    case AM_MUL:
      combinedMap = cv::Mat_<float>::ones(maps.at(0).rows,maps.at(0).cols);
      for(unsigned int i = 0; i < maps.size(); ++i)
      {
        cv::multiply(maps.at(i),combinedMap,combinedMap);
      }
      v4r::normalize(combinedMap,normalization_type);
      //v4r::normalize(combinedMap,v4r::NT_NONE);
      return(AM_OK);
    case AM_MIN:
      combinedMap = cv::Mat_<float>::ones(maps.at(0).rows,maps.at(0).cols);
      for(unsigned int i = 0; i < maps.size(); ++i)
      {
        combinedMap = cv::min(maps.at(i),combinedMap);
      }
      v4r::normalize(combinedMap,normalization_type);
      //v4r::normalize(combinedMap,v4r::NT_NONE);
      return(AM_OK);
    case AM_MAX:
      combinedMap = cv::Mat_<float>::zeros(maps.at(0).rows,maps.at(0).cols);
      for(unsigned int i = 0; i < maps.size(); ++i)
      {
        combinedMap = cv::max(maps.at(i),combinedMap);
      }
      v4r::normalize(combinedMap,normalization_type);
      //v4r::normalize(combinedMap,v4r::NT_NONE);
      return(AM_OK);
    default:
      combinedMap = cv::Mat_<float>::zeros(maps.at(0).rows,maps.at(0).cols);
      return(AM_PARAMETERS);
  }
}

}
