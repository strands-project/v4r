#include "MapsCombination.hpp"

namespace AttentionModule
{

int CombineMaps(std::vector<cv::Mat> &maps, cv::Mat &combinedMap, int combination_type, int normalization_type)
{
  if(!maps.size())
    return(AM_ZEROSIZE);

  switch(combination_type)
  {
    case AM_SUM:
      combinedMap = cv::Mat_<float>::zeros(maps.at(0).rows,maps.at(0).cols);
      for(unsigned int i = 0; i < maps.size(); ++i)
      {
        cv::add(maps.at(i),combinedMap,combinedMap);
      }
      combinedMap = combinedMap / maps.size();
      EPUtils::normalize(combinedMap,normalization_type);
      return(AM_OK);
    case AM_MUL:
      combinedMap = cv::Mat_<float>::ones(maps.at(0).rows,maps.at(0).cols);
      for(unsigned int i = 0; i < maps.size(); ++i)
      {
        cv::multiply(maps.at(i),combinedMap,combinedMap);
      }
      EPUtils::normalize(combinedMap,normalization_type);
      return(AM_OK);
    case AM_MIN:
      combinedMap = cv::Mat_<float>::ones(maps.at(0).rows,maps.at(0).cols);
      for(unsigned int i = 0; i < maps.size(); ++i)
      {
        combinedMap = cv::min(maps.at(i),combinedMap);
      }
      EPUtils::normalize(combinedMap,normalization_type);
      return(AM_OK);
    case AM_MAX:
      combinedMap = cv::Mat_<float>::zeros(maps.at(0).rows,maps.at(0).cols);
      for(unsigned int i = 0; i < maps.size(); ++i)
      {
        combinedMap = cv::max(maps.at(i),combinedMap);
      }
      EPUtils::normalize(combinedMap,normalization_type);
      return(AM_OK);
    default:
      combinedMap = cv::Mat_<float>::zeros(maps.at(0).rows,maps.at(0).cols);
      return(AM_PARAMETERS);
  }
}

}
