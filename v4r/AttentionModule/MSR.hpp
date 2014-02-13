#ifndef MSR_HPP
#define MSR_HPP

#include "headers.hpp"

namespace AttentionModule
{
  
void detectMSR(std::vector<cv::Point> &centers, cv::Mat map_, float th = 0.25);
  
} //namespace AttentionModule

#endif //MSR_HPP