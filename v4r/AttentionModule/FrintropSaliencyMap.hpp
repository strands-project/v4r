#ifndef FRINTROP_SALIENCYMAP_HPP
#define FRINTROP_SALIENCYMAP_HPP

#include "headers.hpp"
#include "pyramids.hpp"
#include "ColorMap.hpp"

namespace AttentionModule
{

struct FrintropMapParameters
{
  cv::Mat                        image;
  cv::Mat                        R, G, B, Y, I;
  int                            normalization_type;
  int                            width;
  int                            height;
  int                            numberOfOrientations;
  PyramidParameters              pyramidIOn;
  PyramidParameters              pyramidIOff;
  PyramidParameters              pyramidR;
  PyramidParameters              pyramidG;
  PyramidParameters              pyramidB;
  PyramidParameters              pyramidY;
  std::vector<PyramidParameters> pyramidO;
  cv::Mat map;
  FrintropMapParameters();
};

void CreateColorChannels(FrintropMapParameters &parameters);
void createFeatureMaps(FrintropMapParameters &parameters);
int CalculateFrintropMap(FrintropMapParameters &parameters);

} // AttentionModule

#endif //FRINTROP_SALIENCYMAP_HPP
