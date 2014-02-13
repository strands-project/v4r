#ifndef IKN_SALIENCYMAP
#define IKN_SALIENCYMAP

#include "headers.hpp"
#include "pyramids.hpp"

namespace AttentionModule
{

struct IKNMapParameters
{
  cv::Mat                        image;
  cv::Mat                        R, G, B, Y, I;
  int                            normalization_type;
  int                            width;
  int                            height;
  int                            weightOfColor;
  int                            weightOfIntensities;
  int                            weightOfOrientations;
  int                            numberOfOrientations;
  PyramidParameters              pyramidI;
  PyramidParameters              pyramidRG;
  PyramidParameters              pyramidBY;
  std::vector<PyramidParameters> pyramidO;
  cv::Mat map;
  IKNMapParameters();
};

void CreateColorChannels(IKNMapParameters &parameters);
void createFeatureMaps(IKNMapParameters &parameters);
int CalculateIKNMap(IKNMapParameters &parameters);

} // AttentionModule

#endif //IKN_SALIENCYMAP
