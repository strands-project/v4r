#ifndef AM_ERRORS_HPP
#define AM_ERRORS_HPP

namespace AttentionModule
{

enum AttentionModuleErrors
{
  AM_OK                  = 0,
  AM_POINTCLOUD          = 1,
  AM_IMAGE,
  AM_PLANECOEFFICIENTS,
  AM_CAMERAPARAMETRS,
  AM_NORMALCLOUD,
  AM_DIFFERENTSIZES,
  AM_NORMALCOEFFICIENTS,
  AM_CURVATURECLOUD,
  AM_ZEROSIZE,
  AM_PARAMETERS,
  AM_DEPTH,
};

} //namespace AttentionModule

#endif // AM_ERRORS_HPP
