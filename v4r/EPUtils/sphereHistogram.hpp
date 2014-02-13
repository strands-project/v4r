#ifndef EPSPHEREHISTOGRAM_HPP
#define EPSPHEREHISTOGRAM_HPP

#include "headers.hpp"
#include "convertions.hpp"

namespace EPUtils
{

struct v1v2new_v{
  unsigned int v1, v2, new_v;
};
  
class FacePatch
{
public:
  unsigned int vs[3];         // vertices
  
  cv::Point3d norm; // normal
  float weight; // what ever you want to accumulate
  
  FacePatch() : weight(0.) {};
};

class SphereHistogram
{
public:
  std::vector<cv::Point3d> vertices;
  std::vector<FacePatch> faces; // 20 icosahedron faces
  
  SphereHistogram();
  void Subdevide();
  void ComputeNormals();
  int FindMatch(cv::Point3d &n);
  
private:
  void InitIcosahedron();
  unsigned int AddMidpoint(unsigned v1, unsigned v2);
  void SubdevideFace(FacePatch &face, std::vector<FacePatch> &newFaces);
  bool findEdge(unsigned int v1,unsigned int v2,unsigned int &new_v);
  
  std::vector<v1v2new_v> checkedVertices;
};

} //namespace EPUtils

#endif //EPSPHEREHISTOGRAM_HPP