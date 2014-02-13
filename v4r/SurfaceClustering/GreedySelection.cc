/**
 *  Copyright (C) 2012  
 *    Andreas Richtsfeld, Johann Prankl, Thomas Mörwald
 *    Automation and Control Institute
 *    Vienna University of Technology
 *    Gusshausstraße 25-29
 *    1170 Vienn, Austria
 *    ari(at)acin.tuwien.ac.at
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


#include "GreedySelection.hh"

namespace surface 
{

using namespace std;

static const bool CmpNumPoints(const SurfaceModel::Ptr &a, const SurfaceModel::Ptr &b)
{
  return (a->indices.size() > b->indices.size());
}


/********************** GreedySelection ************************
 * Constructor/Destructor
 */
GreedySelection::GreedySelection(Parameter p)
 : width(0), height(0)
{
  setParameter(p);
  idMap.resize(640*480);
  mask.resize(640*480);
}

GreedySelection::~GreedySelection()
{
}



/************************** PRIVATE ************************/

/**
 * ComputeSavings
 */
float GreedySelection::ComputeSavings(std::vector<float> &probs)
{
  float savings = probs.size() - param.kappa1;
  float err = 0.;

  for( unsigned i = 0; i < probs.size(); i++ )
    err += (1. - probs[i]);

  savings -= param.kappa2 * err;

  return (savings > 0 ? savings : 0);
}


/**
 * IndexToPointSurfaceProbs
 */
void GreedySelection::IndexToPointSurfaceProbs(SurfaceModel &surf, int idx)
{
  for (unsigned i=0; i<surf.indices.size(); i++)
  {
    ptsSurfaceProbs[surf.indices[i]].push_back( PointSurfaceProb(i, idx, surf.probs[i]) );
  }
}

/**
 * ComputePointProbs
 */
void GreedySelection::ComputePointProbs(pcl::PointCloud<pcl::PointXYZRGB> &cloud, std::vector<int> &indices, 
      std::vector<float> &coeffs, std::vector<float> &probs)
{
  float err;

  probs.resize(indices.size());

  for (unsigned i=0; i<indices.size(); i++)
  {
    err = Plane::ImpPointDist(coeffs[0], coeffs[1], coeffs[2], coeffs[3], &cloud.points[indices[i]].x);
    probs[i] = exp(-(pow(err, 2) * invSqrSigma));
  }
}


/**
 * Init
 */
void GreedySelection::Init(std::vector<SurfaceModel::Ptr> &planes0)
{
  idMap.clear();
  idMap.resize(width*height,-1);
  mask.clear();
  mask.resize(width*height,0);
  mcnt=0;

  for (unsigned i=0; i<planes0.size(); i++)
  {
    SurfaceModel &plane = *planes0[i];

    plane.selected = false;

    for (unsigned j=0; j<plane.indices.size(); j++)
      idMap[plane.indices[j]] = i;
  }
}

/**
 * SelectPlane
 */
bool GreedySelection::SelectPlane(std::vector<SurfaceModel::Ptr> &planes0, SurfaceModel &plane)
{
  float savings0=0;
  unsigned idx;
  indices0.clear();
  probs.clear();
  overlap.clear();
  
  for (unsigned i=0; i<planes0.size(); i++) {
    planes0[i]->idx=0;
  }

  // select overlapping planes and indices
  for (unsigned i=0; i<plane.indices.size(); i++)
  {
    idx = plane.indices[i];
    if (idMap[idx]>=0 && !planes0[idMap[idx]]->selected)
    {
      planes0[idMap[idx]]->idx++;
    }
  }

  for (unsigned i=0; i<planes0.size(); i++)
    if (planes0[i]->idx > (int)planes0[i]->indices.size()/2)   //HACK!!
      overlap.push_back( planes0[i]);
  
  for (unsigned i=0; i<overlap.size(); i++)
  {
    SurfaceModel &p = *overlap[i];
    for (unsigned j=0; j<p.indices.size(); j++)
      indices0.push_back(p.indices[j]);

    ComputePointProbs(*cloud, p.indices, p.coeffs, probs);
    savings0 += ComputeSavings(probs);
  }

  //compare level 0 planes and level +1 plane
  ComputePointProbs(*cloud, indices0, plane.coeffs, probs);
  float savings1 = ComputeSavings(probs);

  if (savings1 > savings0 + feps) //level +1 is fine
  {
    // merge indices
    mcnt++;
    int idx;
    for (unsigned i=0; i<plane.indices.size(); i++)
      mask[plane.indices[i]] = mcnt;

    for (unsigned i=0; i<overlap.size(); i++)
    {
      SurfaceModel &p = *overlap[i];
      p.selected = true;
      for (unsigned j=0; j<p.indices.size(); j++)
      {
        idx = p.indices[j];
        if (mask[idx] != mcnt)
        {
          plane.indices.push_back(idx);
          mask[idx] = mcnt;
        }
      }
    }

    return true;
  }
  
  return false;
}

/************************** PUBLIC *************************/

/**
 * compute
 */
void GreedySelection::compute(std::vector<SurfaceModel::Ptr> &planes0, std::vector<SurfaceModel::Ptr> &planes1)
{
  if (cloud.get()==0 || width==0 || height==0) 
    throw std::runtime_error ("[GreedySelection::compute] Input point cloud not set!");

  if (planes1.size()>0)
  {
    tmpPlanes.clear();
    std::sort(planes1.begin(), planes1.end(), CmpNumPoints);

    Init(planes0);

    //test level-1
    for (unsigned i=0; i<planes1.size(); i++)
    {
      if (SelectPlane(planes0, *planes1[i]))
        tmpPlanes.push_back(planes1[i]);
    }

    // add remaining new planes level
    for (unsigned i=0; i<planes0.size(); i++)
      if (!planes0[i]->selected)
        tmpPlanes.push_back(planes0[i]);
    
    //return planes
    planes1 = tmpPlanes; 
  }
  else
  {
    planes1 = planes0;
  }
}

/**
 * setInputCloud
 */
void GreedySelection::setInputCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &_cloud)
{
  cloud = _cloud;
  width = cloud->width;
  height = cloud->height;
}

/**
 * setParameter
 */
void GreedySelection::setParameter(Parameter &p)
{
  param = p;
  invSqrSigma = 1./(pow(param.sigma, 2));
}



} //-- THE END --

