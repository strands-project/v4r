/**
 * $Id$
 * 
 * Software License Agreement (GNU General Public License)
 *
 *  Copyright (C) 2015:
 *
 *    Johann Prankl, prankl@acin.tuwien.ac.at
 *    Aitor Aldoma, aldoma@acin.tuwien.ac.at
 *
 *      Automation and Control Institute
 *      Vienna University of Technology
 *      Gusshausstraße 25-29
 *      1170 Vienn, Austria
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
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * @author Johann Prankl, Aitor Aldoma
 *
 */

#ifndef KP_CLUSTERING_RNN_HH
#define KP_CLUSTERING_RNN_HH

#include <vector>
#include <string>
#include <stdexcept>
#include <float.h>
#include <v4r/common/impl/DataMatrix2D.hpp>
#include "Clustering.h"




namespace v4r
{

/**
 * ClusteringRNN
 */
class V4R_EXPORTS ClusteringRNN : public Clustering
{
public:
  class Parameter
  {
  public:
    float dist_thr;

    Parameter(float _dist_thr=0.4)
     : dist_thr(_dist_thr) {}
  };

private:
  std::vector< Cluster::Ptr > clusters;

  int getNearestNeighbour(const Cluster &cluster, const std::vector<Cluster::Ptr> &clusters, float &sim);
  void agglomerate(const Cluster &src, Cluster &dst);

  void initDataStructure(const DataMatrix2Df &samples, std::vector< Cluster::Ptr > &data);


 
public:
  Parameter param;
  bool dbg;

  ClusteringRNN(const Parameter &_param = Parameter(), bool _dbg=true);
  ~ClusteringRNN();

  virtual void cluster(const DataMatrix2Df &samples); 
  virtual void getClusters(std::vector<std::vector<int> > &_clusters);
  virtual void getCenters(DataMatrix2Df &_centers);
};





/************************** INLINE METHODES ******************************/



}

#endif

