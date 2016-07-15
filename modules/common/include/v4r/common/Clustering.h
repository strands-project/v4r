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

#ifndef V4R_CLUSTERING_HH
#define V4R_CLUSTERING_HH

#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <v4r/core/macros.h>
#include <v4r/common/impl/DataMatrix2D.hpp>
#include <v4r/common/impl/SmartPtr.hpp>


namespace v4r 
{

/**
 * Cluster
 */
class V4R_EXPORTS Cluster
{
public:
  float sqr_sigma;
  Eigen::VectorXf data;
  std::vector<int> indices;

  Cluster() : sqr_sigma(0) {};
  Cluster(const Eigen::VectorXf &d) : sqr_sigma(0), data(d) {}
  Cluster(const Eigen::VectorXf &d, int idx) : sqr_sigma(0), data(d) {
    indices.push_back(idx);
  }

  typedef SmartPtr< ::v4r::Cluster> Ptr;
  typedef SmartPtr< ::v4r::Cluster const> ConstPtr;
};

/**
 * Clustering
 */
class V4R_EXPORTS Clustering
{
protected:
  std::vector< Cluster::Ptr > clusters;

public:
  Clustering() {}
  virtual ~Clustering() {}

  virtual void cluster(const DataMatrix2Df &) = 0;
  virtual void getClusters(std::vector<std::vector<int> > &) = 0;
  virtual void getCenters(DataMatrix2Df &) = 0;
};

}

#endif

