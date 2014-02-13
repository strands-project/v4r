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

/**
 * @file BoundaryDetectorSplit.h
 * @author Richtsfeld
 * @date January 2013
 * @version 0.2
 * @brief Estimate all boundary structures of surface models and views.
 */

#ifndef SURFACE_BOUNDARY_DETECTOR_H
#define SURFACE_BOUNDARY_DETECTOR_H

#include <vector>
#include <stdio.h>
#include <stdlib.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

#include "v4r/SurfaceUtils/SurfaceModel.hpp"

#include "BoundaryDetector.h"

namespace surface
{


struct path_point {
  std::vector<int> px;          // already gone way, starting with sink
  std::vector<int> py;          // already gone way, starting with sink
  std::vector<int> path;        // indices of splitting path
  int distance2sink;            // minimum distance to sink
  float costs;                  // already summed up sobel costs
  float all_costs;              // costs for sobel*distance
//   float expected_costs;         // TODO Durchschnittskosten werden auf distance2sink aufgerechnet
};
  
  
class BoundaryDetectorSplit : public BoundaryDetector
{
public:
  
protected:

private:

  std::vector<aEdgel> bd_edgels;                         ///< Edgels
  std::vector<aEdge> bd_edges;                           ///< Edge between corners

  /** compute edges and corners **/
  void RecursiveClustering(int x, int y, 
                           int id_0, int id_1, 
                           bool horizontal, aEdge &_ec);

  /** compute edges **/
  void ComputeEdges();
  
  /** Copy structures to view **/
  void CopyEdges();
  
public:
  BoundaryDetector();
  ~BoundaryDetector();
  
  /** Compute boundaries between surfaces and create boundary network **/
  void computeBoundaryNetwork();

  /** Update contour based on a surface indices list **/
//   void updateContour(const std::vector<int> &_indices, 
//                      std::vector< std::vector<int> > &_contours);
};


/*************************** INLINE METHODES **************************/


} //--END--

#endif

