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
 * @file BoundaryDetector.h
 * @author Richtsfeld
 * @date November 2012
 * @version 0.1
 * @brief Estimate all boundary structures of surface models and views.
 */

#include "BoundaryDetectorSplit.h"

namespace surface
{


/************************************************************************************
 * Constructor/Destructor
 */

BoundaryDetectorSplit::BoundaryDetectorSplit() : BoundaryDetector()
{
  have_cloud = false;
  have_view = false;
  initialized = false;
  have_contours = false;
}

BoundaryDetector::~BoundaryDetector()
{
}

// ================================= Private functions ================================= //



// ================================= Public functions ================================= //

double timespec_diff3(struct timespec *x, struct timespec *y)                    /// TODO Remove later
{
  if(x->tv_nsec < y->tv_nsec)
  {
    int nsec = (y->tv_nsec - x->tv_nsec) / 1000000000 + 1;
    y->tv_nsec -= 1000000000 * nsec;
    y->tv_sec += nsec;
  }
  if(x->tv_nsec - y->tv_nsec > 1000000000)
  {
    int nsec = (y->tv_nsec - x->tv_nsec) / 1000000000;
    y->tv_nsec += 1000000000 * nsec;
    y->tv_sec -= nsec;
  }
  return (double)(x->tv_sec - y->tv_sec) +
    (double)(x->tv_nsec - y->tv_nsec)/1000000000.;
}

void BoundaryDetector::RecursiveClustering(int x, int y,
                                           int id_0, int id_1, 
                                           bool horizontal, aEdge &_ec)
{
  /// HORIZONTAL start
  if(horizontal) {

    /// horizontal x, y-1
    if(pre_edgels[GetIdx(x, y-1)].h_valid && pre_edgels[GetIdx(x, y-1)].h_ids[0] == id_0 && pre_edgels[GetIdx(x, y-1)].h_ids[1] == id_1) {
// if(bd_edges.size() == 231) printf("  h: next horizontal 1: %u-%u of surf: %u-%u\n", x, y-1, id_0, id_1);
      pre_edgels[GetIdx(x, y-1)].h_valid = false;
      bd_edgels.push_back(pre_edgels[GetIdx(x, y-1)]);
//       _ec.edgels_new.push_back(&bd_edgels[bd_edgels.size()-1]);
      _ec.edgels.push_back(bd_edgels.size()-1);
      if(pre_edgels[GetIdx(x, y-1)].corner_idx != -1 && _ec.corners[0] != pre_edgels[GetIdx(x, y-1)].corner_idx)                                     /// TODO TODO TODO corner_idx != -1 anstelle von is_corner!!!
// printf("%u-%u => This is now a corner edge => we have to stop here!\n", x, y-1);                                   /// Wenn corner erreicht wird, dann muss corner gelöscht werden (pre_corners???)
        _ec.corners[1] = pre_edgels[GetIdx(x, y-1)].corner_idx;
      else
        RecursiveClustering(x, y-1, id_0, id_1, true, _ec);
    }
    /// horizontal x, y+1
    else if(pre_edgels[GetIdx(x, y+1)].h_valid && pre_edgels[GetIdx(x, y+1)].h_ids[0] == id_0 && pre_edgels[GetIdx(x, y+1)].h_ids[1] == id_1) {
// if(bd_edges.size() == 231)  printf("  h: next horizontal 2: %u-%u of surf: %u-%u\n", x, y+1, id_0, id_1);
      pre_edgels[GetIdx(x, y+1)].h_valid = false;
      bd_edgels.push_back(pre_edgels[GetIdx(x, y+1)]);
//       _ec.edgels_new.push_back(&bd_edgels[bd_edgels.size()-1]);
      _ec.edgels.push_back(bd_edgels.size()-1);
      if(pre_edgels[GetIdx(x, y+1)].corner_idx != -1 && _ec.corners[0] != pre_edgels[GetIdx(x, y+1)].corner_idx)
// printf("%u-%u => This is now a corner edge => we have to stop here!\n", x, y+1);
        _ec.corners[1] = pre_edgels[GetIdx(x, y+1)].corner_idx;
      else
        RecursiveClustering(x, y+1, id_0, id_1, true, _ec);
    }
    
    /// vertical x, y-1
    else if(pre_edgels[GetIdx(x, y-1)].v_valid && pre_edgels[GetIdx(x, y-1)].v_ids[0] == id_1 && pre_edgels[GetIdx(x, y-1)].v_ids[1] == id_0) {
// if(bd_edges.size() == 231)  printf("  h: next vertical 1: %u-%u of surf: %u-%u\n", x, y-1, id_0, id_1);
      pre_edgels[GetIdx(x, y-1)].v_valid = false;
      bd_edgels.push_back(pre_edgels[GetIdx(x, y-1)]);
//       _ec.edgels_new.push_back(&bd_edgels[bd_edgels.size()-1]);
      _ec.edgels.push_back(bd_edgels.size()-1);
      if(pre_edgels[GetIdx(x, y-1)].corner_idx != -1 && _ec.corners[0] != pre_edgels[GetIdx(x, y-1)].corner_idx)
// printf("%u-%u => This is now a corner edge => we have to stop here!\n", x, y-1);
        _ec.corners[1] = pre_edgels[GetIdx(x, y-1)].corner_idx;
      else
        RecursiveClustering(x, y-1, id_1, id_0, false, _ec);    // changed ids
    }
    
    /// vertical x+1, y-1
    else if(pre_edgels[GetIdx(x+1, y-1)].v_valid && pre_edgels[GetIdx(x+1, y-1)].v_ids[0] == id_0 && pre_edgels[GetIdx(x+1, y-1)].v_ids[1] == id_1) {
// if(bd_edges.size() == 231)  printf("  h: next vertical 2: %u-%u of surf: %u-%u\n", x+1, y-1, id_0, id_1);
      pre_edgels[GetIdx(x+1, y-1)].v_valid = false;
      bd_edgels.push_back(pre_edgels[GetIdx(x+1, y-1)]);
//       _ec.edgels_new.push_back(&bd_edgels[bd_edgels.size()-1]);
      _ec.edgels.push_back(bd_edgels.size()-1);
      if(pre_edgels[GetIdx(x+1, y-1)].corner_idx != -1 && _ec.corners[0] != pre_edgels[GetIdx(x+1, y-1)].corner_idx)
// printf("%u-%u => This is now a corner edge => we have to stop here!\n", x+1, y-1);
//       else
        _ec.corners[1] = pre_edgels[GetIdx(x+1, y-1)].corner_idx;
      else
        RecursiveClustering(x+1, y-1, id_0, id_1, false, _ec);
    }
    
    /// vertical x, y
    else if(pre_edgels[GetIdx(x, y)].v_valid && pre_edgels[GetIdx(x, y)].v_ids[0] == id_0 && pre_edgels[GetIdx(x, y)].v_ids[1] == id_1) {
// if(bd_edges.size() == 231)  printf("  h: next vertical 3: %u-%u of surf: %u-%u\n", x, y, id_0, id_1);
      pre_edgels[GetIdx(x, y)].v_valid = false;
      bd_edgels.push_back(pre_edgels[GetIdx(x, y)]);
//       _ec.edgels_new.push_back(&bd_edgels[bd_edgels.size()-1]);
      _ec.edgels.push_back(bd_edgels.size()-1);
      if(pre_edgels[GetIdx(x, y)].corner_idx != -1 && _ec.corners[0] != pre_edgels[GetIdx(x, y)].corner_idx)
// printf("%u-%u => This is now a corner edge => we have to stop here!\n", x, y);
//       else
        _ec.corners[1] = pre_edgels[GetIdx(x, y)].corner_idx;
      else
        RecursiveClustering(x, y, id_0, id_1, false, _ec);
    }
    
    /// vertical x+1, y
    else if(pre_edgels[GetIdx(x+1, y)].v_valid && pre_edgels[GetIdx(x+1, y)].v_ids[0] == id_1 && pre_edgels[GetIdx(x+1, y)].v_ids[1] == id_0) {
// if(bd_edges.size() == 231)  printf("  h: next vertical 4: %u-%u of surf: %u-%u\n", x+1, y, id_0, id_1);
      pre_edgels[GetIdx(x+1, y)].v_valid = false;
      bd_edgels.push_back(pre_edgels[GetIdx(x+1, y)]);
//       _ec.edgels_new.push_back(&bd_edgels[bd_edgels.size()-1]);
      _ec.edgels.push_back(bd_edgels.size()-1);
      if(pre_edgels[GetIdx(x+1, y)].corner_idx != -1 && _ec.corners[0] != pre_edgels[GetIdx(x+1, y)].corner_idx)
// printf("%u-%u => This is now a corner edge => we have to stop here!\n", x+1, y);
//       else
        _ec.corners[1] = pre_edgels[GetIdx(x+1, y)].corner_idx;
      else
        RecursiveClustering(x+1, y, id_1, id_0, false, _ec);    // changed ids
    }
    else
      printf("  h: no more connection found => stop at %u-%u => size: %lu\n", x, y, _ec.edgels.size());                                         /// TODO Das darf nicht passieren? Kein end-corner gefunden!
  }
  
  
  /// VERTICAL
  else {
    
    /// vertical x-1, y
    if(pre_edgels[GetIdx(x-1, y)].v_valid && pre_edgels[GetIdx(x-1, y)].v_ids[0] == id_0 && pre_edgels[GetIdx(x-1, y)].v_ids[1] == id_1) {
// if(bd_edges.size() == 231)  printf("  v: next vertical 1: %u-%u of surf: %u-%u\n", x-1, y, id_0, id_1);
      pre_edgels[GetIdx(x-1, y)].v_valid = false;
      bd_edgels.push_back(pre_edgels[GetIdx(x-1, y)]);
//       _ec.edgels_new.push_back(&bd_edgels[bd_edgels.size()-1]);
      _ec.edgels.push_back(bd_edgels.size()-1);
      if(pre_edgels[GetIdx(x-1, y)].corner_idx != -1 && _ec.corners[0] != pre_edgels[GetIdx(x-1, y)].corner_idx) {
// printf("%u-%u => This is now a corner edge => we have to stop here!\n", x-1, y);
//       else
        _ec.corners[1] = pre_edgels[GetIdx(x-1, y)].corner_idx;
      }
      else
        RecursiveClustering(x-1, y, id_0, id_1, false, _ec);
    }
      
    /// vertical x+1, y
    else if(pre_edgels[GetIdx(x+1, y)].v_valid && pre_edgels[GetIdx(x+1, y)].v_ids[0] == id_0 && pre_edgels[GetIdx(x+1, y)].v_ids[1] == id_1) {
// if(bd_edges.size() == 231)  printf("  v: next vertical 2: %u-%u of surf: %u-%u\n", x+1, y, id_0, id_1);
      pre_edgels[GetIdx(x+1, y)].v_valid = false;
      bd_edgels.push_back(pre_edgels[GetIdx(x+1, y)]);
//       _ec.edgels_new.push_back(&bd_edgels[bd_edgels.size()-1]);
      _ec.edgels.push_back(bd_edgels.size()-1);
      if(pre_edgels[GetIdx(x+1, y)].corner_idx != -1 && _ec.corners[0] != pre_edgels[GetIdx(x+1, y)].corner_idx)
// printf("%u-%u => This is now a corner edge => we have to stop here!\n", x+1, y);
//       else
        _ec.corners[1] = pre_edgels[GetIdx(x+1, y)].corner_idx;
      else
        RecursiveClustering(x+1, y, id_0, id_1, false, _ec);
    }
    
    /// horizontal x-1, y
    else if(pre_edgels[GetIdx(x-1, y)].h_valid && pre_edgels[GetIdx(x-1, y)].h_ids[0] == id_1 && pre_edgels[GetIdx(x-1, y)].h_ids[1] == id_0) {
// if(bd_edges.size() == 231)  printf("  v: next horizontal 1: %u-%u of surf: %u-%u\n", x-1, y, id_0, id_1);
      pre_edgels[GetIdx(x-1, y)].h_valid = false;
      bd_edgels.push_back(pre_edgels[GetIdx(x-1, y)]);
//       _ec.edgels_new.push_back(&bd_edgels[bd_edgels.size()-1]);
      _ec.edgels.push_back(bd_edgels.size()-1);
      if(pre_edgels[GetIdx(x-1, y)].corner_idx != -1 && _ec.corners[0] != pre_edgels[GetIdx(x-1, y)].corner_idx)
// printf("%u-%u => This is now a corner edge => we have to stop here!\n", x-1, y);
//       else
        _ec.corners[1] = pre_edgels[GetIdx(x-1, y)].corner_idx;
      else
        RecursiveClustering(x-1, y, id_1, id_0, true, _ec);    // changed ids
    }
    
    /// horizontal x-1, y+1
    else if(pre_edgels[GetIdx(x-1, y+1)].h_valid && pre_edgels[GetIdx(x-1, y+1)].h_ids[0] == id_0 && pre_edgels[GetIdx(x-1, y+1)].h_ids[1] == id_1) {
// if(bd_edges.size() == 231)  printf("  v: next horizontal 2: %u-%u of surf: %u-%u\n", x-1, y+1, id_0, id_1);
      pre_edgels[GetIdx(x-1, y+1)].h_valid = false;
      bd_edgels.push_back(pre_edgels[GetIdx(x-1, y+1)]);
//       _ec.edgels_new.push_back(&bd_edgels[bd_edgels.size()-1]);
      _ec.edgels.push_back(bd_edgels.size()-1);
      if(pre_edgels[GetIdx(x-1, y+1)].corner_idx != -1 && _ec.corners[0] != pre_edgels[GetIdx(x-1, y+1)].corner_idx)
// printf("%u-%u => This is now a corner edge => we have to stop here!\n", x-1, y+1);
//       else
        _ec.corners[1] = pre_edgels[GetIdx(x-1, y+1)].corner_idx;
      else
        RecursiveClustering(x-1, y+1, id_0, id_1, true, _ec);
    }
    
    /// horizontal x, y
    else if(pre_edgels[GetIdx(x, y)].h_valid && pre_edgels[GetIdx(x, y)].h_ids[0] == id_0 && pre_edgels[GetIdx(x, y)].h_ids[1] == id_1) {
// if(bd_edges.size() == 231)  printf("  v: next horizontal 3: %u-%u of surf: %u-%u\n", x, y, id_0, id_1);
      pre_edgels[GetIdx(x, y)].h_valid = false;
      bd_edgels.push_back(pre_edgels[GetIdx(x, y)]);
//       _ec.edgels_new.push_back(&bd_edgels[bd_edgels.size()-1]);
      _ec.edgels.push_back(bd_edgels.size()-1);
      if(pre_edgels[GetIdx(x, y)].corner_idx != -1 && _ec.corners[0] != pre_edgels[GetIdx(x, y)].corner_idx) {
        
// if(bd_edges.size() == 231) 
//   if(pre_edgels[GetIdx(x, y)].corner_idx != -1)
//     printf("pre_edgels %u-%u is a corner: %u\n", x, y, pre_edgels[GetIdx(x, y)].corner_idx);
  
        _ec.corners[1] = pre_edgels[GetIdx(x, y)].corner_idx;
// if(bd_edges.size() == 231) printf("%u-%u => This is now a corner edge => we have to stop here: corner: %u-%u => %u\n", x, y, _ec.corners[0], pre_edgels[GetIdx(x, y)].corner_idx, _ec.corners[1]);
      }
      else {
        RecursiveClustering(x, y, id_0, id_1, true, _ec);
      }
    }
    
    /// horizontal x, y+1
    else if(pre_edgels[GetIdx(x, y+1)].h_valid && pre_edgels[GetIdx(x, y+1)].h_ids[0] == id_1 && pre_edgels[GetIdx(x, y+1)].h_ids[1] == id_0) {
if(bd_edges.size() == 231)  printf("  v: next horizontal 4: %u-%u of surf: %u-%u\n", x, y+1, id_0, id_1);
      pre_edgels[GetIdx(x, y+1)].h_valid = false;
      bd_edgels.push_back(pre_edgels[GetIdx(x, y+1)]);
//       _ec.edgels_new.push_back(&bd_edgels[bd_edgels.size()-1]);
      _ec.edgels.push_back(bd_edgels.size()-1);
      if(pre_edgels[GetIdx(x, y+1)].corner_idx != -1 && _ec.corners[0] != pre_edgels[GetIdx(x, y+1)].corner_idx)
// printf("%u-%u => This is now a corner edge => we have to stop here!\n", x, y+1);
//       else
        _ec.corners[1] = pre_edgels[GetIdx(x, y+1)].corner_idx;
      else
        RecursiveClustering(x, y+1, id_1, id_0, true, _ec);    // changed ids
    }
    
    else
      printf("  v: no more connection found => stop at %u-%u => size: %u\n", x, y, _ec.edgels.size());
//       /// TODO TODO TODO We have to add here a corner point?
  }
}


/**
 * @brief 
 */
void BoundaryDetector::ComputeEdges()
{
  for(unsigned i=0; i<pre_corners.size(); i++) {
    if(pre_corners[i].index != -1) {
            
      // check all four corner edges
      if(pre_corners[i].ids[0] != pre_corners[i].ids[1]) {
        unsigned id_0 = pre_corners[i].ids[0];
        unsigned id_1 = pre_corners[i].ids[1];
        int x = X(i);                                                           /// TODO Index und x,y vereinheitlichen => nicht so oft GetIdx
        int y = Y(i);
        if(pre_edgels[GetIdx(x, y)].h_valid) {                                  /// TODO Statt GetIdx => i
          surface::aEdge ec;
          ec.surfaces[0] = id_0;
          ec.surfaces[1] = id_1;
          ec.corners[0] = pre_corners[i].index;                                 /// TODO Hier wird pre_corners-index in edge geschrieben!!!
          ec.corners[1] = -1;
          bd_edgels.push_back(pre_edgels[GetIdx(x, y)]);
          pre_edgels[GetIdx(x, y)].h_valid = false;
          ec.edgels.push_back(bd_edgels.size()-1);
          RecursiveClustering(x, y, id_0, id_1, true, ec);
          
if(ec.corners[0] == -1 || ec.corners[1] == -1) {
//          bd_edges.push_back(ec);
  printf(" 1 => edge %lu => corners undefined: %u-%u\n", bd_edges.size(), ec.corners[0], ec.corners[1]);
}
else {
          bd_edges.push_back(ec);
//   printf("      => edge %u => corners are defined: %u-%u\n", bd_edges.size(), ec.corners[0], ec.corners[1]);
}
        }
      }
      
      if(pre_corners[i].ids[2] != pre_corners[i].ids[3]) {
        unsigned id_0 = pre_corners[i].ids[2];
        unsigned id_1 = pre_corners[i].ids[3];
        int x = X(i);
        int y = Y(i)+1;
        if(pre_edgels[GetIdx(x, y)].h_valid) {
          surface::aEdge ec;
          ec.surfaces[0] = id_1;        // TODO changed
          ec.surfaces[1] = id_0;
          ec.corners[0] = pre_corners[i].index;
          ec.corners[1] = -1;
          bd_edgels.push_back(pre_edgels[GetIdx(x, y)]);
          pre_edgels[GetIdx(x, y)].h_valid = false;
          ec.edgels.push_back(bd_edgels.size()-1);
          RecursiveClustering(x, y, id_0, id_1, true, ec);
if(ec.corners[0] == -1 || ec.corners[1] == -1) {
//          bd_edges.push_back(ec);
  printf(" 2 => edge %u => corners undefined: %u-%u between %u-%u\n", bd_edges.size(), ec.corners[0], ec.corners[1], ec.surfaces[0], ec.surfaces[1]);
}
else {
          bd_edges.push_back(ec);
//   printf("      => edge %u => corners are defined: %u-%u\n", bd_edges.size(), ec.corners[0], ec.corners[1]);
}
        }
      }
      
      if(pre_corners[i].ids[0] != pre_corners[i].ids[2]) {
        unsigned id_0 = pre_corners[i].ids[0];
        unsigned id_1 = pre_corners[i].ids[2];
        int x = X(i);
        int y = Y(i);
        if(pre_edgels[GetIdx(x, y)].v_valid) {
          surface::aEdge ec;
          ec.surfaces[0] = id_1;        // TODO changed
          ec.surfaces[1] = id_0;
          ec.corners[0] = pre_corners[i].index;
          ec.corners[1] = -1;
          bd_edgels.push_back(pre_edgels[GetIdx(x, y)]);
          pre_edgels[GetIdx(x, y)].v_valid = false;
          ec.edgels.push_back(bd_edgels.size()-1);
          RecursiveClustering(x, y, id_0, id_1, false, ec);
if(ec.corners[0] == -1 || ec.corners[1] == -1) {
//          bd_edges.push_back(ec);
  printf(" 3 => edge %u => corners undefined: %u-%u\n", bd_edges.size(), ec.corners[0], ec.corners[1]);
}
else {
          bd_edges.push_back(ec);
//   printf("      => edge %u => corners are defined: %u-%u\n", bd_edges.size(), ec.corners[0], ec.corners[1]);
}
        }
      }
      
      if(pre_corners[i].ids[1] != pre_corners[i].ids[3]) {
        unsigned id_0 = pre_corners[i].ids[1];
        unsigned id_1 = pre_corners[i].ids[3];
        int x = X(i)+1;
        int y = Y(i);
// if(x == 320 && y == 328)
//   printf("ACHTUNG, der wird schon probiert!\n");
        if(pre_edgels[GetIdx(x, y)].v_valid) {
          surface::aEdge ec;
          ec.surfaces[0] = id_0;        // TODO changed
          ec.surfaces[1] = id_1;
          ec.corners[0] = pre_corners[i].index;
          ec.corners[1] = -1;
          bd_edgels.push_back(pre_edgels[GetIdx(x, y)]);
          pre_edgels[GetIdx(x, y)].v_valid = false;
          ec.edgels.push_back(bd_edgels.size()-1);
          RecursiveClustering(x, y, id_0, id_1, false, ec);
if(ec.corners[0] == -1 || ec.corners[1] == -1) {
//          bd_edges.push_back(ec);
  printf(" 4 => edge %u => corners undefined: %u-%u\n", bd_edges.size(), ec.corners[0], ec.corners[1]);
}
else {
         bd_edges.push_back(ec);
//   printf("      => edge %u => corners are defined: %u-%u\n", bd_edges.size(), ec.corners[0], ec.corners[1]);
}
        }
      }
    }
  }
  
  /// TODO Are there still unused edges => Edges without corner point.
  /// While edgels are still there:
  ///   create new edge with same corner point.
  ///
}


void BoundaryDetector::CopyEdges()
{
  for(unsigned i=0; i<pre_corners.size(); i++) {
    if(pre_corners[i].index != -1) {
      surface::Corner corner;
      corner.x = ((float) X(pre_corners[i].index)) + 0.5;
      corner.y = ((float) Y(pre_corners[i].index)) + 0.5;
      pre_corners[i].index = view->corners.size();                                      /// TODO Change pre-corners->index to index in list !!!!
      view->corners.push_back(corner);      
    }
  }
  
  // copy edges and edgels
  for(unsigned i=0; i<bd_edges.size(); i++) {
    surface::Edge edge;
    edge.corner[0] = pre_corners[bd_edges[i].corners[0]].index;                         /// TODO Hier werden nun die oben geänderten pre_corners indices eingetragen!
    edge.corner[1] = pre_corners[bd_edges[i].corners[1]].index;
    edge.surface[0] = bd_edges[i].surfaces[0];
    edge.surface[1] = bd_edges[i].surfaces[1];
    
    // copy the edgels
    for(unsigned j=0; j< bd_edges[i].edgels.size(); j++) {
      if(bd_edgels[bd_edges[i].edgels[j]].horizontal) { // horizontal edge
        surface::Edgel e;
        e.x = ((float) X(bd_edgels[bd_edges[i].edgels[j]].index)) + 0.5;
        e.y = (float) Y(bd_edgels[bd_edges[i].edgels[j]].index);
        view->edgels.push_back(e);
        edge.edgels.push_back(view->edgels.size()-1);
      }
      if(bd_edgels[bd_edges[i].edgels[j]].vertical) { // vertical edge
        surface::Edgel e;
        e.x = (float) X(bd_edgels[bd_edges[i].edgels[j]].index);
        e.y = ((float) Y(bd_edgels[bd_edges[i].edgels[j]].index)) + 0.5;
        view->edgels.push_back(e);
        edge.edgels.push_back(view->edgels.size()-1);
      }
    }
    view->edges.push_back(edge);
  }
  
  view->aedgels = bd_edgels;    /// TODO Remove later: Just for debugging
  view->aedges = bd_edges;      /// TODO Remove later: Just for debugging
}


void BoundaryDetector::computeBoundaryNetwork()
{     
  if(!initialized)
    initialize();
    
  bd_edgels.clear();
  bd_edges.clear();

  ComputeEdges();               //< Compute edges
  
static struct timespec start1, current1;
clock_gettime(CLOCK_THREAD_CPUTIME_ID, &start1);

  CopyEdges();                  //< Copy edges with corners and edgels to view
  
clock_gettime(CLOCK_THREAD_CPUTIME_ID, &current1);
printf("[BoundaryDetector::computeBoundaryNetwork] Copy edges: %4.3f\n", timespec_diff3(&current1, &start1));
start1 = current1;

}

// void BoundaryDetector::computeConcavityPoints()
// {
//   unsigned OFFSET = 10;
//   unsigned OFFSET2 = 2*OFFSET;
//   unsigned MIN_PTS = 100;
//   double MIN_ANGLE = -0.8;      // cos⁻¹(−0.9) == 155°  // cos⁻¹(−0.9) == 143°
// 
//   if(!have_contours)
//     computeContours();
//   
// static struct timespec start1, current1;
// clock_gettime(CLOCK_THREAD_CPUTIME_ID, &start1);
// 
//   for(unsigned i=0; i<view->surfaces.size(); i++) {                             /// TODO 14ms => Parallelize?
//     if(view->surfaces[i]->contours.size() > 0) {
//       unsigned cnt_size = view->surfaces[i]->contours[0].size();
//       if(cnt_size >= MIN_PTS) {
// //         OFFSET = cnt_size/30;                                                /// TODO ist das eine gute Idee?
// //         OFFSET2 = 2*OFFSET;
//         unsigned idx_s = 0;
//         unsigned idx_m = 0;
//         unsigned idx_e = 0;
//         double biggest_concave = -1.0;
//         unsigned biggest_concave_index = 0;
//         double biggest_convex = -1.0;
//         unsigned biggest_convex_index = 0;
//         for(idx_s=0; idx_s<cnt_size; idx_s++) {
//           idx_s = idx_s; 
//           idx_m = idx_s + OFFSET;
//           if(idx_m >= cnt_size) idx_m -= cnt_size;
//           idx_e = idx_s + OFFSET2;
//           if(idx_e >= cnt_size) idx_e -= cnt_size;
//           
//           Eigen::Vector2d vec_s_m, vec_m_e;
//           vec_s_m[0] = (double) (X(view->surfaces[i]->contours[0][idx_s]) - X(view->surfaces[i]->contours[0][idx_m]));
//           vec_s_m[1] = (double) (Y(view->surfaces[i]->contours[0][idx_s]) - Y(view->surfaces[i]->contours[0][idx_m]));
//           vec_m_e[0] = (double) (X(view->surfaces[i]->contours[0][idx_e]) - X(view->surfaces[i]->contours[0][idx_m]));
//           vec_m_e[1] = (double) (Y(view->surfaces[i]->contours[0][idx_e]) - Y(view->surfaces[i]->contours[0][idx_m]));
// 
//           vec_s_m.normalize();
//           vec_m_e.normalize();
// //   printf("Vectors: %4.3f / %4.3f --  %4.3f / %4.3f\n", vec_s_m[0], vec_s_m[1], vec_m_e[0], vec_m_e[1]);
// 
//           //y2*x1-y1*x2
//           bool concave = false;
//           bool convex = false;
//           double cp = vec_m_e[1]*vec_s_m[0] - vec_s_m[1]*vec_m_e[0];
//           if(cp > 0.0)
//             concave = true;
//           else 
//             convex = true;
// 
//           double angle = vec_s_m.dot(vec_m_e);              // -1 ... 1 == 180-0°   // -0.9 == 155
// //           double angle_deg = acos(angle)*180./M_PI;       /// TODO Braucht man später nicht
//           
//           if(concave) {
//             if(angle > biggest_concave && angle > MIN_ANGLE) {
//               biggest_concave = angle;
//               biggest_concave_index = idx_m;
//             }
//           }
//           else {        // change from concave to convex => Save the biggest concave pt.
//             if(biggest_concave > MIN_ANGLE) {
// // printf("  ==> save concave %u: %u-%u-%u => pt: %2.2f-%2.2f / %2.2f-%2.2f => angle %4.3f => %4.3f (biggest: %4.3f @ %u)\n", i, idx_s, idx_m, idx_e, vec_s_m[0], vec_s_m[1], vec_m_e[0], vec_m_e[1], angle, angle_deg, biggest_concave, biggest_concave_index);
//               view->surfaces[i]->concave.push_back(view->surfaces[i]->contours[0][biggest_concave_index]);
//             }
//             biggest_concave = -1.0;
//             biggest_concave_index = 0;
//           }
//         }
//       }
//     }
//   }
//   
// clock_gettime(CLOCK_THREAD_CPUTIME_ID, &current1);
// printf("[BoundaryDetector::computeConcavityPoints] Runtime: %4.3f\n", timespec_diff3(&current1, &start1));  
// }



// void BoundaryDetector::computeSplitting()
// {
//   if(!initialized)
//     initialize();
//   
//   cv::Mat_<cv::Vec3b> matImage;         // color image
//   cv::Mat gray_image;                   // gray image
//   cv::Mat sobel_x, abs_sobel_x;         // sobel image x
//   cv::Mat sobel_y, abs_sobel_y;         // sobel image y
//   cv::Mat_<char> sobel, sobel_max;                 // weighted sobel image
//   
//   int scale = 1;
//   int delta = 0;
//   int ddepth = CV_16S;
// 
// static struct timespec start1, current1;
// clock_gettime(CLOCK_THREAD_CPUTIME_ID, &start1);
// 
//   // pcl_cloud to image
//   pclA::ConvertPCLCloud2Image(pcl_cloud, matImage);  
//   cv::cvtColor(matImage, gray_image, CV_BGR2GRAY );
// 
// clock_gettime(CLOCK_THREAD_CPUTIME_ID, &current1);
// printf("[BoundaryDetector::computeSplitting] Convert image to gray: %4.3f\n", timespec_diff3(&current1, &start1));
// start1 = current1;
// 
//   // calculate sobel images
//   cv::Sobel(gray_image, sobel_x, ddepth, 1, 0, 1, scale, delta, cv::BORDER_DEFAULT );
//   convertScaleAbs(sobel_x, abs_sobel_x);
//   cv::Sobel(gray_image, sobel_y, ddepth, 0, 1, 1, scale, delta, cv::BORDER_DEFAULT );
//   convertScaleAbs(sobel_y, abs_sobel_y);
//   cv::addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0, sobel_max);
//   sobel = cv::max(abs_sobel_x, abs_sobel_y);
// 
//   /// we do not want to go along surface boundaries, set penalty
//   for(int row=0; row<contours2.rows-1; row++) {
//     for(int col=0; col<contours2.cols-1; col++) {
//       if(contours2.at<int>(row, col) != -1)
//         sobel.at<char>(row, col) = 0;
//     }
//   }
//   
// clock_gettime(CLOCK_THREAD_CPUTIME_ID, &current1);
// printf("[BoundaryDetector::computeSplitting] Calculate sobel image: %4.3f\n", timespec_diff3(&current1, &start1));
// start1 = current1;
// 
// // printf("Show sobel image\n");
//   cv::imshow("sobel_max", sobel);
//   cv::imshow("sobel", sobel_max);
// //   cv::imshow("sobel_x", abs_sobel_x);
// //   cv::imshow("sobel_y", abs_sobel_y);
//   cv::waitKey(50);
// // printf("Show sobel image done\n");
//   
//   // get each surface with >= 2 concavity points
// #pragma omp parallel for
//   for(unsigned i=0; i<view->surfaces.size(); i++) {                             /// TODO Das könnte man parallelisieren, oder?
// 
//     if(view->surfaces[i]->concave.size() > 1) {
// // printf("[BoundaryDetector::computeSplitting] Surface %u with %u concave points\n", i, view->surfaces[i]->concave.size());
// if(view->surfaces[i]->concave.size() < 15)                                      /// TODO Only surfaces with not to much points!!!
// 
//       for(unsigned j=0; j<view->surfaces[i]->concave.size()-1; j++) {                           /// Eigentlich müsste man hier parallelisieren
//         for(unsigned k=j+1; k<view->surfaces[i]->concave.size(); k++) { 
//           
//           bool end_not_reached = true;
//           std::map<float, path_point> sps;    // map for shortest path search
//           cv::Mat_<int> black_list = cv::Mat_<int>(pcl_cloud->height, pcl_cloud->width);     // already visited patches
//           black_list.setTo(1);
// 
//           int source_idx = view->surfaces[i]->concave[j];                       // index of source
//           int sink_idx = view->surfaces[i]->concave[k];                         // index of sink
//           int source_x = X(source_idx);
//           int source_y = Y(source_idx);
//           int sink_x = X(sink_idx);
//           int sink_y = Y(sink_idx);
//           int patch_number = patches.at<int>(source_y, source_x);         // number of the patch in the patches
//           int min_dist = abs(sink_x - source_x) + abs(sink_y - source_y);
// 
// // if(i != 17) continue;
// printf("We start with patch %u @ %u-%u => %u-%u (min_dist: %u)\n", patch_number, source_x, source_y, sink_x, sink_y, min_dist);
// 
// if(patches.at<int>(source_y, source_x) != patches.at<int>(sink_y, sink_x))
//   printf("Problem => Source and sink not on the same patch!\n");
// 
//          
//           // add source to map
//           float costs = 0.0f;
//           path_point p;
//           p.px.clear();
//           p.px.push_back(source_x);
//           p.py.clear();
//           p.py.push_back(source_y);
//           p.path.clear();
//           p.path.push_back(source_idx);
//           p.distance2sink = min_dist;
//           p.costs = costs;
//           sps.insert(std::make_pair<float, path_point>(costs, p));
//           black_list.at<int>(source_y, source_x) = 0;
//           
//           while(end_not_reached) {
//             // get the first point in the map and add all neighbors:
//             if(sps.size() == 0) {
//               printf(" ###################### sps is empty!\n");
//               break;
//             }
//               
//             
//             float act_costs = (sps.begin())->second.costs;      // real costs from sps
//             path_point best_point = (sps.begin())->second;
// // printf("all_cost @ %u-%u: %4.3f and path.size: %lu - dist to sink (of %u): %lu\n", best_point.px[best_point.px.size()-1],  
// //                                                                            best_point.py[best_point.py.size()-1], 
// //                                                                            best_point.all_costs, 
// //                                                                            best_point.path.size(),
// //                                                                            min_dist,
// //                                                                            best_point.distance2sink);
//             sps.erase(sps.begin());                        /// TODO We have to delete that entry!!!!
// 
//             // bool new_found = false;
//             
//             /// check top
//             int next_x = best_point.px[best_point.px.size()-1];
//             int next_y = best_point.py[best_point.py.size()-1]-1;
//             if(black_list.at<int>(next_y, next_x) != 0)
//               if(next_y >= 0 && patches.at<int>(next_y, next_x) == patch_number) {
// //                 new_found = true;
//                 path_point npt;
//                 for(unsigned l=0; l<best_point.px.size(); l++) { // copy px, py
//                   npt.px.push_back(best_point.px[l]);
//                   npt.py.push_back(best_point.py[l]);
//                   npt.path.push_back(best_point.path[l]);
//                 }
//                 npt.px.push_back(next_x);
//                 npt.py.push_back(next_y); // to top
//                 npt.path.push_back(GetIdx(next_x, next_y));
//                 npt.distance2sink = abs(sink_x - next_x) + abs(sink_y - next_y);
//                 black_list.at<int>(next_y, next_x) = 0;
//                 npt.costs = act_costs + (255.- (float) sobel.at<char>(next_y, next_x));
//                 npt.all_costs = npt.costs * ((float) (best_point.path.size() + npt.distance2sink)) / ((float) (min_dist));
//                 if(npt.distance2sink == 0) {
//                   sps.clear();
//                   end_not_reached = false;
//                 }
//                 sps.insert(std::make_pair<float, path_point>(npt.all_costs, npt));
//               }
//             
//             /// check right
//             next_x = best_point.px[best_point.px.size()-1]+1;
//             next_y = best_point.py[best_point.py.size()-1];
//             if(black_list.at<int>(next_y, next_x) != 0)
//               if(next_x < pcl_cloud->width && patches.at<int>(next_y, next_x) == patch_number) {
// //                 new_found = true;
//                 path_point npt;
//                 for(unsigned l=0; l<best_point.px.size(); l++) { // copy px, py
//                   npt.px.push_back(best_point.px[l]);
//                   npt.py.push_back(best_point.py[l]);
//                   npt.path.push_back(best_point.path[l]);
//                 }
//                 npt.px.push_back(next_x);
//                 npt.py.push_back(next_y); // to top
//                 npt.path.push_back(GetIdx(next_x, next_y));
//                 npt.distance2sink = abs(sink_x - next_x) + abs(sink_y - next_y);
//                 black_list.at<int>(next_y, next_x) = 0;
//                 npt.costs = act_costs + (255.- (float) sobel.at<char>(next_y, next_x));
//                 npt.all_costs = npt.costs * ((float) (best_point.path.size() + npt.distance2sink)) / ((float) (min_dist));
//                 if(npt.distance2sink == 0) {
//                   sps.clear();
//                   end_not_reached = false;
//                 }
//                 sps.insert(std::make_pair<float, path_point>(npt.all_costs, npt));
//               }
//             
//             /// check left
//             next_x = best_point.px[best_point.px.size()-1]-1;
//             next_y = best_point.py[best_point.py.size()-1];
//             if(black_list.at<int>(next_y, next_x) != 0)
//               if(next_x >= 0 && patches.at<int>(next_y, next_x) == patch_number) {
// //                 new_found = true;
//                 path_point npt;
//                 for(unsigned l=0; l<best_point.px.size(); l++) { // copy px, py
//                   npt.px.push_back(best_point.px[l]);
//                   npt.py.push_back(best_point.py[l]);
//                   npt.path.push_back(best_point.path[l]);
//                 }
//                 npt.px.push_back(next_x);
//                 npt.py.push_back(next_y); // to top
//                 npt.path.push_back(GetIdx(next_x, next_y));
//                 npt.distance2sink = abs(sink_x - next_x) + abs(sink_y - next_y);
//                 black_list.at<int>(next_y, next_x) = 0;
//                 npt.costs = act_costs + (255.- (float) sobel.at<char>(next_y, next_x));
//                 npt.all_costs = npt.costs * ((float) (best_point.path.size() + npt.distance2sink)) / ((float) (min_dist));
//                 if(npt.distance2sink == 0) {
//                   sps.clear();
//                   end_not_reached = false;
//                 }
//                 sps.insert(std::make_pair<float, path_point>(npt.all_costs, npt));
//               }            
//             
//             /// check bottom
//             next_x = best_point.px[best_point.px.size()-1];
//             next_y = best_point.py[best_point.py.size()-1]+1;
//             if(black_list.at<int>(next_y, next_x) != 0)
//               if(next_y < pcl_cloud->height && patches.at<int>(next_y, next_x) == patch_number) {
// //                 new_found = true;
//                 path_point npt;
//                 for(unsigned l=0; l<best_point.px.size(); l++) { // copy px, py
//                   npt.px.push_back(best_point.px[l]);
//                   npt.py.push_back(best_point.py[l]);
//                   npt.path.push_back(best_point.path[l]);
//                 }
//                 npt.px.push_back(next_x);
//                 npt.py.push_back(next_y); // to top
//                 npt.path.push_back(GetIdx(next_x, next_y));
//                 npt.distance2sink = abs(sink_x - next_x) + abs(sink_y - next_y);
//                 black_list.at<int>(next_y, next_x) = 0;
//                 npt.costs = act_costs + (255.- (float) sobel.at<char>(next_y, next_x));
//                 npt.all_costs = npt.costs * ((float) (best_point.path.size() + npt.distance2sink)) / ((float) (min_dist));
//                 if(npt.distance2sink == 0) {
//                   sps.clear();
//                   end_not_reached = false;
//                 }
//                 sps.insert(std::make_pair<float, path_point>(npt.all_costs, npt));
//               } 
//             
//             // add new neighboring pathes, if possible
// // if(!new_found) printf("  => end here!\n");
//           }
//           
//           if(sps.size() != 0) {
//             int costs = (int) (sps.begin()->second.costs/sps.begin()->second.path.size());
//             if(view->surfaces[i]->split_pathes.size() == 0) {
//               view->surfaces[i]->split_pathes.push_back(sps.begin()->second.path);
//               view->surfaces[i]->costs.push_back(costs);
//               printf("  [BoundaryDetector::computeSplitting] Found path for splitting surf %u: costs: %4.3f (%lu)\n", i, sps.begin()->second.all_costs, (int) sps.begin()->second.costs/sps.begin()->second.path.size());
//             }
//             else {
//               if(view->surfaces[i]->costs[0] > costs) {
//                 view->surfaces[i]->split_pathes.clear();
//                 view->surfaces[i]->costs.clear();
//                 view->surfaces[i]->split_pathes.push_back(sps.begin()->second.path);
//                 view->surfaces[i]->costs.push_back(costs);
//                 printf("  [BoundaryDetector::computeSplitting] Found new path for splitting surf %u: costs: %4.3f (%lu)\n", i, sps.begin()->second.all_costs, (int) sps.begin()->second.costs/sps.begin()->second.path.size());
//               }
//             }
//           }
//           else
//             printf("[BoundaryDetector::computeSplitting] Warning: No end found!\n");
//           
//           // What happens now if we have found a way from source to sink?
//             // save best path in surface?
//             // split surface immediately?
//             // b-spline fitting?
//           
//         }
//       }
//     }
//   }
//   
//   /// print the best splits:
//   for(unsigned i=0; i<view->surfaces.size(); i++)
//     if(view->surfaces[i]->costs.size() > 0) {
//       printf("Surface %u: split costs: %u\n", i, view->surfaces[i]->costs[0]);
//     }
//   
// clock_gettime(CLOCK_THREAD_CPUTIME_ID, &current1);
// printf("[BoundaryDetector::computeSplitting] Shortest path search: %4.3f\n", timespec_diff3(&current1, &start1));
// start1 = current1;
// }


// void BoundaryDetector::updateContour(const std::vector<int> &_indices, 
//                                      std::vector< std::vector<int> > &_contours)
// {
//   size_t min_contour_size = 5;
//   _contours.clear();
// 
//   patches = cv::Mat_<int>(pcl_cloud->height, pcl_cloud->width);
//   patches.setTo(-1);
//   contours = cv::Mat_<int>(pcl_cloud->height, pcl_cloud->width);
//   contours.setTo(-1);
//   
//   for(unsigned j=0; j<_indices.size(); j++)
//     patches.at<int>(Y(_indices[j]), X(_indices[j])) = 0;
// 
//   #pragma omp parallel for
//   for(int row=0; row<patches.rows-1; row++) {
//     for(int col=0; col<patches.cols-1; col++) {
//       // create contour image with 4-neighborhood
//       if((row == 0 || row == (int) pcl_cloud->height) && patches.at<int>(row, col) != -1)
//         contours.at<int>(row,col) = patches.at<int>(row,col);
//       if((col == 0  || row == (int) pcl_cloud->width) && patches.at<int>(row, col) != -1)
//         contours.at<int>(row,col) = patches.at<int>(row,col);
//       
//       if(patches.at<int>(row, col) != patches.at<int>(row, col+1)) {    // horizontal edge
//         contours.at<int>(row, col) = patches.at<int>(row, col);
//         contours.at<int>(row, col+1) = patches.at<int>(row, col+1);
//       }
//       if(patches.at<int>(row, col) != patches.at<int>(row+1, col)) {    // vertical edge
//         contours.at<int>(row, col) = patches.at<int>(row,col);
//         contours.at<int>(row+1, col) = patches.at<int>(row+1, col);
//       } 
//     }
//   }
//   
//   // start at top left and go through contours
//   for(int row=0; row<contours.rows; row++) {
//     for(int col=0; col<contours.cols; col++) {
//       if(contours.at<int>(row,col) != -1) {
//         int id = contours.at<int>(row,col);
//         std::vector<int> contour;
//         
//         contour.push_back(GetIdx(col, row));
//         bool end = false;
//         RecursiveContourClustering(id, col, row, col, row, 4, contour, end);
//         
//         if(!end && contour.size() > 6)
//           printf("[BoundaryDetector::updateContour] Error: This contour has NO end: %u => size: %u\n", id, contour.size());
//         
//         if(contour.size() > min_contour_size)
//           _contours.push_back(contour);
//         else
//           printf("[BoundaryDetector::updateContour] Warning: Contour %u with too less points: %u\n", id, contour.size());
//         
//         // delete contour points from contours map!
//         for(unsigned i=0; i<contour.size(); i++)
//           contours.at<int>(Y(contour[i]), X(contour[i])) = -1; 
//       }
//     }
//   }
// }

} // end surface












