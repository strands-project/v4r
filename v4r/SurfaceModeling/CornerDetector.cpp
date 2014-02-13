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
 * @file CornerDetector.cpp
 * @author Richtsfeld
 * @date October 2012
 * @version 0.1
 * @brief Detect the corners in the segmented image.
 */

#include "CornerDetector.h"

namespace surface
{


/************************************************************************************
 * Constructor/Destructor
 */

CornerDetector::CornerDetector()
{
  have_cloud = false;
  have_view = false;
}

CornerDetector::~CornerDetector()
{
}

// ================================= Private functions ================================= //



// ================================= Public functions ================================= //



void CornerDetector::setInputCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr & _pcl_cloud)
{
  pcl_cloud = _pcl_cloud;
  have_cloud = true;
  
  corners.resize(pcl_cloud->width * pcl_cloud->height);
  edgels.resize(pcl_cloud->width * pcl_cloud->height);
}


void CornerDetector::setView(surface::View *_view)
{
  view = _view;
  have_view = true;
}


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



void CornerDetector::compute()
{     
  if(!have_cloud) {
    printf("[CornerDetector::compute] Error: No input cloud set.\n");
    exit(0);
  }

  if(!have_view) {
    printf("[CornerDetector::compute] Error: No view set.\n");
    exit(0);
  }

static struct timespec start1, current1;
clock_gettime(CLOCK_THREAD_CPUTIME_ID, &start1);

  view->corners.clear();
  view->edgels.clear();
  view->edges.clear();

  cv::Mat_<int> patches;
  patches = cv::Mat_<int>(pcl_cloud->height, pcl_cloud->width);
  patches.setTo(-1);
  for(unsigned i=0; i<view->surfaces.size(); i++) {
    for(unsigned j=0; j<view->surfaces[i]->indices.size(); j++) {
      int row = view->surfaces[i]->indices[j] / pcl_cloud->width;
      int col = view->surfaces[i]->indices[j] % pcl_cloud->width;
      patches.at<int>(row, col) = i;
    }
  }
    
  for(int row=0; row<patches.rows-1; row++) {
    for(int col=0; col<patches.cols-1; col++) {
      
      // initialize corner
      surface::Corner c;
      c.index = -1;
      corners[GetIdx(col, row)] = c;

      // initialize edge
      surface::Edgel e;
      e.index = -1;
      e.is_corner = false;
      e.horizontal = false;
      e.h_valid = false;
      e.vertical = false;
      e.v_valid = false;
      edgels[GetIdx(col, row)] = e;
      
      unsigned found = 0;
      if(patches.at<int>(row, col) != patches.at<int>(row, col+1))
        found++;
      if(patches.at<int>(row, col+1) != patches.at<int>(row+1, col+1))
        found++;
      if(patches.at<int>(row+1, col+1) != patches.at<int>(row+1, col))
        found++;
      if(patches.at<int>(row+1, col) != patches.at<int>(row, col))
        found++;
      
      if(found >= 3) {
        surface::Corner c;
        c.index = row*pcl_cloud->width + col;
        c.ids[0] = patches.at<int>(row, col);
        c.ids[1] = patches.at<int>(row, col+1);
        c.ids[2] = patches.at<int>(row+1, col);
        c.ids[3] = patches.at<int>(row+1, col+1);
        corners[GetIdx(col, row)] = c;
        view->corners.push_back(corners[GetIdx(col, row)]);
        
        // create corners and edges
        if(patches.at<int>(row, col) != patches.at<int>(row, col+1)) {
          edgels[GetIdx(col, row)].is_corner = true;
          edgels[GetIdx(col, row)].index = GetIdx(col, row);
          edgels[GetIdx(col, row)].horizontal = true;
          edgels[GetIdx(col, row)].h_valid = true;
          edgels[GetIdx(col, row)].h_ids[0] = patches.at<int>(row, col);
          edgels[GetIdx(col, row)].h_ids[1] = patches.at<int>(row, col+1);
        }
        if(patches.at<int>(row+1, col) != patches.at<int>(row+1, col+1)) {
          edgels[GetIdx(col, row+1)].is_corner = true;
          edgels[GetIdx(col, row+1)].index = GetIdx(col, row);
          edgels[GetIdx(col, row+1)].horizontal = true;
          edgels[GetIdx(col, row+1)].h_valid = true;
          edgels[GetIdx(col, row+1)].h_ids[0] = patches.at<int>(row+1, col);
          edgels[GetIdx(col, row+1)].h_ids[1] = patches.at<int>(row+1, col+1);
        }
        
        if(patches.at<int>(row, col) != patches.at<int>(row+1, col)) {
          edgels[GetIdx(col, row)].is_corner = true;
          edgels[GetIdx(col, row)].index = GetIdx(col, row);
          edgels[GetIdx(col, row)].vertical = true;
          edgels[GetIdx(col, row)].v_valid = true;
          edgels[GetIdx(col, row)].v_ids[0] = patches.at<int>(row, col);
          edgels[GetIdx(col, row)].v_ids[1] = patches.at<int>(row+1, col);
        }
        if(patches.at<int>(row, col+1) != patches.at<int>(row+1, col+1)) {
          edgels[GetIdx(col+1, row)].is_corner = true;
          edgels[GetIdx(col+1, row)].index = GetIdx(col, row);
          edgels[GetIdx(col+1, row)].vertical = true;
          edgels[GetIdx(col+1, row)].v_valid = true;
          edgels[GetIdx(col+1, row)].v_ids[0] = patches.at<int>(row, col+1);
          edgels[GetIdx(col+1, row)].v_ids[1] = patches.at<int>(row+1, col+1);
        }
      }
      
      
      else {  // if no corner, create edge
        if(patches.at<int>(row, col) != patches.at<int>(row, col+1)) {
          edgels[GetIdx(col, row)].index = GetIdx(col, row);
          edgels[GetIdx(col, row)].horizontal = true;
          edgels[GetIdx(col, row)].h_valid = true;
          edgels[GetIdx(col, row)].h_ids[0] = patches.at<int>(row, col);
          edgels[GetIdx(col, row)].h_ids[1] = patches.at<int>(row, col+1);
        }
        if(patches.at<int>(row, col) != patches.at<int>(row+1, col)) {
          edgels[GetIdx(col, row)].index = GetIdx(col, row);
          edgels[GetIdx(col, row)].vertical = true;
          edgels[GetIdx(col, row)].v_valid = true;
          edgels[GetIdx(col, row)].v_ids[0] = patches.at<int>(row, col);
          edgels[GetIdx(col, row)].v_ids[1] = patches.at<int>(row+1, col);
        }        
      }
    }
  }

  /// copy corners and edges to view
//   for(int row=0; row<patches.rows-1; row++) {
//     for(int col=0; col<patches.cols-1; col++) {
// //       if(corners[GetIdx(col, row)].index != -1)
// //         view->corners.push_back(corners[GetIdx(col, row)]);
//       if(edges[GetIdx(col, row)].index != -1)
//         view->edgels.push_back(edgels[GetIdx(col, row)]);
//     }
//   }
  
  
  int next_id = 0;
  for(unsigned i=0; i<corners.size(); i++) {
    if(corners[i].index != -1) {
            
      // check all four corner edges
      if(corners[i].ids[0] != corners[i].ids[1]) {
        unsigned id_0 = corners[i].ids[0];
        unsigned id_1 = corners[i].ids[1];
        int x = X(i);
        int y = Y(i);
        if(edgels[GetIdx(x, y)].h_valid) {
printf("start horizontal 1: %u-%u\n", x, y);
          surface::Edge ec;
          ec.index = next_id;
          next_id++;
printf("  ec.index: %u\n", ec.index);
          view->edgels.push_back(edgels[GetIdx(x, y)]);
          edgels[GetIdx(x, y)].h_valid = false;
          ec.edgels_new.push_back(&view->edgels[view->edgels.size()-1]);
          ec.edgels.push_back(view->edgels.size()-1);
          RecursiveClustering(x, y, id_0, id_1, true, ec);
          view->edges.push_back(ec);
        }
      }
      
      if(corners[i].ids[2] != corners[i].ids[3]) {
        unsigned id_0 = corners[i].ids[2];
        unsigned id_1 = corners[i].ids[3];
        int x = X(i);
        int y = Y(i)+1;
        if(edgels[GetIdx(x, y)].h_valid) {
printf("start horizontal 2: %u-%u\n", x, y);
          surface::Edge ec;
          ec.index = next_id;
          next_id++;
printf("  ec.index: %u\n", ec.index);
          view->edgels.push_back(edgels[GetIdx(x, y)]);
          edgels[GetIdx(x, y)].h_valid = false;
          ec.edgels_new.push_back(&view->edgels[view->edgels.size()-1]);
          ec.edgels.push_back(view->edgels.size()-1);
          RecursiveClustering(x, y, id_0, id_1, true, ec);
          view->edges.push_back(ec);
        }
      }
      
      if(corners[i].ids[0] != corners[i].ids[2]) {
        unsigned id_0 = corners[i].ids[0];
        unsigned id_1 = corners[i].ids[2];
        int x = X(i);
        int y = Y(i);
        if(edgels[GetIdx(x, y)].v_valid) {
printf("start vertical 1: %u-%u\n", x, y);
          surface::Edge ec;
          ec.index = next_id;
          next_id++;
printf("  ec.index: %u\n", ec.index);
          view->edgels.push_back(edgels[GetIdx(x, y)]);
          edgels[GetIdx(x, y)].v_valid = false;
          ec.edgels_new.push_back(&view->edgels[view->edgels.size()-1]);
          ec.edgels.push_back(view->edgels.size()-1);
          RecursiveClustering(x, y, id_0, id_1, false, ec);
          view->edges.push_back(ec);
        }
      }
      
      if(corners[i].ids[1] != corners[i].ids[3]) {
        unsigned id_0 = corners[i].ids[1];
        unsigned id_1 = corners[i].ids[3];
        int x = X(i)+1;
        int y = Y(i);
        if(edgels[GetIdx(x, y)].v_valid) {
printf("start vertical 2: %u-%u\n", x, y);
          surface::Edge ec;
          ec.index = next_id;
          next_id++;
printf("  ec.index: %u\n", ec.index);
          view->edgels.push_back(edgels[GetIdx(x, y)]);
          edgels[GetIdx(x, y)].v_valid = false;
          ec.edgels_new.push_back(&view->edgels[view->edgels.size()-1]);
          ec.edgels.push_back(view->edgels.size()-1);
          RecursiveClustering(x, y, id_0, id_1, false, ec);
          view->edges.push_back(ec);
        }
      }
    }
  }
  
  /// TODO Are there still unused edges => Edges without corner point.
  
  /// While edgels are still there:
  ///   create new edge with same corner point.
  ///
  
clock_gettime(CLOCK_THREAD_CPUTIME_ID, &current1);
printf("[CornerDetector::compute] Runtime: %4.3f\n", timespec_diff3(&current1, &start1));
}


void CornerDetector::RecursiveClustering(int x, int y,
                                         int id_0, int id_1, 
                                         bool horizontal, Edge &_ec)
{
  /// HORIZONTAL start
  if(horizontal) {
    
    /// horizontal x, y-1
    if(edgels[GetIdx(x, y-1)].h_valid && edgels[GetIdx(x, y-1)].h_ids[0] == id_0 && edgels[GetIdx(x, y-1)].h_ids[1] == id_1) {
// printf("  h: next horizontal 1: %u-%u of surf: %u-%u\n", x, y-1, id_0, id_1);
      edgels[GetIdx(x, y-1)].h_valid = false;
      view->edgels.push_back(edgels[GetIdx(x, y-1)]);
      _ec.edgels_new.push_back(&view->edgels[view->edgels.size()-1]);
      _ec.edgels.push_back(view->edgels.size()-1);
      if(edgels[GetIdx(x, y-1)].is_corner)
printf("%u-%u => This is now a corner edge => we have to stop here!\n", x, y-1);
      else
        RecursiveClustering(x, y-1, id_0, id_1, true, _ec);
    }
      
    /// horizontal x, y+1
    else if(edgels[GetIdx(x, y+1)].h_valid && edgels[GetIdx(x, y+1)].h_ids[0] == id_0 && edgels[GetIdx(x, y+1)].h_ids[1] == id_1) {
// printf("  h: next horizontal 2: %u-%u of surf: %u-%u\n", x, y+1, id_0, id_1);
      edgels[GetIdx(x, y+1)].h_valid = false;
      view->edgels.push_back(edgels[GetIdx(x, y+1)]);
      _ec.edgels_new.push_back(&view->edgels[view->edgels.size()-1]);
      _ec.edgels.push_back(view->edgels.size()-1);
      if(edgels[GetIdx(x, y+1)].is_corner)
printf("%u-%u => This is now a corner edge => we have to stop here!\n", x, y+1);
      else
        RecursiveClustering(x, y+1, id_0, id_1, true, _ec);
    }
    
    /// vertical x, y-1
    else if(edgels[GetIdx(x, y-1)].v_valid && edgels[GetIdx(x, y-1)].v_ids[0] == id_1 && edgels[GetIdx(x, y-1)].v_ids[1] == id_0) {
// printf("  h: next vertical 1: %u-%u of surf: %u-%u\n", x, y-1, id_0, id_1);
      edgels[GetIdx(x, y-1)].v_valid = false;
      view->edgels.push_back(edgels[GetIdx(x, y-1)]);
      _ec.edgels_new.push_back(&view->edgels[view->edgels.size()-1]);
      _ec.edgels.push_back(view->edgels.size()-1);
      if(edgels[GetIdx(x, y-1)].is_corner)
printf("%u-%u => This is now a corner edge => we have to stop here!\n", x, y-1);
      else
        RecursiveClustering(x, y-1, id_1, id_0, false, _ec);    // changed ids
    }
    
    /// vertical x+1, y-1
    else if(edgels[GetIdx(x+1, y-1)].v_valid && edgels[GetIdx(x+1, y-1)].v_ids[0] == id_0 && edgels[GetIdx(x+1, y-1)].v_ids[1] == id_1) {
// printf("  h: next vertical 2: %u-%u of surf: %u-%u\n", x+1, y-1, id_0, id_1);
      edgels[GetIdx(x+1, y-1)].v_valid = false;
      view->edgels.push_back(edgels[GetIdx(x+1, y-1)]);
      _ec.edgels_new.push_back(&view->edgels[view->edgels.size()-1]);
      _ec.edgels.push_back(view->edgels.size()-1);
      if(edgels[GetIdx(x+1, y-1)].is_corner)
printf("%u-%u => This is now a corner edge => we have to stop here!\n", x+1, y-1);
      else
        RecursiveClustering(x+1, y-1, id_0, id_1, false, _ec);
    }
    
    /// vertical x, y
    else if(edgels[GetIdx(x, y)].v_valid && edgels[GetIdx(x, y)].v_ids[0] == id_0 && edgels[GetIdx(x, y)].v_ids[1] == id_1) {
// printf("  h: next vertical 3: %u-%u of surf: %u-%u\n", x, y, id_0, id_1);
      edgels[GetIdx(x, y)].v_valid = false;
      view->edgels.push_back(edgels[GetIdx(x, y)]);
      _ec.edgels_new.push_back(&view->edgels[view->edgels.size()-1]);
      _ec.edgels.push_back(view->edgels.size()-1);
      if(edgels[GetIdx(x, y)].is_corner)
printf("%u-%u => This is now a corner edge => we have to stop here!\n", x, y);
      else
        RecursiveClustering(x, y, id_0, id_1, false, _ec);
    }
    
    /// vertical x+1, y
    else if(edgels[GetIdx(x+1, y)].v_valid && edgels[GetIdx(x+1, y)].v_ids[0] == id_1 && edgels[GetIdx(x+1, y)].v_ids[1] == id_0) {
// printf("  h: next vertical 4: %u-%u of surf: %u-%u\n", x+1, y, id_0, id_1);
      edgels[GetIdx(x+1, y)].v_valid = false;
      view->edgels.push_back(edgels[GetIdx(x+1, y)]);
      _ec.edgels_new.push_back(&view->edgels[view->edgels.size()-1]);
      _ec.edgels.push_back(view->edgels.size()-1);
      if(edgels[GetIdx(x+1, y)].is_corner)
printf("%u-%u => This is now a corner edge => we have to stop here!\n", x+1, y);
      else
        RecursiveClustering(x+1, y, id_1, id_0, false, _ec);    // changed ids
    }
    else
      printf("  h: no more connection found => stop at %u-%u\n\n", x, y);
  }
  
  
  /// VERTICAL
  else {
    /// vertical x-1, y
    if(edgels[GetIdx(x-1, y)].v_valid && edgels[GetIdx(x-1, y)].v_ids[0] == id_0 && edgels[GetIdx(x-1, y)].v_ids[1] == id_1) {
// printf("  v: next vertical 1: %u-%u of surf: %u-%u\n", x-1, y, id_0, id_1);
      edgels[GetIdx(x-1, y)].v_valid = false;
      view->edgels.push_back(edgels[GetIdx(x-1, y)]);
      _ec.edgels_new.push_back(&view->edgels[view->edgels.size()-1]);
      _ec.edgels.push_back(view->edgels.size()-1);
      if(edgels[GetIdx(x-1, y)].is_corner)
printf("%u-%u => This is now a corner edge => we have to stop here!\n", x-1, y);
      else
        RecursiveClustering(x-1, y, id_0, id_1, false, _ec);
    }
      
    /// vertical x+1, y
    else if(edgels[GetIdx(x+1, y)].v_valid && edgels[GetIdx(x+1, y)].v_ids[0] == id_0 && edgels[GetIdx(x+1, y)].v_ids[1] == id_1) {
// printf("  v: next vertical 2: %u-%u of surf: %u-%u\n", x+1, y, id_0, id_1);
      edgels[GetIdx(x+1, y)].v_valid = false;
      view->edgels.push_back(edgels[GetIdx(x+1, y)]);
      _ec.edgels_new.push_back(&view->edgels[view->edgels.size()-1]);
      _ec.edgels.push_back(view->edgels.size()-1);
      if(edgels[GetIdx(x+1, y)].is_corner)
printf("%u-%u => This is now a corner edge => we have to stop here!\n", x+1, y);
      else
        RecursiveClustering(x+1, y, id_0, id_1, false, _ec);
    }
    
    /// horizontal x-1, y
    else if(edgels[GetIdx(x-1, y)].h_valid && edgels[GetIdx(x-1, y)].h_ids[0] == id_1 && edgels[GetIdx(x-1, y)].h_ids[1] == id_0) {
// printf("  v: next horizontal 1: %u-%u of surf: %u-%u\n", x-1, y, id_0, id_1);
      edgels[GetIdx(x-1, y)].h_valid = false;
      view->edgels.push_back(edgels[GetIdx(x-1, y)]);
      _ec.edgels_new.push_back(&view->edgels[view->edgels.size()-1]);
      _ec.edgels.push_back(view->edgels.size()-1);
      if(edgels[GetIdx(x-1, y)].is_corner)
printf("%u-%u => This is now a corner edge => we have to stop here!\n", x-1, y);
      else
        RecursiveClustering(x-1, y, id_1, id_0, true, _ec);    // changed ids
    }
    
    /// horizontal x-1, y+1
    else if(edgels[GetIdx(x-1, y+1)].h_valid && edgels[GetIdx(x-1, y+1)].h_ids[0] == id_0 && edgels[GetIdx(x-1, y+1)].h_ids[1] == id_1) {
// printf("  v: next horizontal 2: %u-%u of surf: %u-%u\n", x-1, y+1, id_0, id_1);
      edgels[GetIdx(x-1, y+1)].h_valid = false;
      view->edgels.push_back(edgels[GetIdx(x-1, y+1)]);
      _ec.edgels_new.push_back(&view->edgels[view->edgels.size()-1]);
      _ec.edgels.push_back(view->edgels.size()-1);
      if(edgels[GetIdx(x-1, y+1)].is_corner)
printf("%u-%u => This is now a corner edge => we have to stop here!\n", x-1, y+1);
      else
        RecursiveClustering(x-1, y+1, id_0, id_1, true, _ec);
    }
    
    /// horizontal x, y
    else if(edgels[GetIdx(x, y)].h_valid && edgels[GetIdx(x, y)].h_ids[0] == id_0 && edgels[GetIdx(x, y)].h_ids[1] == id_1) {
// printf("  v: next horizontal 3: %u-%u of surf: %u-%u\n", x, y, id_0, id_1);
      edgels[GetIdx(x, y)].h_valid = false;
      view->edgels.push_back(edgels[GetIdx(x, y)]);
      _ec.edgels_new.push_back(&view->edgels[view->edgels.size()-1]);
      _ec.edgels.push_back(view->edgels.size()-1);
      if(edgels[GetIdx(x, y)].is_corner)
printf("%u-%u => This is now a corner edge => we have to stop here!\n", x, y);
      else
        RecursiveClustering(x, y, id_0, id_1, true, _ec);
    }
    
    /// horizontal x, y+1
    else if(edgels[GetIdx(x, y+1)].h_valid && edgels[GetIdx(x, y+1)].h_ids[0] == id_1 && edgels[GetIdx(x, y+1)].h_ids[1] == id_0) {
// printf("  v: next horizontal 4: %u-%u of surf: %u-%u\n", x, y+1, id_0, id_1);
      edgels[GetIdx(x, y+1)].h_valid = false;
      view->edgels.push_back(edgels[GetIdx(x, y+1)]);
      _ec.edgels_new.push_back(&view->edgels[view->edgels.size()-1]);
      _ec.edgels.push_back(view->edgels.size()-1);
      if(edgels[GetIdx(x, y+1)].is_corner)
printf("%u-%u => This is now a corner edge => we have to stop here!\n", x, y+1);
      else
        RecursiveClustering(x, y+1, id_1, id_0, true, _ec);    // changed ids
    }
    
    else
      printf("  v: no more connection found => stop at %u-%u\n\n", x, y);
      /// TODO TODO TODO We have to add here a corner point
      
  }
}


} // end surface












