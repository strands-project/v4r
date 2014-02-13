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
 * @file ConcavityDetector.cpp
 * @author Richtsfeld
 * @date October 2012
 * @version 0.1
 * @brief Detect the concave corners in segments.
 */

#include "ConcavityDetector.h"

namespace surface
{


/************************************************************************************
 * Constructor/Destructor
 */

ConcavityDetector::ConcavityDetector()
{
  have_cloud = false;
  have_view = false;
}

ConcavityDetector::~ConcavityDetector()
{
}

// ================================= Private functions ================================= //



// ================================= Public functions ================================= //



void ConcavityDetector::setInputCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr & _pcl_cloud)
{
  pcl_cloud = _pcl_cloud;
  have_cloud = true;
}


void ConcavityDetector::setView(surface::View *_view)
{
  view = _view;
  have_view = true;
}


double timespec_diff4(struct timespec *x, struct timespec *y)                    /// TODO Remove later
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



void ConcavityDetector::compute()
{ 
  unsigned OFFSET = 10;
  unsigned OFFSET2 = 2*OFFSET;
  unsigned MIN_PTS = 100;
  double MIN_ANGLE = -0.9;      // 155°
                                // in rad 0.44 = 25°
  
  
  if(!have_cloud) {
    printf("[ConcavityDetector::compute] Error: No input cloud set.\n");
    exit(0);
  }

  if(!have_view) {
    printf("[ConcavityDetector::compute] Error: No view set.\n");
    exit(0);
  }

static struct timespec start1, current1;
clock_gettime(CLOCK_THREAD_CPUTIME_ID, &start1);

  for(unsigned i=0; i<view->surfaces.size(); i++) {
    unsigned cnt_size = view->surfaces[i]->contour.size();
    if(cnt_size >= MIN_PTS) {
      unsigned idx_s = 0;
      unsigned idx_m = 0;
      unsigned idx_e = 0;
      double biggest_concave = -1.0;
      unsigned biggest_concave_index = 0;
      double biggest_convex = -1.0;
      unsigned biggest_convex_index = 0;
      for(idx_s=0; idx_s<cnt_size; idx_s++) {
        idx_s = idx_s; 
        idx_m = idx_s + OFFSET;
        if(idx_m >= cnt_size) idx_m -= cnt_size;
        idx_e = idx_s + OFFSET2;
        if(idx_e >= cnt_size) idx_e -= cnt_size;
        
        Eigen::Vector2d vec_s_m, vec_m_e;
        vec_s_m[0] = (double) (X(view->surfaces[i]->contour[idx_s]) - X(view->surfaces[i]->contour[idx_m]));
        vec_s_m[1] = (double) (Y(view->surfaces[i]->contour[idx_s]) - Y(view->surfaces[i]->contour[idx_m]));
        vec_m_e[0] = (double) (X(view->surfaces[i]->contour[idx_e]) - X(view->surfaces[i]->contour[idx_m]));
        vec_m_e[1] = (double) (Y(view->surfaces[i]->contour[idx_e]) - Y(view->surfaces[i]->contour[idx_m]));

        vec_s_m.normalize();
        vec_m_e.normalize();
printf("Vectors: %4.3f / %4.3f --  %4.3f / %4.3f\n", vec_s_m[0], vec_s_m[1], vec_m_e[0], vec_m_e[1]);

        //y2*x1-y1*x2
        bool concave = false;
        bool convex = false;
        double cp = vec_m_e[1]*vec_s_m[0] - vec_s_m[1]*vec_m_e[0];
        if(cp > 0.0)
          concave = true;
        else 
          convex = true;

        double angle = vec_s_m.dot(vec_m_e);              // -1 ... 1 == 180-0°   // -0.9 == 155
        double angle_deg = acos(angle)*180./M_PI;       /// TODO Braucht man eigentlich nicht
        
if(concave) {
  printf("  concave: %u => %4.3f\n", i, cp);
//   printf("  %u: %u-%u-%u => pt: %2.2f-%2.2f / %2.2f-%2.2f => angle %4.3f => %4.3f\n", i, idx_s, idx_m, idx_e, vec_s_m[0], vec_s_m[1], vec_m_e[0], vec_m_e[1], angle, angle_deg);
}
else
  printf("  convex: %u => %4.3f\n", i, cp);
        
        
        
        if(concave) {
          if(angle > biggest_concave && angle > MIN_ANGLE) {
// printf("  => new biggest concave id: %u\n", idx_m);
            biggest_concave = angle;
            biggest_concave_index = idx_m;
          }
        }
        else {
          if(biggest_concave > MIN_ANGLE) {
printf("  ==> save concave %u: %u-%u-%u => pt: %2.2f-%2.2f / %2.2f-%2.2f => angle %4.3f => %4.3f (biggest: %4.3f @ %u)\n", i, idx_s, idx_m, idx_e, vec_s_m[0], vec_s_m[1], vec_m_e[0], vec_m_e[1], angle, angle_deg, biggest_concave, biggest_concave_index);
            view->surfaces[i]->concave.push_back(view->surfaces[i]->contour[biggest_concave_index]);
          }
          biggest_concave = -1.0;
          biggest_concave_index = 0;
        }
        
//         if(convex) {
//           if(angle > biggest_convex && angle > MIN_ANGLE) {
// // printf("  => new biggest convex id: %u\n", idx_m);
//             biggest_convex = angle;
//             biggest_convex_index = idx_m;
//           }
//         }
//         else {
//           if(biggest_convex > MIN_ANGLE) {
// printf("  ==> save convex %u: %u-%u-%u => pt: %2.2f-%2.2f / %2.2f-%2.2f => angle %4.3f => %4.3f (biggest: %4.3f @ %u)\n", i, idx_s, idx_m, idx_e, vec_s_m[0], vec_s_m[1], vec_m_e[0], vec_m_e[1], angle, angle_deg, biggest_convex, biggest_convex_index);
//             view->surfaces[i]->convex.push_back(view->surfaces[i]->contour[biggest_convex_index]);
//           }
//           biggest_convex = -1.0;
//           biggest_convex_index = 0;
//         }
        
      }
    }
  }
  
clock_gettime(CLOCK_THREAD_CPUTIME_ID, &current1);
printf("[ConcavityDetector::compute] Runtime: %4.3f\n", timespec_diff4(&current1, &start1));
}




} // end surface












