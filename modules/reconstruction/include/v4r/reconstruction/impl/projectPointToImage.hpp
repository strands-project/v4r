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

#ifndef V4R_PROJECT_POINT_TO_IMAGE_HPP
#define V4R_PROJECT_POINT_TO_IMAGE_HPP


namespace v4r
{

template<typename T1,typename T2, typename T3>
inline void projectPointToImage(T1 p[3], T2 C[9], T3 i[2])
{
  i[0] = C[0] * p[0]/p[2] + C[2];
  i[1] = C[4] * p[1]/p[2] + C[5];
}

template<typename T1,typename T2, typename T3, typename T4>
inline void projectPointToImage(const T1 p[3], const T2 C[9], const T3 D[8], T4 i[2])
{
  double r2, r4, r6, a1, a2, a3, cdist, icdist2;
  double xd, yd;

  double z = p[2] ? 1./p[2] : 1;
  double x = p[0] * z;
  double y = p[1] * z;

  r2 = x*x + y*y;
  r4 = r2*r2;
  r6 = r4*r2;
  a1 = 2*x*y;
  a2 = r2 + 2*x*x;
  a3 = r2 + 2*y*y;
  cdist = 1 + D[0]*r2 + D[1]*r4 + D[4]*r6;
  icdist2 = 1./(1 + D[5]*r2 + D[6]*r4 + D[7]*r6);
  xd = x*cdist*icdist2 + D[2]*a1 + D[3]*a2;
  yd = y*cdist*icdist2 + D[2]*a3 + D[3]*a1;

  i[0] = xd*C[0] + C[2];
  i[1] = yd*C[4] + C[5];
}

} //--END--

#endif




