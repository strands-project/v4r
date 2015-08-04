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

#ifndef V4R_RANDOM_NUMBERS_HPP
#define V4R_RANDOM_NUMBERS_HPP


namespace v4r
{

inline bool contains(const std::vector<int> &idx, int num)
{
  for (unsigned i=0; i<idx.size(); i++)
    if (idx[i]==num)
      return true;
  return false;
}

/**
 * @brief Returns a pseudo random number in [0.0, 1.0]
 */
inline float frand()
{
  return rand()/((float)RAND_MAX + 1.);
}

inline float expPdf(float lambda)
{
  float dum;
  do
    dum = frand();
  while (dum == 0.);
  return -log(dum)/lambda;
}


/**
 * expSelect
 */
inline int expSelect(int max)
{
  int i;
  /* we want 99% probability of getting with expdev() a number smaller max
   * this requires a lambda of the exponential distribution:
   * lambda = -log(0.01)/max;    (-log(0.01) = 4.6) */
  float lambda = 4.6/(float)max;
  do
    i = (int)(expPdf(lambda));
  while(i > max);
  return i;
}

/**
 * getRandIdx
 */
inline void getRandIdx(int size, int num, std::vector<int> &idx)
{
  int temp;
  idx.clear();
  for (int i=0; i<num; i++)
  {
    do{
      temp = rand()%size;
    }while(contains(idx,temp));
    idx.push_back(temp);
  }
}

/**
 * getExpRandIdx
 */
inline void getExpRandIdx(int size, int num, std::vector<int> &idx)
{
  int temp;
  idx.clear();
  for (int i=0; i<num; i++)
  {
    do{
      temp = expSelect(size);
    }while(contains(idx,temp));
    idx.push_back(temp);
  }
}



} //--END--

#endif




