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

#ifndef V4R_TRIPLE_HPP
#define V4R_TRIPLE_HPP


namespace v4r
{

/**
 * triple
 */
template <class T1, class T2, class T3>
class triple
{
public:
  T1 first;
  T2 second;
  T3 third;

  triple(){}
  triple(const T1 &_f, const T2 &_s, const T3 &_t) : first(_f), second(_s), third(_t) {}
};

/**
 * tripleIIF
 */
class tripleIIF : public triple<int, int, float>
{
public:
  tripleIIF() {}
  tripleIIF(const int &_f, const int &_s, const float &_t) : triple(_f,_s,_t) {}
};

inline bool CmpIncIIF(const tripleIIF &i, const tripleIIF &j)
{
  return (i.third<j.third);
}

inline bool CmpDecIIF(const tripleIIF &i, const tripleIIF &j)
{
  return (i.third>j.third);
}



} //--END--

#endif

