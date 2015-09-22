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

#ifndef KP_POSE_IO_HPP
#define KP_POSE_IO_HPP

#ifdef Success
#undef Success
#endif

#include <Eigen/Dense>
#include <string>
#include <iostream>
#include <fstream>


namespace v4r
{

/**
 * writePose
 */
inline void writePose(const std::string &filename, const std::string &label, const Eigen::Matrix4f &pose)
{
  std::ofstream out(filename.c_str(), std::ios::out); //ios::app
  if (label.size()!=0) out<<label<<' ';
  for (unsigned v=0; v<4; v++)
    for (unsigned u=0; u<4; u++)
      out<<pose(v,u)<<' ';
  out.close();
}

/**
 * readPose
 */
inline bool readPose(const std::string &filename, std::string &label, Eigen::Matrix4f &pose)
{
  std::ifstream in(filename.c_str(), std::ios::in);
  if (in.is_open())
  {
    in>>label;
    for (unsigned v=0; v<4; v++)
      for (unsigned u=0; u<4; u++)
        in>>pose(v,u);
    in.close();
    return true;
  }
  return false;
}

/**
 * readPose
 */
inline bool readPose(const std::string &filename, Eigen::Matrix4f &pose)
{
  std::ifstream in(filename.c_str(), std::ios::in);
  if (in.is_open())
  {
    for (unsigned v=0; v<4; v++)
      for (unsigned u=0; u<4; u++)
        in>>pose(v,u);
    in.close();
    return true;
  }
  return false;
}

}

#endif
