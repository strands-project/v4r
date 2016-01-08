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

#include <v4r/keypoints/io.h>

namespace v4r
{
namespace io
{

bool write(const std::string &file, const ArticulatedObject::Ptr &model)
{
  if (model.get()==0)
    return false;

  std::ofstream ofs(file.c_str());
  boost::archive::binary_oarchive oa(ofs);
  oa << model;

  return true;
}


bool read(const std::string &file, ArticulatedObject::Ptr &model)
{
  model.reset(new ArticulatedObject());

  std::ifstream ifs(file.c_str());
  if (ifs.is_open())
  {
    boost::archive::binary_iarchive ia(ifs);
    ia >> model;
    return true;
  }
  return false;
}


}
} //--END--


